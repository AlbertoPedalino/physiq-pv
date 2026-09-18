"""Download monthly ERA5 NetCDFs: 3-hourly sampling on a 0.5 x 0.5 degree grid.

Select valid times 00/03/06/09/12/15/18/21 UTC; no temporal aggregation.
For ERA5 reanalysis, tp (m) and ssrd (J m-2) retain their one-hour accumulation
ending at each selected valid time. They are NOT three-hour totals.
The spatial grid is requested from CDS; downloaded fields are never re-encoded.

Install: uv sync
Check authentication (no download): python scripts/download_era5.py --check-api
Preview: python scripts/download_era5.py --dry-run
Small live test: python scripts/download_era5.py --test-days 2
Parallel download: python scripts/download_era5.py --max-parallel-requests 8
Credentials: ~/.cdsapirc (and dataset licences accepted on the CDS website).
Ctrl+C stops new monthly requests and retry backoff, then waits for in-flight
CDS calls and file commits to finish. Queued CDS jobs can make shutdown slow.

CDS can split instantaneous/accumulated fields into a ZIP despite 'unarchived'.
Its NetCDF members are extracted byte-for-byte as <monthly-stem>__<member>.nc,
not merged or renamed to a misleading single .nc. A monthly JSON receipt records
the request and SHA256 of every member. Only a fully validated month is committed.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import logging
import math
import os
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import CancelledError, FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from zipfile import ZipFile, is_zipfile

LOG = logging.getLogger("era5")
CATALOGUE = "https://cds.climate.copernicus.eu/api/catalogue/v1/collections/"
GRID = 0.5
TIMES = tuple(f"{hour:02d}:00" for hour in range(0, 24, 3))
DEFAULT_OUTPUT_DIR = Path("/home/apedalino/physiq_pv/data/era5")
DEFAULT_MAX_PARALLEL_REQUESTS = 8
# NetCDF/HDF5 validation uses native libraries that must not run concurrently.
_NETCDF_LOCK = threading.Lock()
_CLIENT_LOCK = threading.Lock()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physiq_pv.era5.features import SINGLE, PRESSURE


@dataclass(frozen=True)
class Product:
    dataset: str
    directory: str
    prefix: str
    variables: dict[str, tuple[str, ...]]
    pressure: bool = False


PRODUCTS = (
    Product("reanalysis-era5-single-levels", "single_levels", "era5_single", SINGLE),
    Product("reanalysis-era5-pressure-levels", "pressure_850", "era5_850", PRESSURE, True),
)


def fetch_json(url: str, attempts: int = 3, delay: float = 10) -> dict | list:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except (OSError, ValueError) as exc:
            if attempt + 1 == attempts:
                raise RuntimeError(f"CDS catalogue unavailable: {url}: {exc}") from exc
            LOG.warning("Catalogue request failed (%s/%s): %s; retrying", attempt + 1, attempts, exc)
            time.sleep(min(60, delay * 2**attempt))
    raise AssertionError("attempts must be positive")


def form_values(form: list, name: str) -> set[str]:
    field = next((field for field in form if field.get("name") == name), None)
    if field is None:
        raise ValueError(f"CDS form has no {name!r} field")
    details = field["details"]
    values = set(details.get("values", []))
    for group in details.get("groups", []):
        values.update(group.get("values", []))
    return values


def inspect_catalogue(fetch=fetch_json) -> dict:
    """Validate exact API identifiers against live official download forms."""
    result = {}
    for product in PRODUCTS:
        metadata = fetch(CATALOGUE + product.dataset)
        form_url = next(link["href"] for link in metadata["links"] if link["rel"] == "form")
        form = fetch(form_url)
        selections = {"variable": set(product.variables), "product_type": {"reanalysis"},
                      "data_format": {"netcdf"}, "download_format": {"unarchived"}}
        if product.pressure:
            selections["pressure_level"] = {"850"}
        for field, requested in selections.items():
            missing = requested - form_values(form, field)
            if missing:
                raise ValueError(f"Invalid CDS {product.dataset}/{field}: {sorted(missing)}")
        intervals = metadata["extent"]["temporal"]["interval"]
        # ERA5 has one continuous interval. Do not guess around gaps or open bounds.
        if len(intervals) != 1 or not all(intervals[0]):
            raise ValueError(f"Cannot establish continuous availability for {product.dataset}")
        start, end = (datetime.fromisoformat(value.replace("Z", "+00:00")) for value in intervals[0])
        result[product.dataset] = {
            "start": start.isoformat(), "end": end.isoformat(),
            "form_url": form_url, "variables": list(product.variables),
            "years": sorted(form_values(form, "year")),
        }
        LOG.info("Verified %s variables: %s; available through %s",
                 len(product.variables), product.dataset, end.date())
    return result


def latest_complete_year(catalogue: dict, today: date | None = None) -> int:
    """Intersect both daily catalogue extents; never include the current year."""
    today = today or datetime.now(timezone.utc).date()
    end = min(date.fromisoformat(item["end"][:10]) for item in catalogue.values())
    # STAC availability is day-granular (00:00 denotes the last available day).
    return min(today.year - 1, end.year if (end.month, end.day) == (12, 31) else end.year - 1)


def validate_area(area: list[float]) -> None:
    north, west, south, east = area
    if not all(math.isfinite(x) for x in area):
        raise ValueError("Area coordinates must be finite")
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180 and east - west < 360):
        raise ValueError("Require -90 <= south < north <= 90 and -180 <= west < east <= 180; no dateline crossing/global wrap")
    if any(not math.isclose(x / GRID, round(x / GRID), abs_tol=1e-8) for x in area):
        raise ValueError(f"Area bounds must be multiples of {GRID} degrees; no implicit rounding")


def grid_shape(area: list[float]) -> tuple[int, int]:
    north, west, south, east = area
    return round((north - south) / GRID) + 1, round((east - west) / GRID) + 1


def make_request(product: Product, year: int, month: int, days: list[int], area: list[float]) -> dict:
    request = {
        "product_type": ["reanalysis"], "variable": list(product.variables),
        "year": [str(year)], "month": [f"{month:02d}"],
        "day": [f"{day:02d}" for day in days],
        "time": list(TIMES), "grid": [GRID, GRID],
        "area": area, "data_format": "netcdf", "download_format": "unarchived",
    }
    # 'time' selects valid times from hourly reanalysis; it does not change the
    # accumulation interval of tp/ssrd. No local sum, mean, or unit conversion.
    if product.pressure:
        request["pressure_level"] = ["850"]
    return request


def expected_times(request: dict):
    import numpy as np
    return np.array([f"{request['year'][0]}-{request['month'][0]}-{day}T{hour}"
                     for day in request["day"] for hour in request["time"]], dtype="datetime64[ns]")


def validate_netcdfs(paths: list[Path], product: Product, request: dict, *, read_payload: bool = True) -> dict:
    with _NETCDF_LOCK:
        return _validate_netcdfs(paths, product, request, read_payload=read_payload)


def _validate_netcdfs(paths: list[Path], product: Product, request: dict, *, read_payload: bool = True) -> dict:
    """Validate all variables/selected timestamps/grid/850 hPa in bounded chunks.

    xarray decoding is used only for inspection. Downloaded bytes are never
    re-encoded. Soil moisture may legitimately be missing over ocean cells.
    """
    import numpy as np
    import xarray as xr

    required = set(product.variables)
    found = set()
    summaries = []
    north, west, south, east = request["area"]
    expected_lat = np.arange(round(south / GRID), round(north / GRID) + 1) * GRID
    expected_lon = np.arange(round(west / GRID), round(east / GRID) + 1) * GRID
    expected_lon = np.sort((expected_lon + 180) % 360 - 180)
    times = expected_times(request)
    for path in paths:
        with xr.open_dataset(path, engine="netcdf4", cache=False) as ds:
            time_name = "valid_time" if "valid_time" in ds.coords else "time"
            if time_name not in ds.coords or ds[time_name].ndim != 1:
                raise ValueError(f"{path.name}: missing one-dimensional time coordinate")
            if not np.array_equal(ds[time_name].values.astype("datetime64[ns]"), times):
                raise ValueError(f"{path.name}: wrong/incomplete/duplicated selected timestamps")
            for coord, expected in (("latitude", expected_lat), ("longitude", expected_lon)):
                if coord not in ds.coords or ds[coord].dims != (coord,):
                    raise ValueError(f"{path.name}: missing regular {coord} axis")
                values = ds[coord].values
                if coord == "longitude":
                    values = (values + 180) % 360 - 180
                if values.shape != expected.shape or not np.allclose(np.sort(values), expected, rtol=0, atol=1e-5):
                    raise ValueError(f"{path.name}: wrong area or non-{GRID}-degree {coord}")
            if product.pressure:
                level = next((name for name in ("pressure_level", "level", "isobaricInhPa") if name in ds.coords), None)
                if level is None:
                    raise ValueError(f"{path.name}: missing pressure coordinate")
                values = np.asarray(ds[level].values).reshape(-1)
                if ds[level].attrs.get("units", "").lower() == "pa":
                    values = values / 100
                if values.size != 1 or not np.allclose(values, [850], rtol=0, atol=1e-6):
                    raise ValueError(f"{path.name}: pressure level is not exclusively 850 hPa")
            time_dim = ds[time_name].dims[0]
            file_found = []
            for variable, aliases in product.variables.items():
                name = next((name for name in (*aliases, variable) if name in ds.data_vars), None)
                if name is None:
                    continue
                if variable in found:
                    raise ValueError(f"Duplicate variable across CDS members: {variable}")
                field = ds[name]
                if not {time_dim, "latitude", "longitude"}.issubset(field.dims):
                    raise ValueError(f"{name}: missing time or spatial dimensions")
                if any(size != 1 and dim not in (time_dim, "latitude", "longitude", "expver")
                       for dim, size in field.sizes.items()):
                    raise ValueError(f"{name}: unexpected non-singleton dimension")
                if read_payload:
                    per_timestamp = math.prod(size for dim, size in field.sizes.items() if dim != time_dim)
                    chunk = max(1, min(24, (32 * 1024**2) // (per_timestamp * 8)))
                    for offset in range(0, len(times), chunk):
                        values = field.isel({time_dim: slice(offset, offset + chunk)}).transpose(time_dim, ...).values
                        if np.isinf(values).any():
                            raise ValueError(f"{name}: infinite values")
                        if variable != "volumetric_soil_water_layer_1" and not np.isfinite(values).reshape(len(values), -1).any(axis=1).all():
                            raise ValueError(f"{name}: entirely missing field at a selected timestamp")
                found.add(variable)
                file_found.append(name)
            if not file_found:
                raise ValueError(f"{path.name}: no requested variables")
            summaries.append({"file": path.name, "sizes": dict(ds.sizes), "variables": file_found})
    if found != required:
        raise ValueError(f"Missing ERA5 variables: {sorted(required - found)}")
    return {"timestamps": len(times), "files": summaries}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_receipt(target: Path, product: Product, request: dict, paths: list[Path], summary: dict) -> None:
    receipt = {"schema_version": 1, "dataset": product.dataset, "request": request,
               "validated_utc": datetime.now(timezone.utc).isoformat(), "validation": summary,
               "files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths]}
    destination = target.with_suffix(".json")
    temporary = destination.with_suffix(".json.part")
    try:
        temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def existing_valid(target: Path, product: Product, request: dict) -> bool:
    receipt_path = target.with_suffix(".json")
    try:
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("schema_version") != 1 or receipt["dataset"] != product.dataset or receipt["request"] != request:
                raise ValueError("saved request differs from requested month/area/grid/times/variables")
            paths = []
            for entry in receipt["files"]:
                name = entry["name"]
                if not re.fullmatch(r"[A-Za-z0-9_.-]+\.nc", name) or not (name == target.name or name.startswith(target.stem + "__")):
                    raise ValueError("invalid receipt member path")
                path = target.parent / name
                if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                    raise ValueError(f"checksum mismatch: {name}")
                paths.append(path)
            if not paths:
                raise ValueError("empty receipt")
            validate_netcdfs(paths, product, request, read_payload=False)
            return True
        if target.is_file():
            # Adopt an existing plain NetCDF only after full validation.
            summary = validate_netcdfs([target], product, request)
            write_receipt(target, product, request, [target], summary)
            return True
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        LOG.warning("RE-DOWNLOAD %s: %s", target.name, exc)
    return False


def unpack_raw(payload: Path, stage: Path, target: Path) -> list[tuple[Path, Path]]:
    if not is_zipfile(payload):
        return [(payload, target)]
    with ZipFile(payload) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if not members or any(not re.fullmatch(r"[A-Za-z0-9_.-]+\.nc", member.filename) for member in members):
            raise ValueError("CDS archive must contain only flat NetCDF members")
        if len({member.filename for member in members}) != len(members):
            raise ValueError("Duplicate ZIP member names")
        pairs = []
        for member in members:
            final = target if len(members) == 1 else target.with_name(target.stem + "__" + member.filename)
            part = stage / (final.name + ".part")
            with archive.open(member) as source, part.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024**2)
            pairs.append((part, final))
    return pairs


def download_month(client, product: Product, request: dict, target: Path, *, force=False,
                   attempts=4, retry_delay=10, stop_event: threading.Event | None = None) -> str:
    if stop_event is not None and stop_event.is_set():
        raise CancelledError("Downloader stopping")
    target.parent.mkdir(parents=True, exist_ok=True)
    # OS file lock releases on Ctrl+C, exceptions, or process death; the tiny lock
    # file is retained to avoid inode races. Prevent concurrent writers per month.
    from contextlib import ExitStack
    with ExitStack() as stack:
        lock = stack.enter_context(target.with_suffix(".lock").open("a+b"))
        if os.name == "nt":
            import msvcrt
            if lock.seek(0, os.SEEK_END) == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Another downloader is writing {target}") from exc
        else:
            import fcntl
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(f"Another downloader is writing {target}") from exc
        if not force and existing_valid(target, product, request):
            LOG.info("SKIP verified %s", target)
            return "skipped"
        for attempt in range(attempts):
            if stop_event is not None and stop_event.is_set():
                raise CancelledError("Downloader stopping")
            try:
                # Only this invocation's staging directory is ever cleaned up.
                with tempfile.TemporaryDirectory(prefix=target.stem + ".", suffix=".part", dir=target.parent) as staging:
                    stage = Path(staging)
                    payload = stage / "response.part"
                    LOG.info("DOWNLOAD %s (%s/%s)", target, attempt + 1, attempts)
                    client.retrieve(product.dataset, request, str(payload))
                    pairs = unpack_raw(payload, stage, target)
                    summary = validate_netcdfs([part for part, _ in pairs], product, request)
                    for part, final in pairs:
                        os.replace(part, final)
                    paths = [final for _, final in pairs]
                    write_receipt(target, product, request, paths, summary)
                LOG.info("OK %s: %s timestamps; %s raw NetCDF file(s)", target.stem, summary["timestamps"], len(paths))
                return "downloaded"
            except KeyboardInterrupt:
                LOG.warning("Interrupted %s; uncommitted temporary download removed", target.stem)
                raise
            except Exception as exc:
                # Auth/licence/request errors require user action, not repeated jobs.
                # CDS may report queue/rate limits as 400/403/422, not only 429.
                status = getattr(getattr(exc, "response", None), "status_code", None)
                throttled = status == 429 or (
                    status in (400, 403, 422) and re.search(
                        r"too many (?:requests|jobs)|rate[ _-]?limit|throttl|"
                        r"queue[ _-]+(?:is[ _-]+)?(?:full|limit)|"
                        r"(?:maximum|concurrent).*(?:requests|jobs)|"
                        r"limit.*concurrent", str(exc), re.IGNORECASE
                    ) is not None
                )
                if (status in (400, 401, 403, 404, 422) and not throttled) or attempt + 1 == attempts:
                    raise RuntimeError(f"Download failed for {target}: {exc}") from exc
                pause = min(60, retry_delay * 2**attempt)
                LOG.warning("Attempt failed for %s: %s; retry in %ss", target.name, exc, pause)
                if stop_event is None:
                    time.sleep(pause)
                elif stop_event.wait(pause):
                    raise CancelledError("Downloader stopping") from exc
    raise AssertionError("attempts must be positive")


def create_client():
    # Never read or log the token ourselves. cdsapi reads the user's config.
    if not (Path.home() / ".cdsapirc").is_file():
        raise RuntimeError("Missing ~/.cdsapirc. Configure your CDS token outside the repository and accept both dataset licences at https://cds.climate.copernicus.eu/how-to-api")
    try:
        import cdsapi
        import requests
    except ImportError as exc:
        raise RuntimeError("Install the downloader dependency: uv sync (or python -m pip install 'cdsapi>=0.7.7')") from exc
    try:
        with _CLIENT_LOCK:
            # Keep stable handlers: CDS otherwise installs/removes handlers while
            # logging, which races across clients. Records propagate to our root.
            for name in ("cdsapi", "ecmwf.datastores.legacy_client"):
                logger = logging.getLogger(name)
                if not logger.handlers:
                    logger.addHandler(logging.NullHandler())
            # Explicit sessions also isolate older CDS clients whose default
            # session is shared. Concurrent progress bars would interleave.
            session = requests.Session()
            try:
                return cdsapi.Client(timeout=120, retry_max=3, sleep_max=30, quiet=False,
                                     progress=False, debug=False, session=session,
                                     info_callback=LOG.info, warning_callback=LOG.warning,
                                     error_callback=LOG.error, debug_callback=LOG.debug)
            except Exception:
                session.close()
                raise
    except Exception:
        # Do not echo parser errors: they can contain configuration/credential text.
        raise RuntimeError("Cannot initialise CDS API. Check url and key in ~/.cdsapirc; see https://cds.climate.copernicus.eu/how-to-api") from None


def check_api(client) -> None:
    """Authenticate through the CDS profile API, without submitting any jobs."""
    backend = getattr(client, "client", None)
    if backend is None or not callable(getattr(backend, "check_authentication", None)):
        raise RuntimeError("CDS API authentication check requires the current CDS URL and personal access token in ~/.cdsapirc; see https://cds.climate.copernicus.eu/how-to-api")
    try:
        backend.check_authentication()
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        detail = f" (HTTP {status})" if isinstance(status, int) else ""
        raise RuntimeError(
            f"CDS API check failed{detail}. Verify url and key in ~/.cdsapirc "
            "and network access to CDS."
        ) from None
    print("CDS API authentication OK. No download submitted. Dataset licences must be accepted separately on the CDS website.")


def free_disk(path: Path) -> int:
    path = path.resolve()
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def download_parallel(jobs: list[tuple[Product, dict, Path]], *,
                      max_parallel_requests: int = DEFAULT_MAX_PARALLEL_REQUESTS,
                      force=False, attempts=4, retry_delay=10) -> dict[str, int]:
    """Run independent monthly jobs, with one private CDS client per worker.

    active includes retrieval, validation and retry backoff; pending counts jobs
    not yet dispatched. Only the coordinator updates counters/progress. Logging
    handlers serialize worker records; monthly receipts retain their file locks.
    Ctrl+C stops dispatch/retries and drains in-flight calls and file commits.
    """
    if max_parallel_requests < 1:
        raise ValueError("Require max-parallel-requests >= 1")
    counts = {"skipped": 0, "downloaded": 0}
    stop_event = threading.Event()
    local = threading.local()

    def execute(job):
        if stop_event.is_set():
            raise CancelledError("Downloader stopping")
        if not hasattr(local, "client"):
            local.client = create_client()
        product, request, target = job
        return download_month(local.client, product, request, target, force=force,
                              attempts=attempts, retry_delay=retry_delay, stop_event=stop_event)

    executor = ThreadPoolExecutor(max_workers=max_parallel_requests, thread_name_prefix="era5")
    futures = set()
    submitted = 0

    def progress():
        LOG.info("Requests: active=%s pending=%s downloaded=%s skipped=%s",
                 len(futures), len(jobs) - submitted, counts["downloaded"], counts["skipped"])

    try:
        while submitted < len(jobs) or futures:
            while submitted < len(jobs) and len(futures) < max_parallel_requests:
                futures.add(executor.submit(execute, jobs[submitted]))
                submitted += 1
            progress()
            completed, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
            for future in completed:
                counts[future.result()] += 1
                futures.remove(future)
        progress()
    except BaseException:
        stop_event.set()
        for future in futures:
            future.cancel()
        LOG.warning("Stopping: active=%s pending=%s; no new requests or retries. "
                    "Waiting for in-flight CDS calls and file commits to finish.",
                    sum(not future.done() for future in futures), len(jobs) - submitted)
        raise
    finally:
        # A second Ctrl+C must not interrupt shutdown while workers commit files.
        # Python cannot safely interrupt a thread blocked inside cdsapi.retrieve.
        previous = None
        if threading.current_thread() is threading.main_thread():
            previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        finally:
            if previous is not None:
                signal.signal(signal.SIGINT, previous)
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", default="latest", help="Last complete year (integer or latest)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Destination (default: {DEFAULT_OUTPUT_DIR.as_posix()})")
    for name, default in (("north", 60), ("west", -15), ("south", 20), ("east", 50)):
        parser.add_argument("--" + name, type=float, default=default)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Query public metadata and print plan; no download, credentials, or output writes")
    mode.add_argument("--check-api", action="store_true", help="Verify CDS authentication using ~/.cdsapirc and exit; no download or output writes")
    parser.add_argument("--force", action="store_true", help="Re-download even valid months; keep existing files until replacements validate")
    parser.add_argument("--test-days", type=int, choices=(1, 2), help="ONLY January 1 (or 1-2), 1980; isolated under OUTPUT/test_1980_01_DD")
    parser.add_argument("--months", nargs="+", type=int, choices=range(1, 13), default=list(range(1, 13)))
    parser.add_argument("--attempts", type=int, default=4, help="Maximum download attempts per month")
    parser.add_argument("--max-parallel-requests", type=int, default=DEFAULT_MAX_PARALLEL_REQUESTS,
                        help="Maximum concurrent monthly CDS requests (default: 8)")
    parser.add_argument("--retry-delay", type=float, default=10, help="Initial retry delay in seconds, exponentially increased to 60")
    return parser


def run(args) -> int:
    if args.check_api:
        check_api(create_client())
        return 0
    area = [args.north, args.west, args.south, args.east]
    validate_area(area)
    if args.attempts < 1 or not math.isfinite(args.retry_delay) or args.retry_delay < 0:
        raise ValueError("Require attempts >= 1 and finite retry-delay >= 0")
    if args.max_parallel_requests < 1:
        raise ValueError("Require max-parallel-requests >= 1")
    if not args.dry_run:
        create_client()  # Fail on missing/invalid configuration before network access.
    catalogue = inspect_catalogue(fetch=lambda url: fetch_json(url, args.attempts, args.retry_delay))
    latest = latest_complete_year(catalogue)
    if args.test_days:
        start = end = 1980
        months = [1]
        output = args.output_dir / f"test_1980_01_{args.test_days:02d}"
    else:
        start = args.start_year
        end = latest if args.end_year == "latest" else int(args.end_year)
        months = sorted(set(args.months))
        output = args.output_dir
    if not 1940 <= start <= end <= latest:
        raise ValueError(f"Require 1940 <= start-year <= end-year <= {latest} (last complete year available in both products)")
    for item in catalogue.values():
        if not set(map(str, range(start, end + 1))) <= set(item["years"]):
            raise ValueError("Requested years are not all present in both CDS forms")
    schedule = [(year, month, list(range(1, (args.test_days or calendar.monthrange(year, month)[1]) + 1)))
                for year in range(start, end + 1) for month in months]
    timestamps = sum(len(days) * len(TIMES) for _, _, days in schedule)
    nlat, nlon = grid_shape(area)
    payload_bytes = timestamps * nlat * nlon * (len(SINGLE) + len(PRESSURE)) * 4
    available = free_disk(output)
    print(f"Years: {start}-{end}; latest complete common year: {latest}")
    print(f"Area [N,W,S,E]: {area}; grid: {GRID} degrees; cells: {nlat} x {nlon} = {nlat*nlon:,}")
    print(f"Temporal sampling: every 3 hours; {len(TIMES)} timestamps/day (UTC: {', '.join(TIMES)})")
    print("No temporal aggregation: tp/ssrd keep the one-hour accumulation ending at each selected valid time")
    print(f"Months: {len(schedule)}; timestamps: {timestamps:,}; CDS requests: {2*len(schedule)}")
    print(f"Maximum parallel requests: {args.max_parallel_requests}")
    print(f"NetCDF files: approximately {2*len(schedule)}-{3*len(schedule)} (CDS may split stepTypes)")
    print(f"Output: {output.resolve()}")
    print(f"Indicative uncompressed float32 payload: {payload_bytes/1e12:.3f} TB ({payload_bytes/1024**4:.3f} TiB; {payload_bytes/1024**2:,.1f} MiB)")
    print(f"Planning allowance (+25%): {payload_bytes*1.25/1e12:.3f} TB; actual compression/packing may differ")
    print(f"Free disk: {available/1e12:.3f} TB ({available/1024**3:.1f} GiB); staging needs extra space for up to {min(args.max_parallel_requests, 2*len(schedule))} requests and extraction", flush=True)
    if available < payload_bytes * 1.25:
        LOG.warning("Free space below full-plan estimate (existing valid months are not deducted)")
    if args.dry_run:
        return 0
    jobs = []
    for year, month, days in schedule:
        for product in PRODUCTS:
            target = output / product.directory / str(year) / f"{product.prefix}_{year}_{month:02d}.nc"
            request = make_request(product, year, month, days, area)
            jobs.append((product, request, target))
    counts = download_parallel(jobs, max_parallel_requests=args.max_parallel_requests,
                               force=args.force, attempts=args.attempts, retry_delay=args.retry_delay)
    LOG.info("Finished: %s", counts)
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        LOG.warning("Stopped by Ctrl+C. Completed months remain valid; rerun the same command to resume.")
        return 130
    except Exception as exc:
        LOG.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
