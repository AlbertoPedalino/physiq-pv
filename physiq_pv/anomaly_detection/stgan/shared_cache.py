"""A normalised copy of the data, built once and read by every run with the same inputs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from .data import ContextArray, _array_state

MANIFEST = "cache.json"


def source_identity(array) -> list[dict]:
    """The files an array is read from: path, size, time of the last change and the part that is read."""
    if isinstance(array, ContextArray):
        return source_identity(array.prefix) + source_identity(array.test)
    state = _array_state(array)
    if not isinstance(state, tuple):
        raise ValueError("A shared cache needs data read from files (memory-mapped arrays).")
    filename, dtype, shape, offset = state
    status = os.stat(filename)
    return [{"file": str(Path(filename).resolve()), "bytes": status.st_size, "modified_ns": status.st_mtime_ns,
             "dtype": str(dtype), "shape": list(shape), "offset": int(offset)}]


def shared_cache(root, inputs: dict, build):
    """Directory of the cache of `inputs` under `root`, what its build returned, and whether it was already there.

    `inputs` names everything the content depends on; its hash names the directory, so caches of
    different inputs live side by side and none is ever rewritten. `build(directory)` writes the
    files and returns what a later run needs to know (JSON). The cache is written under a
    temporary name and renamed when complete: an interrupted build is never read, and of two runs
    that build at the same time the second keeps the cache of the first.
    """
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    root = Path(root)
    directory, manifest = root / key, root / key / MANIFEST
    if manifest.exists():
        return directory, json.loads(manifest.read_text(encoding="utf-8"))["built"], True
    temporary = root / f".{key}.building-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    temporary.mkdir(parents=True)
    try:
        built = build(temporary)
        (temporary / MANIFEST).write_text(json.dumps({"inputs": inputs, "built": built}, indent=2), encoding="utf-8")
        try:
            temporary.rename(directory)
        except OSError:
            if not manifest.exists():
                raise
            shutil.rmtree(temporary)  # another run completed the same cache first
            return directory, json.loads(manifest.read_text(encoding="utf-8"))["built"], True
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return directory, built, False
