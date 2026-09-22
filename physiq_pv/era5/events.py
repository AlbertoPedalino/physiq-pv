"""Disk-backed cluster graph and event trajectories, streaming over score frames."""
from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

from .cube import CubeGrid, disk_percentile, morphology, score_cube_chunks, spatial_clusters


@dataclass(frozen=True)
class EventConfig:
    top_percent: float = 1.0
    absolute_threshold: float | None = None
    threshold_scope: str = "global"  # or frame
    kernel_size: int = 3
    opening_iterations: int = 1
    closing_iterations: int = 1
    min_cells: int = 1
    link_policy: str = "dilated_overlap"  # overlap, centroid, any
    min_overlap: float = 0.1
    dilation_cells: int = 1
    centroid_distance_km: float = 100.0
    timestep_hours: float = 3.0
    chunk_size: int = 32

    def __post_init__(self):
        if not 0 < self.top_percent < 100 or self.threshold_scope not in ("global", "frame"):
            raise ValueError("Invalid percentile threshold")
        if self.absolute_threshold is not None and not np.isfinite(self.absolute_threshold):
            raise ValueError("Absolute threshold must be finite")
        for name in ("kernel_size", "min_cells", "chunk_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("opening_iterations", "closing_iterations", "dilation_cells"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.kernel_size % 2 != 1 or self.link_policy not in ("overlap", "dilated_overlap", "centroid", "any"):
            raise ValueError("Invalid morphology/link policy")
        if not 0 < self.min_overlap <= 1:
            raise ValueError("min_overlap must be in (0,1]")
        if not np.isfinite(self.centroid_distance_km) or not 0 <= self.centroid_distance_km <= 20000:
            raise ValueError("Invalid centroid distance")
        if not np.isfinite(self.timestep_hours) or self.timestep_hours <= 0:
            raise ValueError("Invalid timestep_hours")


def _xyz(clusters):
    latitude = np.deg2rad([c["centroid_lat"] for c in clusters])
    longitude = np.deg2rad([c["centroid_lon"] for c in clusters])
    return np.column_stack((np.cos(latitude)*np.cos(longitude),
                            np.cos(latitude)*np.sin(longitude), np.sin(latitude)))


def link_clusters(previous, current, current_labels, config: EventConfig):
    """Return all accepted edges; never force one-to-one tracking at split/merge."""
    if not previous or not current:
        return []
    h, w = current_labels.shape
    by_label = {cluster["label"]: j for j, cluster in enumerate(current)}
    candidates = set()
    expanded = {}
    for i, cluster in enumerate(previous):
        cells = cluster["flat_cells"]
        if config.link_policy in ("dilated_overlap", "any") and config.dilation_cells:
            rr, cc = cells // w, cells % w
            radius = config.dilation_cells
            r0, r1 = max(0, int(rr.min())-radius), min(h, int(rr.max())+radius+1)
            c0, c1 = max(0, int(cc.min())-radius), min(w, int(cc.max())+radius+1)
            patch = np.zeros((r1-r0, c1-c0), dtype=bool)
            patch[rr-r0, cc-c0] = True
            dr, dc = np.nonzero(ndimage.binary_dilation(patch, np.ones((3,3)), iterations=radius))
            cells = (dr+r0)*w + dc+c0
        expanded[i] = cells
        if config.link_policy != "centroid":
            candidates.update((i, by_label[int(label)]) for label in np.unique(current_labels.ravel()[cells])
                              if int(label) in by_label)
    pxyz, cxyz = _xyz(previous), _xyz(current)
    if config.link_policy in ("centroid", "any"):
        chord = 2*np.sin(config.centroid_distance_km/(2*6371.0088))
        for i, neighbours in enumerate(cKDTree(pxyz).query_ball_tree(cKDTree(cxyz), chord)):
            candidates.update((i, j) for j in neighbours)
    edges = []
    for i, j in sorted(candidates):
        before, after = previous[i]["flat_cells"], current[j]["flat_cells"]
        overlap = len(np.intersect1d(before, after, assume_unique=True)) / min(len(before), len(after))
        dilated = len(np.intersect1d(expanded[i], after, assume_unique=True)) / min(len(expanded[i]), len(after))
        distance = 2*6371.0088*np.arcsin(np.clip(np.linalg.norm(pxyz[i]-cxyz[j])/2, 0, 1))
        accepted = {"overlap": overlap >= config.min_overlap,
                    "dilated_overlap": dilated >= config.min_overlap,
                    "centroid": distance <= config.centroid_distance_km}
        if (any(accepted.values()) if config.link_policy == "any" else accepted[config.link_policy]):
            edges.append((i, j, overlap, dilated, float(distance)))
    return edges


CLUSTER_COLUMNS = ("cluster_id", "event_id", "time_index", "timestamp", "timestamp_ns",
                   "cells", "area_km2", "centroid_lat", "centroid_lon", "north", "south",
                   "west", "east", "mean_score", "max_score", "mean_uncertainty", "max_uncertainty")


def _database(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA cache_size=-32768")
    db.execute("PRAGMA temp_store=FILE")
    db.executescript("""
        CREATE TABLE roots(event_id INTEGER PRIMARY KEY, parent INTEGER NOT NULL);
        CREATE TABLE frames(time_index INTEGER PRIMARY KEY, timestamp TEXT, threshold REAL);
        CREATE TABLE clusters(cluster_id INTEGER PRIMARY KEY, event_id INTEGER,
            time_index INTEGER, timestamp TEXT, timestamp_ns INTEGER, cells INTEGER,
            area_km2 REAL, centroid_lat REAL, centroid_lon REAL,
            north REAL, south REAL, west REAL, east REAL, mean_score REAL, max_score REAL,
            mean_uncertainty REAL, max_uncertainty REAL);
        CREATE INDEX cluster_event ON clusters(event_id);
        CREATE TABLE links(source_cluster INTEGER, target_cluster INTEGER,
            overlap REAL, dilated_overlap REAL, centroid_distance_km REAL, relation TEXT,
            PRIMARY KEY(source_cluster,target_cluster));
    """)
    return db


def _root(db, event_id):
    path = []
    while True:
        parent = db.execute("SELECT parent FROM roots WHERE event_id=?", (event_id,)).fetchone()[0]
        if parent == event_id:
            break
        path.append(event_id)
        event_id = parent
    for child in path:
        db.execute("UPDATE roots SET parent=? WHERE event_id=?", (event_id, child))
    return event_id


def _finish(db, timestep_hours):
    # Canonicalize historical clusters after merges. Indexed updates and an
    # on-disk union forest avoid retaining the entire event history in RAM.
    for (old,) in db.execute("SELECT event_id FROM roots ORDER BY event_id"):
        root = _root(db, old)
        if old != root:
            db.execute("UPDATE clusters SET event_id=? WHERE event_id=?", (root, old))
    db.executescript("""
        CREATE TABLE event_steps AS SELECT event_id, time_index, timestamp,
            COUNT(*) AS n_clusters, SUM(cells) AS cells, SUM(area_km2) AS area_km2,
            SUM(centroid_lat*area_km2)/SUM(area_km2) AS centroid_lat,
            SUM(centroid_lon*area_km2)/SUM(area_km2) AS centroid_lon,
            SUM(mean_score*cells)/SUM(cells) AS mean_score, MAX(max_score) AS max_score,
            SUM(mean_uncertainty*cells)/SUM(cells) AS mean_uncertainty,
            MAX(max_uncertainty) AS max_uncertainty
            FROM clusters GROUP BY event_id,time_index ORDER BY event_id,time_index;
        CREATE INDEX step_event ON event_steps(event_id,time_index);
        CREATE TABLE events AS SELECT event_id, MIN(timestamp) AS start,
            MAX(timestamp) AS end, (MAX(timestamp_ns)-MIN(timestamp_ns))/3600000000000.0 AS elapsed_hours,
            COUNT(DISTINCT time_index) AS n_timestamps, COUNT(*) AS n_clusters,
            SUM(cells) AS cell_observations, SUM(mean_score*cells)/SUM(cells) AS mean_score,
            MAX(max_score) AS max_score,
            SUM(mean_uncertainty*cells)/SUM(cells) AS mean_uncertainty,
            MAX(max_uncertainty) AS max_uncertainty
            FROM clusters GROUP BY event_id ORDER BY event_id;
        ALTER TABLE events ADD COLUMN duration_hours REAL;
        CREATE UNIQUE INDEX event_id_index ON events(event_id);
    """)
    db.execute("UPDATE events SET duration_hours=elapsed_hours+?", (timestep_hours,))
    db.commit()


def _export(db, table, path):
    cursor = db.execute(f"SELECT * FROM {table}")  # table names are internal constants
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([c[0] for c in cursor.description])
        writer.writerows(cursor)


def process_events(scores, timestamps, grid: CubeGrid, output_dir, config=None, *, uncertainty=None):
    """Map (T,N) scores to disk (T,H,W), morphology, clusters, and event graph.

    `scores` may be a memmap. Only two frames' clusters remain active across chunk
    boundaries. Percentiles use the complete finite score population by default.
    Event IDs are connected components of the temporal cluster graph; split
    branches retain one event and merges unify all ancestor IDs retrospectively.
    `scores` is anomaly_mean; uncertainty (anomaly_std) annotates clusters only.
    Event uncertainty is cell-observation-weighted, including all split branches.
    """
    config = config or EventConfig()
    timestamps = pd.DatetimeIndex(timestamps).as_unit("ns")
    if timestamps.hasnans or len(timestamps) != len(scores) or not len(scores) or np.any(np.diff(timestamps.asi8) <= 0):
        raise ValueError("Scores require matching strictly increasing timestamps")
    if scores.ndim != 2 or scores.shape[1] != len(grid.rows):
        raise ValueError("Expected scores (T,N) in grid location order")
    if uncertainty is not None and uncertainty.shape != scores.shape:
        raise ValueError("Uncertainty must match scores (T,N)")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty event output directory")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"status": "running", "config": asdict(config), "grid": grid.to_dict(),
                "shape": [len(scores), *grid.shape], "score_dtype": str(scores.dtype),
                "anomaly_mean_cube_file": "anomaly_mean_cube.npy",
                "uncertainty_cube_file": "uncertainty_cube.npy",
                "uncertainty_available": uncertainty is not None,
                "uncertainty_policy": "cell-observation-weighted mean and maximum; never used for detection/linking",
                "threshold_comparison": "strictly greater; ties can reduce top-percent fraction",
                "event_policy": "connected components of consecutive-time cluster graph; retrospective split/merge union",
                "duration_policy": "end-start plus one timestep; also export elapsed_hours",
                "area_policy": "spherical cell area in km2; centroids weighted by area"}
    manifest = output / "metadata.json"
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    cutoff = config.absolute_threshold
    if cutoff is None and config.threshold_scope == "global":
        cutoff = disk_percentile(scores, 100-config.top_percent, output, config.chunk_size)
    metadata["global_threshold"] = cutoff
    cube = np.lib.format.open_memmap(output/"anomaly_mean_cube.npy", mode="w+",
        dtype=np.result_type(scores.dtype, np.float32), shape=(len(scores), *grid.shape))
    uncertainty_cube = np.lib.format.open_memmap(output/"uncertainty_cube.npy", mode="w+",
        dtype=np.float32, shape=cube.shape)
    labels_store = np.lib.format.open_memmap(output/"cluster_labels.npy", mode="w+", dtype=np.int64, shape=cube.shape)
    np.save(output/"timestamps.npy", timestamps.asi8)
    db = _database(output/"events.sqlite")
    previous, counter = [], 0
    try:
        for start, block in score_cube_chunks(scores, grid, config.chunk_size):
            cube[start:start+len(block)] = block
            for offset, frame in enumerate(block):
                index = start+offset
                uncertainty_frame = None if uncertainty is None else grid.frame(uncertainty[index])
                uncertainty_cube[index] = np.nan if uncertainty_frame is None else uncertainty_frame
                threshold = cutoff
                if threshold is None:
                    finite = frame[np.isfinite(frame)]
                    threshold = float(np.percentile(finite, 100-config.top_percent)) if finite.size else float("inf")
                _, _, binary = morphology(frame, threshold, kernel_size=config.kernel_size,
                    opening_iterations=config.opening_iterations, closing_iterations=config.closing_iterations)
                labels, current = spatial_clusters(frame, binary, grid, min_cells=config.min_cells,
                                                   uncertainty=uncertainty_frame)
                if index and timestamps.asi8[index]-timestamps.asi8[index-1] != pd.Timedelta(hours=config.timestep_hours).value:
                    previous = []  # explicit time gap: no bridging
                edges = link_clusters(previous, current, labels, config)
                parents = {}
                for i, j, *_ in edges:
                    parents.setdefault(j, []).append(previous[i]["event_id"])
                db.execute("INSERT INTO frames VALUES(?,?,?)", (index, timestamps[index].isoformat(), threshold))
                output_labels = np.zeros(grid.shape, dtype=np.int64)
                for j, cluster in enumerate(current):
                    counter += 1
                    roots = sorted({_root(db, parent) for parent in parents.get(j, [])})
                    event_id = roots[0] if roots else counter
                    if not roots:
                        db.execute("INSERT INTO roots VALUES(?,?)", (event_id, event_id))
                    for old in roots[1:]:
                        db.execute("UPDATE roots SET parent=? WHERE event_id=?", (event_id, old))
                    cluster.update(cluster_id=counter, event_id=event_id, time_index=index,
                                   timestamp=timestamps[index].isoformat(), timestamp_ns=int(timestamps.asi8[index]))
                    db.execute("INSERT INTO clusters VALUES("+",".join("?" for _ in CLUSTER_COLUMNS)+")",
                               tuple(cluster[key] for key in CLUSTER_COLUMNS))
                    output_labels.ravel()[cluster["flat_cells"]] = counter
                outgoing = Counter(i for i, *_ in edges)
                incoming = Counter(j for _, j, *_ in edges)
                for i, j, overlap, dilated, distance in edges:
                    relation = ("split_merge" if outgoing[i] > 1 and incoming[j] > 1 else
                                "split" if outgoing[i] > 1 else "merge" if incoming[j] > 1 else "continue")
                    db.execute("INSERT INTO links VALUES(?,?,?,?,?,?)",
                               (previous[i]["cluster_id"], current[j]["cluster_id"], overlap, dilated, distance, relation))
                labels_store[index] = output_labels
                previous = current
            db.commit()
            cube.flush()
            uncertainty_cube.flush()
            labels_store.flush()
        _finish(db, config.timestep_hours)
        for table in ("frames", "clusters", "links", "events", "event_steps"):
            _export(db, table, output/(table+".csv"))
        metadata.update(status="complete", n_clusters=counter,
                        n_events=db.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    finally:
        db.close()
        cube._mmap.close()
        uncertainty_cube._mmap.close()
        labels_store._mmap.close()
    return metadata


def event_trajectory(output_dir, event_id):
    """Load just one event trajectory for interactive analysis."""
    with closing(sqlite3.connect(Path(output_dir)/"events.sqlite")) as db:
        return pd.read_sql_query("SELECT * FROM event_steps WHERE event_id=? ORDER BY time_index", db, params=(int(event_id),))
