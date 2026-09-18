"""Frame and trajectory plots: read only selected frames from disk-backed cubes."""
from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from .cube import CubeGrid, morphology
from .events import EventConfig, event_trajectory, process_events


def _frame(path, index):
    array = np.load(path,mmap_mode="r")
    try:
        return np.array(array[index])
    finally:
        array._mmap.close()


def plot_frame(directory, time_index=0):
    import matplotlib.pyplot as plt
    directory = Path(directory)
    metadata = json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
    grid = CubeGrid(**metadata["grid"])
    config = EventConfig(**metadata["config"])
    frame = _frame(directory/"anomaly_cube.npy",time_index)
    labels = _frame(directory/"cluster_labels.npy",time_index)
    with closing(sqlite3.connect(directory/"events.sqlite")) as db:
        row = db.execute("SELECT timestamp,threshold FROM frames WHERE time_index=?",(int(time_index),)).fetchone()
        clusters = pd.read_sql_query("SELECT * FROM clusters WHERE time_index=?",db,params=(int(time_index),))
    if row is None:
        raise IndexError(time_index)
    raw, opened, closed = morphology(frame,row[1],kernel_size=config.kernel_size,
        opening_iterations=config.opening_iterations,closing_iterations=config.closing_iterations)
    fig, axes = plt.subplots(1,5,figsize=(20,4),layout="constrained")
    half = grid.spacing/2
    extent = [grid.longitudes[0]-half,grid.longitudes[-1]+half,grid.latitudes[-1]-half,grid.latitudes[0]+half]
    for axis, values, title in zip(axes,(frame,raw,opened,closed,np.ma.masked_equal(labels,0)),
                                   ("Anomaly score","Threshold","Opening","Closing","Cluster ID")):
        plot = axis.imshow(values,origin="upper",extent=extent,aspect="auto",interpolation="nearest")
        axis.set(title=title,xlabel="Longitude",ylabel="Latitude")
        fig.colorbar(plot,ax=axis,shrink=.7)
    fig.suptitle(f"{row[0]} | threshold > {row[1]:.6g}")
    return fig, clusters


def plot_event(directory, event_id):
    import matplotlib.pyplot as plt
    directory = Path(directory)
    trajectory = event_trajectory(directory,event_id)
    if trajectory.empty:
        raise ValueError(f"Unknown event {event_id}")
    times = pd.to_datetime(trajectory.timestamp)
    fig, axes = plt.subplots(1,3,figsize=(15,4),layout="constrained")
    axes[0].plot(trajectory.centroid_lon,trajectory.centroid_lat,"o-")
    axes[0].set(xlabel="Longitude",ylabel="Latitude",title=f"Event {event_id}: area-weighted centroid")
    axes[1].plot(times,trajectory.area_km2)
    axes[1].set(ylabel="Area (km²)",title="Area evolution")
    axes[2].plot(times,trajectory.mean_score,label="mean")
    axes[2].plot(times,trajectory.max_score,label="max")
    axes[2].set(title="Anomaly scores")
    axes[2].legend()
    for axis in axes[1:]:
        axis.tick_params(axis="x",rotation=30)
    return fig, trajectory


def plot_event_snapshots(directory, event_id, n_frames=3):
    """Show first/intermediate/last event footprint, including split branches."""
    import matplotlib.pyplot as plt
    if n_frames < 1:
        raise ValueError("n_frames must be positive")
    directory = Path(directory)
    trajectory = event_trajectory(directory,event_id)
    if trajectory.empty:
        raise ValueError(f"Unknown event {event_id}")
    metadata = json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
    grid = CubeGrid(**metadata["grid"])
    selected = np.unique(np.linspace(0,len(trajectory)-1,min(n_frames,len(trajectory)),dtype=int))
    fig,axes = plt.subplots(1,len(selected),figsize=(5*len(selected),4),squeeze=False,layout="constrained")
    half = grid.spacing/2
    extent = [grid.longitudes[0]-half,grid.longitudes[-1]+half,grid.latitudes[-1]-half,grid.latitudes[0]+half]
    with closing(sqlite3.connect(directory/"events.sqlite")) as db:
        for axis,position in zip(axes.ravel(),selected):
            step = trajectory.iloc[position]
            ids = [r[0] for r in db.execute("SELECT cluster_id FROM clusters WHERE event_id=? AND time_index=?",
                                          (int(event_id),int(step.time_index)))]
            labels = _frame(directory/"cluster_labels.npy",int(step.time_index))
            values = _frame(directory/"anomaly_cube.npy",int(step.time_index))
            footprint = np.ma.masked_where(~np.isin(labels,ids),values)
            image = axis.imshow(footprint,extent=extent,origin="upper",aspect="auto",vmin=0,vmax=2)
            axis.set(title=step.timestamp,xlabel="Longitude",ylabel="Latitude")
            fig.colorbar(image,ax=axis,label="Anomaly score")
    return fig


def create_synthetic_demo(directory):
    """Synthetic moving/expanding anomaly; no ERA5 retrieval or model execution."""
    directory = Path(directory)
    if (directory/"metadata.json").is_file():
        if json.loads((directory/"metadata.json").read_text())["status"] == "complete":
            return directory
    rows, cols = np.indices((15,25))
    grid = CubeGrid(50-np.arange(15)*.5, np.arange(25)*.5, rows.ravel(),cols.ravel())
    scores = np.zeros((12,15,25),dtype=np.float32)
    for i in range(12):
        width = 5 + (i//3)%3
        scores[i,5:10,2+i:2+i+width] = 1.2+i*.03
    process_events(scores.reshape(12,-1),pd.date_range("2005-01-01",periods=12,freq="3h"),grid,directory,
                   EventConfig(absolute_threshold=1,chunk_size=3))
    return directory
