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


def plot_frame(directory, time_index=0, *, event_id=None):
    import matplotlib.pyplot as plt
    directory = Path(directory)
    metadata = json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
    grid = CubeGrid(**metadata["grid"])
    config = EventConfig(**metadata["config"])
    frame = _frame(directory/metadata.get("anomaly_mean_cube_file", "anomaly_cube.npy"),time_index)
    uncertainty = (_frame(directory/metadata.get("uncertainty_cube_file", "uncertainty_cube.npy"),time_index)
                   if metadata.get("uncertainty_available") else None)
    labels = _frame(directory/"cluster_labels.npy",time_index)
    with closing(sqlite3.connect(directory/"events.sqlite")) as db:
        row = db.execute("SELECT timestamp,threshold FROM frames WHERE time_index=?",(int(time_index),)).fetchone()
        clusters = pd.read_sql_query("SELECT * FROM clusters WHERE time_index=?",db,params=(int(time_index),))
    if row is None:
        raise IndexError(time_index)
    raw, opened, closed = morphology(frame,row[1],kernel_size=config.kernel_size,
        opening_iterations=config.opening_iterations,closing_iterations=config.closing_iterations)
    panels = [(frame,"Anomaly score"),(raw,"Threshold"),(opened,"Opening"),
              (closed,"Closing"),(np.ma.masked_equal(labels,0),"Cluster ID")]
    if uncertainty is not None:
        panels.append((uncertainty,"MC std"))
    selected_mask = None
    row_start, row_end = 0, grid.shape[0]
    col_start, col_end = 0, grid.shape[1]
    if event_id is not None:
        clusters = clusters.loc[clusters.event_id == int(event_id)]
        selected_mask = np.isin(labels,clusters.cluster_id.to_numpy())
        if not np.any(selected_mask):
            raise ValueError(f"Event {event_id} has no cluster at time_index={time_index}")
        rows = np.flatnonzero(selected_mask.any(axis=1))
        cols = np.flatnonzero(selected_mask.any(axis=0))
        row_start, row_end = max(0,int(rows.min())-2), min(grid.shape[0],int(rows.max())+3)
        col_start, col_end = max(0,int(cols.min())-2), min(grid.shape[1],int(cols.max())+3)
    fig, axes = plt.subplots(1,len(panels),figsize=(4*len(panels),4),layout="constrained")
    half = grid.spacing/2
    extent = [grid.longitudes[col_start]-half,grid.longitudes[col_end-1]+half,
              grid.latitudes[row_end-1]-half,grid.latitudes[row_start]+half]
    for axis, (values,title) in zip(axes,panels):
        local = values[row_start:row_end,col_start:col_end]
        plot = axis.imshow(local,origin="upper",extent=extent,aspect="auto",interpolation="nearest")
        if selected_mask is not None:
            axis.contour(selected_mask[row_start:row_end,col_start:col_end].astype(float),
                         levels=[.5],colors="cyan",linewidths=1.5,extent=extent,origin="upper")
        axis.set(title=title,xlabel="Longitude",ylabel="Latitude")
        fig.colorbar(plot,ax=axis,shrink=.7)
    event_label = f" | event {event_id} outlined in cyan" if event_id is not None else ""
    fig.suptitle(f"{row[0]} | threshold > {row[1]:.6g}{event_label}")
    return fig, clusters


def plot_event(directory, event_id):
    import matplotlib.pyplot as plt
    directory = Path(directory)
    trajectory = event_trajectory(directory,event_id)
    if trajectory.empty:
        raise ValueError(f"Unknown event {event_id}")
    times = pd.to_datetime(trajectory.timestamp)
    mean_uncertainty = pd.to_numeric(trajectory.mean_uncertainty,errors="coerce")
    max_uncertainty = pd.to_numeric(trajectory.max_uncertainty,errors="coerce")
    has_uncertainty = mean_uncertainty.notna().any()
    n_panels = 4 if has_uncertainty else 3
    fig, axes = plt.subplots(1,n_panels,figsize=(5*n_panels,4),layout="constrained")
    axes[0].plot(trajectory.centroid_lon,trajectory.centroid_lat,"o-")
    axes[0].set(xlabel="Longitude",ylabel="Latitude",title=f"Event {event_id}: area-weighted centroid")
    axes[1].plot(times,trajectory.area_km2,"o-")
    axes[1].set(ylabel="Area (km²)",title="Area evolution")
    axes[2].plot(times,trajectory.mean_score,"o-",label="mean")
    axes[2].plot(times,trajectory.max_score,"o-",label="max")
    axes[2].set(title="Anomaly scores")
    axes[2].legend()
    if has_uncertainty:
        axes[3].plot(times,mean_uncertainty,"o-",label="mean")
        axes[3].plot(times,max_uncertainty,"o-",label="max")
        axes[3].set(title="MC std (score variability)")
        axes[3].legend()
    if len(trajectory) == 1:
        fig.suptitle(f"Event {event_id}: one timestamp, no temporal evolution")
    for axis in axes[1:]:
        axis.tick_params(axis="x",rotation=30)
    return fig, trajectory


def plot_event_snapshots(directory, event_id, n_frames=3):
    """Compare local scores with the selected event at matching timestamps."""
    import matplotlib.pyplot as plt
    if n_frames < 1:
        raise ValueError("n_frames must be positive")
    directory = Path(directory)
    trajectory = event_trajectory(directory,event_id)
    if trajectory.empty:
        raise ValueError(f"Unknown event {event_id}")
    metadata = json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
    grid = CubeGrid(**metadata["grid"])
    has_uncertainty = bool(metadata.get("uncertainty_available"))
    selected = np.unique(np.linspace(0,len(trajectory)-1,min(n_frames,len(trajectory)),dtype=int))
    frames = []
    with closing(sqlite3.connect(directory/"events.sqlite")) as db:
        for position in selected:
            step = trajectory.iloc[position]
            ids = [r[0] for r in db.execute("SELECT cluster_id FROM clusters WHERE event_id=? AND time_index=?",
                                          (int(event_id),int(step.time_index)))]
            labels = _frame(directory/"cluster_labels.npy",int(step.time_index))
            values = _frame(directory/metadata.get("anomaly_mean_cube_file", "anomaly_cube.npy"),int(step.time_index))
            uncertainty = (_frame(directory/metadata.get("uncertainty_cube_file", "uncertainty_cube.npy"),
                                  int(step.time_index)) if has_uncertainty else None)
            mask = np.isin(labels,ids)
            if not np.any(mask):
                raise ValueError(f"Event {event_id} has no cells at {step.timestamp}")
            frames.append((step, values, mask, uncertainty))
    row_positions = np.concatenate([np.flatnonzero(mask.any(axis=1)) for _,_,mask,_ in frames])
    col_positions = np.concatenate([np.flatnonzero(mask.any(axis=0)) for _,_,mask,_ in frames])
    row_start = max(0, int(row_positions.min())-2)
    row_end = min(grid.shape[0], int(row_positions.max())+3)
    col_start = max(0, int(col_positions.min())-2)
    col_end = min(grid.shape[1], int(col_positions.max())+3)
    half = grid.spacing/2
    extent = [grid.longitudes[col_start]-half, grid.longitudes[col_end-1]+half,
              grid.latitudes[row_end-1]-half, grid.latitudes[row_start]+half]
    rows = 3 if has_uncertainty else 2
    fig,axes = plt.subplots(rows,len(frames),figsize=(5*len(frames),4*rows),
                            squeeze=False,layout="constrained")
    uncertainty_max = (max(float(np.nanmax(uncertainty[mask]))
                           for _,_,mask,uncertainty in frames) if has_uncertainty else None)
    for column,(step,values,mask,uncertainty) in enumerate(frames):
        local_scores = values[row_start:row_end,col_start:col_end]
        local_event = mask[row_start:row_end,col_start:col_end]
        context_axis = axes[0,column]
        context_image = context_axis.imshow(local_scores,extent=extent,origin="upper",
                                            aspect="auto",interpolation="nearest",vmin=0,vmax=2)
        context_axis.contour(local_event.astype(float),levels=[.5],colors="cyan",
                             linewidths=1.5,extent=extent,origin="upper")
        context_axis.set(title=f"{step.timestamp} | all scores",xlabel="Longitude",ylabel="Latitude")
        fig.colorbar(context_image,ax=context_axis,label="Anomaly score")
        footprint = np.ma.masked_where(~local_event,local_scores)
        event_axis = axes[1,column]
        event_image = event_axis.imshow(footprint,extent=extent,origin="upper",aspect="auto",
                                        interpolation="nearest",vmin=0,vmax=2)
        event_axis.set(title=f"Event {event_id} only",xlabel="Longitude",ylabel="Latitude")
        fig.colorbar(event_image,ax=event_axis,label="Anomaly score")
        if has_uncertainty:
            std_footprint = np.ma.masked_where(~mask[row_start:row_end,col_start:col_end],
                                               uncertainty[row_start:row_end,col_start:col_end])
            std_axis = axes[2,column]
            std_image = std_axis.imshow(std_footprint,extent=extent,origin="upper",
                                        aspect="auto",interpolation="nearest",vmin=0,
                                        vmax=max(uncertainty_max,1e-8))
            std_axis.set(title="MC std",xlabel="Longitude",ylabel="Latitude")
            fig.colorbar(std_image,ax=std_axis,label="Score variability")
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
