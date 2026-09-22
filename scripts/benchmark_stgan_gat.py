"""Synthetic CPU/CUDA GAN + MC benchmark; never opens ERA5 observations."""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from physiq_pv.anomaly_detection.stgan import STGANGAT, build_spatial_grid, grid_edge_index
from physiq_pv.anomaly_detection.stgan.scoring import scoring_mode
from physiq_pv.anomaly_detection.stgan.training import gan_train_step


def peak_rss_bytes():
    """OS process high-water mark, includes Python/torch libraries, not just tensors."""
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        value = Counters()
        value.cb = ctypes.sizeof(value)
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = ctypes.c_void_p
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        if not query(process(), ctypes.byref(value), value.cb):
            raise ctypes.WinError()
        return value.PeakWorkingSetSize
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def benchmark(args, device, batch_size):
    torch.manual_seed(20)
    torch.set_num_threads(args.cpu_threads)
    rows, cols = np.indices((args.height, args.width)).reshape(2, -1)
    # Fixed synthetic mask; coordinates are never compressed across holes.
    if args.valid_fraction < 1:
        keep = np.random.default_rng(20).random(len(rows)) < args.valid_fraction
        rows, cols = rows[keep], cols[keep]
    grid = build_spatial_grid(60-rows*.5, -15+cols*.5, grid_crs="EPSG:4326",
                              angular_spacing=.5, audit_knn=False)
    edges = grid_edge_index(grid.row_indices, grid.column_indices)
    nodes = grid.n_locations
    target = torch.device(device)
    def sync():
        if target.type == "cuda":
            torch.cuda.synchronize(target)
    model = STGANGAT(n_features=15, edge_index=edges, node_indices=grid.node_indices,
        hidden_size=args.hidden_size, n_layers=2, cnn_channels=args.cnn_channels, cnn_layers=2,
        gat_hidden_dim=args.gat_hidden_dim, gat_heads=args.gat_heads, gat_layers=2,
        recent_steps=args.recent_steps, discriminator_chunk_size=args.discriminator_chunk_size,
        trend_chunk_size=args.trend_chunk_size).to(target)
    batch = (torch.randn(batch_size, args.recent_steps, nodes, 15, device=target),
             torch.randn(batch_size, nodes, args.trend_steps, 15, device=target),
             torch.ones(batch_size, nodes, 1, device=target),
             torch.randn(batch_size, 31, device=target),
             torch.randn(batch_size, nodes, 15, device=target))
    go = torch.optim.Adam(model.generator.parameters(), lr=1e-3)
    do = torch.optim.Adam(model.discriminator.parameters(), lr=1e-3)
    for _ in range(args.warmup):
        gan_train_step(model, batch, go, do)
    sync()
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    start = time.perf_counter()
    for _ in range(args.steps):
        losses = gan_train_step(model, batch, go, do)
    sync()
    train_seconds = time.perf_counter() - start
    train_peak = torch.cuda.max_memory_allocated(target) if target.type == "cuda" else None
    train_reserved = torch.cuda.max_memory_reserved(target) if target.type == "cuda" else None
    model.zero_grad(set_to_none=True)
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    start = time.perf_counter()
    with scoring_mode(model, True), torch.no_grad():
        for _ in range(args.mc_samples):
            model.score_draw(*batch)
    sync()
    score_seconds = time.perf_counter() - start
    return {"status": "ok", "device": device, "device_name": torch.cuda.get_device_name(target) if target.type == "cuda" else "CPU",
        "torch": torch.__version__, "cpu_threads": args.cpu_threads,
        "nodes": nodes, "edges_including_self": edges.shape[1], "batch_timestamps": batch_size,
        "recent_steps": args.recent_steps, "trend_steps": args.trend_steps,
        "parameters": model.parameter_counts(),
        "input_shapes": [list(t.shape) for t in batch], "output_shape": [batch_size, nodes, 15],
        "edge_index_bytes": edges.numel() * edges.element_size(),
        "equivalent_dense_float32_adjacency_bytes_not_allocated": nodes * nodes * 4,
        "train_steps": args.steps, "train_seconds": train_seconds,
        "train_timestamps_per_second": args.steps * batch_size / train_seconds,
        "train_node_targets_per_second": args.steps * batch_size * nodes / train_seconds,
        "train_cuda_peak_allocated_bytes": train_peak, "train_cuda_peak_reserved_bytes": train_reserved,
        "mc_samples": args.mc_samples, "mc_seconds": score_seconds,
        "mc_timestamps_per_second": batch_size / score_seconds,
        "mc_cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(target) if target.type == "cuda" else None,
        "process_lifetime_peak_rss_bytes": peak_rss_bytes(),
        "last_losses": [float(x) for x in losses],
        "scope": "synthetic full G+D Adam steps (two G forwards) and MC scoring; excludes disk/normalization/events"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "all"), default="all")
    parser.add_argument("--height", type=int, default=12)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--valid-fraction", type=float, default=1)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--cuda-memory-fraction", type=float, default=1.,
                        help="Allocator cap for safely probing larger synthetic graphs")
    parser.add_argument("--recent-steps", type=int, default=1)
    parser.add_argument("--trend-steps", type=int, default=56)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--cnn-channels", type=int, default=32)
    parser.add_argument("--gat-hidden-dim", type=int, default=16)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--discriminator-chunk-size", type=int, default=256)
    parser.add_argument("--trend-chunk-size", type=int, default=256)
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if (min(args.height, args.width, args.steps, args.cpu_threads, args.recent_steps,
            args.trend_steps, args.mc_samples, *args.batches) < 1
            or args.warmup < 0 or not 0 < args.valid_fraction <= 1
            or not 0 < args.cuda_memory_fraction <= 1):
        parser.error("Dimensions/counts must be positive; warmup >= 0; 0 < valid-fraction <= 1")
    devices = ["cpu", "cuda"] if args.device == "all" else [args.device]
    results = []
    for device in devices:
        for batch_size in args.batches:
            if device == "cuda" and not torch.cuda.is_available():
                results.append({"device": device, "batch_timestamps": batch_size, "status": "unavailable"})
                continue
            if device == "cuda":
                torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
            try:
                result = benchmark(args, device, batch_size)
            except torch.cuda.OutOfMemoryError:
                result = {"device": device, "batch_timestamps": batch_size, "status": "cuda_out_of_memory",
                          "hint": "Reduce graph batch size/dimensions; the old patch batch size is unsuitable."}
            results.append(result)
            print(json.dumps(result), flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"config": vars(args) | {"output": str(args.output)},
                                           "results": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
