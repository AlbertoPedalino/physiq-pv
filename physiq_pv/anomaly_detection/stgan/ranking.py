"""Exact global ranking and summaries with an external-sort option."""
from dataclasses import dataclass
import sqlite3
import tempfile
from pathlib import Path
import numpy as np
from .scoring import ScoreStore


@dataclass
class Ranking:
    flags: np.ndarray
    ranks: np.ndarray
    percentiles: np.ndarray
    order: np.ndarray


def rank_scores(scores, percentage, store, *, chunk_size=65536):
    """Stable descending score, then ascending original flattened index.

    SQLite performs the external ORDER BY in native code with disk-backed temp
    storage and a bounded page cache. Python only holds one fetch/insert chunk.
    It is slower than the in-memory path but does not need O(N) heap RAM.
    """
    if not np.isfinite(percentage) or not 0 < percentage <= 100:
        raise ValueError("paper_top_k percentage must be in (0, 100].")
    flat = scores.reshape(-1)
    disk = store.backend == "memmap" or flat.size * 80 > store.memory_limit_mb * 1024**2
    if not disk:
        values = np.asarray(flat, dtype=np.float64)
        finite = np.flatnonzero(np.isfinite(values))
        if not len(finite):
            raise ValueError("Cannot rank STGAN test scores without finite values.")
        order = finite[np.argsort(-values[finite], kind="stable")]
        return _write_ranking(scores.shape, len(order), (order,), percentage, store, chunk_size, False)
    with tempfile.TemporaryDirectory(prefix="ranking_", dir=store.directory()) as tmp:
        connection = sqlite3.connect(str(Path(tmp) / "scores.sqlite3"))
        try:
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-32768")
            connection.execute("PRAGMA mmap_size=0")
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")  # Disposable sort workspace.
            connection.execute("CREATE TABLE scores (idx INTEGER PRIMARY KEY, score REAL NOT NULL)")
            count = 0
            for start in range(0, flat.size, chunk_size):
                block = np.asarray(flat[start:start+chunk_size])
                indices = np.flatnonzero(np.isfinite(block))
                connection.executemany("INSERT INTO scores VALUES (?, ?)",
                    ((start+int(i), float(block[i])) for i in indices))
                count += len(indices)
            connection.commit()
            if not count:
                raise ValueError("Cannot rank STGAN test scores without finite values.")
            cursor = connection.execute("SELECT idx FROM scores ORDER BY score DESC, idx ASC")
            def chunks():
                while rows := cursor.fetchmany(chunk_size):
                    yield np.fromiter((row[0] for row in rows), dtype=np.int64, count=len(rows))
            return _write_ranking(scores.shape, count, chunks(), percentage, store, chunk_size, True)
        finally:
            connection.close()


def _write_ranking(shape, count, chunks, percentage, store, chunk_size, disk):
    flags = store.allocate("rank_flags", shape, np.bool_, disk=disk)
    ranks = store.allocate("ranks", shape, np.float64, disk=disk)
    percentiles = store.allocate("percentiles", shape, np.float64, disk=disk)
    order = store.allocate("rank_order", (count,), np.int64, disk=disk)
    ff, rf, pf = flags.reshape(-1), ranks.reshape(-1), percentiles.reshape(-1)
    for start in range(0, ff.size, chunk_size):
        ff[start:start+chunk_size] = False
        rf[start:start+chunk_size] = np.nan
        pf[start:start+chunk_size] = np.nan
    keep = min(count, max(1, int(np.ceil(count * percentage / 100.0))))
    position = 0
    for indices in chunks:
        # Also bound temporary arrays for the in-memory sort path.
        for start in range(0, len(indices), chunk_size):
            block = indices[start:start+chunk_size]
            numbers = np.arange(position+1, position+len(block)+1, dtype=np.float64)
            rf[block] = numbers
            pf[block] = 100.0 * (count - numbers + 1.0) / count
            ff[block] = numbers <= keep
            order[position:position+len(block)] = block
            position += len(block)
    for array in (flags,ranks,percentiles,order):
        if isinstance(array,np.memmap):
            array.flush()
    return Ranking(flags,ranks,percentiles,order)


def boundary_summaries(scores, ranking, groups, *, chunk_size=65536):
    """Compute exact order statistics from the already sorted global order.

    No full group copy or second sort. Mean uses float64 streaming accumulation,
    rounded to the original float32 output precision.
    """
    flat, flags = scores.reshape(-1), ranking.flags.reshape(-1)
    n_times, n_locations = scores.shape
    states = []
    for name, selected in groups:
        count = n_times * int(np.sum(selected))
        if not count:
            continue
        positions = {}
        for q in (.5,.95):
            index = (count-1)*q
            positions[q] = (count-1-int(np.floor(index)),count-1-int(np.ceil(index)),index%1)
        states.append(dict(name=name, selected=np.asarray(selected), count=count, seen=0,
                           total=0., anomalies=0, positions=positions, values={}))
    for start in range(0, len(ranking.order), chunk_size):
        indices = np.asarray(ranking.order[start:start+chunk_size])
        locations = indices % n_locations
        values = np.asarray(flat[indices])
        labels = np.asarray(flags[indices])
        for state in states:
            select = state['selected'][locations]
            block = values[select]
            offset = state['seen']
            for lower,upper,_ in state['positions'].values():
                for position in (lower,upper):
                    if offset <= position < offset+len(block):
                        state['values'][position] = block[position-offset]
            state['total'] += float(np.sum(block,dtype=np.float64))
            state['anomalies'] += int(labels[select].sum())
            state['seen'] += len(block)
    output=[]
    for state in states:
        def quantile(q):
            lower,upper,fraction=state['positions'][q]
            pair=np.array([state['values'][lower],state['values'][upper]],dtype=scores.dtype)
            return float(np.mean(pair)) if q==.5 else float(np.quantile(pair,fraction))
        output.append(dict(group=state['name'], n_locations=int(state['selected'].sum()),
            n_scored=state['count'], score_mean=float(np.float32(state['total']/state['count'])),
            score_median=quantile(.5), score_q95=quantile(.95), n_anomaly=state['anomalies'],
            anomaly_share_pct=float(state['anomalies']/state['count']*100)))
    return output
