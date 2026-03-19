# parallel_stream.py
# Splits signal into sub-streams and processes them in parallel

import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from utils import get_backend, to_numpy

xp = get_backend()


def split_signal(signal, num_streams=8):
    """
    Split a signal into N sub-streams of equal size.
    Returns: list of signal chunks (numpy arrays)
    """
    signal_np = to_numpy(signal)
    chunks = np.array_split(signal_np, num_streams)
    print(f"[parallel] Split signal into {num_streams} sub-streams "
          f"of ~{len(chunks[0])} samples each")
    return chunks


def process_chunk(args):
    """
    Process a single chunk — analyze + pack.
    Designed to be called in parallel.
    args: (chunk_index, chunk_array, sample_rate)
    Returns: dict with analysis + packed result
    """
    from dsp_processor import analyze_signal
    from adaptive_bitwidth import decide_bit_width
    from bit_packer import pack_chunk

    chunk_index, chunk_np, sample_rate = args

    # Convert back to xp array for processing
    chunk = xp.array(chunk_np)

    # Analyze
    analysis = analyze_signal(chunk, sample_rate)
    snr = analysis["snr_db"]
    entropy = analysis["spectral_entropy"]

    # Decide bit width
    bit_width, reason, complexity = decide_bit_width(snr, entropy)

    # Pack
    packed = pack_chunk(chunk, bit_width)

    return {
        "chunk_index"        : chunk_index,
        "snr_db"             : snr,
        "spectral_entropy"   : entropy,
        "complexity"         : complexity,
        "bit_width"          : bit_width,
        "reason"             : reason,
        "original_size_bytes": packed["original_size_bytes"],
        "packed_size_bytes"  : packed["packed_size_bytes"],
        "compression_ratio"  : packed["compression_ratio"],
        "packed_data"        : packed["data"],
    }


def run_single_threaded(signal, sample_rate, num_streams=8):
    """
    Process all sub-streams sequentially (baseline).
    Returns: (results list, elapsed_ms)
    """
    chunks = split_signal(signal, num_streams)
    args_list = [(i, chunk, sample_rate) for i, chunk in enumerate(chunks)]

    start = time.perf_counter()
    results = [process_chunk(args) for args in args_list]
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"[parallel] Single-threaded: {elapsed_ms:.2f} ms")
    return results, elapsed_ms


def run_multi_threaded(signal, sample_rate, num_streams=8, max_workers=4):
    """
    Process all sub-streams in parallel using ThreadPoolExecutor.
    Returns: (results list, elapsed_ms)
    """
    chunks = split_signal(signal, num_streams)
    args_list = [(i, chunk, sample_rate) for i, chunk in enumerate(chunks)]

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_chunk, args_list))
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"[parallel] Multi-threaded ({max_workers} workers): {elapsed_ms:.2f} ms")
    return results, elapsed_ms


def run_parallel(signal, sample_rate, num_streams=8, mode="multi"):
    """
    Unified runner.
    mode: 'single' or 'multi'
    Returns: (results list, elapsed_ms)
    """
    if mode == "single":
        return run_single_threaded(signal, sample_rate, num_streams)
    elif mode == "multi":
        return run_multi_threaded(signal, sample_rate, num_streams)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'single' or 'multi'.")


def aggregate_metrics(results):
    """
    Aggregate per-chunk results into summary metrics.
    Returns: dict of summary stats
    """
    total_original = sum(r["original_size_bytes"] for r in results)
    total_packed = sum(r["packed_size_bytes"] for r in results)
    avg_bw = np.mean([r["bit_width"] for r in results])
    avg_cr = np.mean([r["compression_ratio"] for r in results])

    return {
        "total_original_bytes" : total_original,
        "total_packed_bytes"   : total_packed,
        "overall_compression"  : round(total_original / total_packed, 3),
        "avg_bit_width"        : round(float(avg_bw), 2),
        "avg_compression_ratio": round(float(avg_cr), 3),
        "num_chunks"           : len(results),
        "bit_widths"           : [r["bit_width"] for r in results],
        "complexities"         : [r["complexity"] for r in results],
        "snr_values"           : [r["snr_db"] for r in results],
    }
