# benchmark.py
# 3-way benchmark: Single-threaded CPU vs Multi-threaded CPU vs GPU (CuPy)

import time
import numpy as np


def run_benchmark(signal_np, sample_rate, num_streams=8):
    """
    Run all three processing modes and compare performance.
    signal_np  : numpy array of the signal (always pass numpy here)
    sample_rate: int
    Returns    : dict of benchmark results
    """
    results = {}

    print("\n" + "="*50)
    print("  BENCHMARK: DSP Memory Compactor")
    print("="*50)

    # --- Mode 1: Single-threaded CPU ---
    print("\n[1/3] Running Single-threaded CPU...")
    from parallel_stream import run_single_threaded
    import utils
    # Force numpy backend for CPU runs
    _, t_single = run_single_threaded(
        utils.get_backend().array(signal_np), sample_rate, num_streams
    )
    results["single_cpu_ms"] = round(t_single, 3)

    # --- Mode 2: Multi-threaded CPU ---
    print("\n[2/3] Running Multi-threaded CPU...")
    from parallel_stream import run_multi_threaded
    _, t_multi = run_multi_threaded(
        utils.get_backend().array(signal_np), sample_rate, num_streams
    )
    results["multi_cpu_ms"] = round(t_multi, 3)

    # --- Mode 3: GPU (CuPy) if available ---
    print("\n[3/3] Checking GPU availability...")
    try:
        import cupy as cp
        signal_gpu = cp.array(signal_np)

        # Temporarily patch utils to use cupy
        import utils as u
        original_np = u.np
        u.np = cp

        start = time.perf_counter()
        from parallel_stream import run_multi_threaded as run_gpu
        _, t_gpu = run_gpu(signal_gpu, sample_rate, num_streams)
        elapsed_gpu = (time.perf_counter() - start) * 1000

        u.np = original_np  # restore
        results["gpu_ms"] = round(t_gpu, 3)
        results["gpu_available"] = True
        print(f"[benchmark] GPU (CuPy): {t_gpu:.2f} ms")

    except ImportError:
        results["gpu_ms"] = None
        results["gpu_available"] = False
        print("[benchmark] GPU not available (CuPy not installed) — skipping")

    # --- Summary ---
    print("\n" + "="*50)
    print("  BENCHMARK RESULTS")
    print("="*50)
    print(f"  Single-threaded CPU : {results['single_cpu_ms']} ms")
    print(f"  Multi-threaded CPU  : {results['multi_cpu_ms']} ms  "
          f"({speedup(results['single_cpu_ms'], results['multi_cpu_ms'])}x speedup)")
    if results["gpu_available"]:
        print(f"  GPU (CuPy/CUDA)     : {results['gpu_ms']} ms  "
              f"({speedup(results['single_cpu_ms'], results['gpu_ms'])}x speedup)")
    else:
        print("  GPU (CuPy/CUDA)     : N/A")
    print("="*50 + "\n")

    return results


def speedup(baseline_ms, target_ms):
    """Compute speedup ratio."""
    if target_ms and target_ms > 0:
        return round(baseline_ms / target_ms, 2)
    return "N/A"


def get_benchmark_labels(results):
    """
    Returns labels and values for plotting in dashboard.
    """
    labels = ["Single CPU", "Multi CPU"]
    values = [results["single_cpu_ms"], results["multi_cpu_ms"]]

    if results.get("gpu_available") and results.get("gpu_ms"):
        labels.append("GPU (CuPy)")
        values.append(results["gpu_ms"])

    return labels, values
