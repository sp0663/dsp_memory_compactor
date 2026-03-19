# utils.py
# Backend switcher (numpy <-> cupy), timing, and shared metrics

import time
import numpy

# Try to import cupy for GPU support, fall back to numpy
try:
    import cupy as np
    GPU = True
    print("[utils] CuPy detected — using GPU backend (CUDA)")
except ImportError:
    import numpy as np
    GPU = False
    print("[utils] CuPy not found — using CPU backend (numpy)")


def get_backend():
    """Returns the active backend module (cupy or numpy)."""
    return np


def get_backend_name():
    """Returns a string name of the active backend."""
    return "CuPy (GPU)" if GPU else "NumPy (CPU)"


def to_numpy(array):
    """Converts cupy array to numpy if needed (for plotting etc.)"""
    if GPU:
        return numpy.asnumpy(array)
    return array


def timer(func):
    """Decorator to time any function and print elapsed time."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = (end - start) * 1000  # ms
        print(f"[timer] {func.__name__} took {elapsed:.3f} ms")
        return result, elapsed
    return wrapper


def compute_throughput(data_size_bytes, elapsed_ms):
    """
    Compute throughput in GB/s.
    data_size_bytes: size of processed data in bytes
    elapsed_ms: time taken in milliseconds
    """
    if elapsed_ms == 0:
        return 0.0
    elapsed_s = elapsed_ms / 1000
    throughput_gbs = (data_size_bytes / elapsed_s) / (1024 ** 3)
    return round(throughput_gbs, 4)


def compute_compression_ratio(original_bits, packed_bits):
    """Returns compression ratio as a float."""
    if packed_bits == 0:
        return 0.0
    return round(original_bits / packed_bits, 4)


def log_metrics(label, throughput, compression_ratio, bit_width):
    """Prints a formatted metrics summary."""
    print(f"\n--- Metrics [{label}] ---")
    print(f"  Backend       : {get_backend_name()}")
    print(f"  Bit Width     : {bit_width} bits")
    print(f"  Throughput    : {throughput} GB/s")
    print(f"  Compression   : {compression_ratio}x")
    print(f"------------------------\n")
