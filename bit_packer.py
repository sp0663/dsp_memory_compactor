# bit_packer.py
# Compresses and decompresses signal chunks using bit-packing via numpy/cupy

from utils import get_backend, to_numpy
import numpy as np_cpu  # always need numpy for some ops

xp = get_backend()


def quantize(signal, bit_width):
    """
    Quantize a float signal to a given bit width.
    Maps signal values (assumed -1 to 1) to integer levels.
    bit_width : number of bits (e.g. 4, 8, 16)
    Returns   : quantized integer array
    """
    levels = 2 ** bit_width
    # Clip signal to [-1, 1] range
    clipped = xp.clip(signal, -1.0, 1.0)
    # Map to [0, levels-1]
    quantized = xp.floor((clipped + 1.0) / 2.0 * (levels - 1)).astype(xp.int32)
    return quantized


def dequantize(quantized, bit_width):
    """
    Reconstruct float signal from quantized integer array.
    Reverses the quantize() operation.
    Returns: float signal array (approx. original, with quantization noise)
    """
    levels = 2 ** bit_width
    signal = (quantized.astype(xp.float32) / (levels - 1)) * 2.0 - 1.0
    return signal


def pack_chunk(signal, bit_width):
    """
    Full pack pipeline for a single chunk:
    1. Quantize to bit_width
    2. Return quantized data + metadata
    Returns: dict with packed data and metadata
    """
    quantized = quantize(signal, bit_width)
    original_size_bytes = len(signal) * 4  # float32 = 4 bytes
    packed_size_bytes = len(signal) * (bit_width / 8)  # approximate

    return {
        "data"               : quantized,
        "bit_width"          : bit_width,
        "original_size_bytes": original_size_bytes,
        "packed_size_bytes"  : packed_size_bytes,
        "compression_ratio"  : round(original_size_bytes / packed_size_bytes, 3),
    }


def unpack_chunk(packed):
    """
    Reconstruct signal from a packed chunk dict.
    Returns: float signal array
    """
    return dequantize(packed["data"], packed["bit_width"])


def compute_reconstruction_error(original, reconstructed):
    """
    Compute Mean Squared Error between original and reconstructed signal.
    Lower is better — shows quality loss from compression.
    """
    orig_np = to_numpy(original)
    recon_np = to_numpy(reconstructed)
    mse = np_cpu.mean((orig_np - recon_np) ** 2)
    return round(float(mse), 6)


def pack_signal(signal, decisions):
    """
    Pack an entire signal using per-chunk bit width decisions.
    signal    : full signal array
    decisions : list of dicts from adaptive_bitwidth.adapt_stream()
    Returns   : list of packed chunk dicts
    """
    num_chunks = len(decisions)
    chunk_size = len(signal) // num_chunks
    packed_chunks = []

    for i, decision in enumerate(decisions):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else len(signal)
        chunk = signal[start:end]
        packed = pack_chunk(chunk, decision["bit_width"])
        packed["chunk_index"] = i
        packed_chunks.append(packed)

    return packed_chunks


def unpack_signal(packed_chunks):
    """
    Reconstruct the full signal from packed chunks.
    Returns: reconstructed signal array
    """
    parts = [unpack_chunk(p) for p in packed_chunks]
    return xp.concatenate(parts)
