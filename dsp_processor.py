# dsp_processor.py
# FFT analysis, filtering, and signal complexity scoring

import numpy as np
from utils import get_backend, to_numpy

xp = get_backend()


def compute_fft(signal, sample_rate):
    """
    Compute the FFT of a signal.
    Returns: (frequencies array, magnitudes array)
    """
    n = len(signal)
    fft_vals = xp.fft.fft(signal)
    magnitudes = xp.abs(fft_vals[:n // 2])  # one-sided spectrum
    frequencies = xp.fft.fftfreq(n, d=1 / sample_rate)[:n // 2]
    return frequencies, magnitudes


def low_pass_filter(signal, sample_rate, cutoff_hz=1000):
    """
    Apply a low-pass Butterworth filter to the signal.
    Removes high frequency noise.
    cutoff_hz : frequency above which to attenuate
    """
    from scipy.signal import butter, filtfilt
    nyq = sample_rate / 2
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    # scipy works on numpy arrays
    signal_np = to_numpy(signal)
    filtered = filtfilt(b, a, signal_np)
    return xp.array(filtered)


def high_pass_filter(signal, sample_rate, cutoff_hz=100):
    """
    Apply a high-pass Butterworth filter.
    Removes low frequency drift/DC offset.
    """
    from scipy.signal import butter, filtfilt
    nyq = sample_rate / 2
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(4, normal_cutoff, btype='high', analog=False)
    signal_np = to_numpy(signal)
    filtered = filtfilt(b, a, signal_np)
    return xp.array(filtered)


def compute_snr(signal, sample_rate):
    """
    Estimate Signal-to-Noise Ratio (SNR) in dB.
    Uses FFT to separate signal power from noise floor.
    Higher SNR = cleaner signal.
    """
    _, magnitudes = compute_fft(signal, sample_rate)
    magnitudes_np = to_numpy(magnitudes)

    peak = magnitudes_np.max()
    noise_floor = np.median(magnitudes_np)

    if noise_floor == 0:
        return float('inf')

    snr = 20 * np.log10(peak / noise_floor)
    return round(float(snr), 2)


def compute_spectral_entropy(signal, sample_rate):
    """
    Compute spectral entropy — a measure of signal complexity.
    Low entropy  = simple signal (e.g. pure sine)
    High entropy = complex/noisy signal
    Returns: entropy score (float, 0 to 1 normalized)
    """
    _, magnitudes = compute_fft(signal, sample_rate)
    magnitudes_np = to_numpy(magnitudes)

    # Normalize to probability distribution
    power = magnitudes_np ** 2
    total = power.sum()
    if total == 0:
        return 0.0
    prob = power / total

    # Shannon entropy
    prob = prob[prob > 0]  # avoid log(0)
    entropy = -np.sum(prob * np.log2(prob))

    # Normalize by max possible entropy
    max_entropy = np.log2(len(prob))
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0

    return round(float(normalized), 4)


def analyze_signal(signal, sample_rate):
    """
    Full signal analysis — returns a dict of metrics.
    Used by adaptive_bitwidth.py to make bit-width decisions.
    """
    snr = compute_snr(signal, sample_rate)
    entropy = compute_spectral_entropy(signal, sample_rate)
    freqs, mags = compute_fft(signal, sample_rate)

    return {
        "snr_db"          : snr,
        "spectral_entropy": entropy,
        "fft_freqs"       : freqs,
        "fft_magnitudes"  : mags,
        "num_samples"     : len(signal),
        "sample_rate"     : sample_rate,
    }
