# dsp_processor.py
# FFT analysis, filtering, and signal complexity scoring

import numpy as np
from scipy.signal import welch
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
    Estimate Signal-to-Noise Ratio (SNR) in dB using Welch's method.

    Welch's method:
    - Splits signal into overlapping windows (segments)
    - Computes FFT on each window
    - Averages the power spectra across all windows
    - Result: Power Spectral Density (PSD) — much more stable than single FFT

    SNR estimation:
    - Signal power  = peak of the averaged PSD
    - Noise power   = mean of PSD excluding the peak region
    - SNR (dB)      = 10 * log10(signal_power / noise_power)

    Note: we use 10*log10 here (not 20) because Welch gives us
    power directly (amplitude²), not amplitude.

    Higher SNR = cleaner signal.
    """
    signal_np = to_numpy(signal)

    # Welch's PSD estimate
    # nperseg: window size — 256 is a good balance of freq resolution vs averaging
    freqs, psd = welch(signal_np, fs=sample_rate, nperseg=256)

    # Find peak (signal)
    peak_idx = np.argmax(psd)
    signal_power = psd[peak_idx]

    # Noise = mean of all bins excluding a small window around the peak
    # This avoids the signal itself contaminating the noise estimate
    noise_mask = np.ones(len(psd), dtype=bool)
    half_window = 5  # exclude 5 bins either side of peak
    noise_mask[max(0, peak_idx - half_window): peak_idx + half_window + 1] = False
    noise_power = np.mean(psd[noise_mask])

    if noise_power == 0:
        return float('inf')

    # 10*log10 because PSD is already in power units
    snr = 10 * np.log10(signal_power / noise_power)
    return round(float(snr), 2)


def compute_spectral_entropy(signal, sample_rate):
    """
    Compute spectral entropy using Welch's PSD instead of raw FFT magnitudes.

    Using Welch here too gives a smoother, more stable probability
    distribution for entropy calculation — less sensitive to single-frame noise.

    Low entropy  = simple signal (e.g. pure sine)
    High entropy = complex/noisy signal
    Returns: entropy score (float, 0 to 1 normalized)
    """
    signal_np = to_numpy(signal)

    # Welch PSD
    _, psd = welch(signal_np, fs=sample_rate, nperseg=256)

    # Normalize PSD to probability distribution
    total = psd.sum()
    if total == 0:
        return 0.0
    prob = psd / total

    # Shannon entropy
    prob = prob[prob > 0]  # avoid log(0)
    entropy = -np.sum(prob * np.log2(prob))

    # Normalize by max possible entropy (log2 of number of bins)
    max_entropy = np.log2(len(prob))
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0

    return round(float(normalized), 4)


def analyze_signal(signal, sample_rate):
    """
    Full signal analysis — returns a dict of metrics.
    Used by adaptive_bitwidth.py to make bit-width decisions.
    Runs Welch's PSD once and reuses for both SNR and entropy.
    """
    signal_np = to_numpy(signal)

    # Run Welch once — reuse for both SNR and entropy
    freqs_welch, psd = welch(signal_np, fs=sample_rate, nperseg=256)

    # --- SNR from Welch PSD ---
    peak_idx = np.argmax(psd)
    signal_power = psd[peak_idx]
    noise_mask = np.ones(len(psd), dtype=bool)
    noise_mask[max(0, peak_idx - 5): peak_idx + 6] = False
    noise_power = np.mean(psd[noise_mask])
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

    # --- Spectral Entropy from Welch PSD ---
    total = psd.sum()
    prob = psd / total if total > 0 else psd
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    max_entropy = np.log2(len(prob))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # --- FFT for dashboard visualization (kept separate) ---
    freqs_fft, mags_fft = compute_fft(signal, sample_rate)

    return {
        "snr_db"          : round(float(snr), 2),
        "spectral_entropy": round(float(normalized_entropy), 4),
        "fft_freqs"       : freqs_fft,
        "fft_magnitudes"  : mags_fft,
        "welch_freqs"     : freqs_welch,
        "welch_psd"       : psd,
        "num_samples"     : len(signal),
        "sample_rate"     : sample_rate,
    }
