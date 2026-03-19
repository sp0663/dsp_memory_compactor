# signal_generator.py
# Generates synthetic signals and loads real-world datasets (.wav or .csv)

import numpy as np
from utils import get_backend

xp = get_backend()


def generate_sine_wave(freq=440, sample_rate=44100, duration=1.0, amplitude=1.0):
    """
    Generate a clean sine wave signal.
    freq        : frequency in Hz
    sample_rate : samples per second
    duration    : length in seconds
    amplitude   : signal amplitude
    Returns     : (signal array, sample_rate)
    """
    t = xp.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = amplitude * xp.sin(2 * xp.pi * freq * t)
    return signal, sample_rate


def generate_noisy_signal(freq=440, sample_rate=44100, duration=1.0, noise_level=0.5):
    """
    Generate a sine wave with added Gaussian noise.
    noise_level : standard deviation of noise (0 = clean, 1+ = very noisy)
    Returns     : (signal array, sample_rate)
    """
    signal, sr = generate_sine_wave(freq, sample_rate, duration)
    noise = xp.random.normal(0, noise_level, signal.shape)
    noisy_signal = signal + noise
    return noisy_signal, sr


def generate_multi_tone(freqs=[440, 880, 1320], sample_rate=44100, duration=1.0):
    """
    Generate a complex signal with multiple frequency components.
    Useful for testing adaptive bit-width on complex signals.
    freqs : list of frequencies to mix
    Returns: (signal array, sample_rate)
    """
    t = xp.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = xp.zeros_like(t)
    for f in freqs:
        signal += xp.sin(2 * xp.pi * f * t)
    signal /= len(freqs)  # normalize
    return signal, sample_rate


def load_wav(filepath):
    """
    Load a .wav audio file as a signal.
    Returns: (signal array, sample_rate)
    Requires: scipy
    """
    from scipy.io import wavfile
    sr, data = wavfile.read(filepath)
    # Convert to float32 and normalize
    data = data.astype(np.float32)
    if data.max() > 0:
        data = data / data.max()
    # If stereo, take one channel
    if data.ndim > 1:
        data = data[:, 0]
    signal = xp.array(data)
    print(f"[signal_generator] Loaded WAV: {filepath} | SR: {sr} | Samples: {len(signal)}")
    return signal, sr


def load_csv_signal(filepath, column=0):
    """
    Load a time-series signal from a CSV file (e.g. Kaggle sensor datasets).
    column : which column to use as the signal
    Returns: (signal array, sample_rate=1000 assumed)
    """
    import pandas as pd
    df = pd.read_csv(filepath)
    data = df.iloc[:, column].dropna().values.astype(np.float32)
    # Normalize
    if data.max() - data.min() > 0:
        data = (data - data.min()) / (data.max() - data.min()) * 2 - 1
    signal = xp.array(data)
    sample_rate = 1000  # assumed default for sensor data
    print(f"[signal_generator] Loaded CSV: {filepath} | Samples: {len(signal)}")
    return signal, sample_rate


def get_signal(mode="sine", **kwargs):
    """
    Unified signal loader.
    mode options: 'sine', 'noisy', 'multi_tone', 'wav', 'csv'
    Pass relevant kwargs for each mode.
    """
    modes = {
        "sine"       : generate_sine_wave,
        "noisy"      : generate_noisy_signal,
        "multi_tone" : generate_multi_tone,
        "wav"        : load_wav,
        "csv"        : load_csv_signal,
    }
    if mode not in modes:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(modes.keys())}")
    return modes[mode](**kwargs)
