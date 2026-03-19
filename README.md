# DSP Smart Memory Compactor

A real-time DSP pipeline that dynamically adapts bit-precision based on signal complexity, reducing memory bandwidth while preserving signal quality.

## Concept

When processing high-speed signal data, not all parts of the signal are equally complex. This system:
- Analyzes each chunk using **FFT** and **spectral entropy**
- Decides the optimal **bit-width** dynamically (4 to 16 bits)
- Compresses simple/clean chunks more aggressively
- Preserves precision for complex/noisy chunks
- Processes all chunks in **parallel**

## Project Structure

```
dsp_memory_compactor/
├── main.py                # Entry point
├── signal_generator.py    # Synthetic + real signal loading
├── dsp_processor.py       # FFT, filtering, complexity analysis
├── adaptive_bitwidth.py   # Core adaptive algorithm
├── bit_packer.py          # Quantization + compression
├── parallel_stream.py     # Multi-threaded stream processing
├── benchmark.py           # 3-way CPU vs GPU benchmark
├── dashboard.py           # Plotly interactive dashboard
└── utils.py               # Backend switcher (numpy <-> cupy)
```

## Setup

```bash
pip install -r requirements.txt
```

For GPU support (Kaggle or Nvidia GPU):
```bash
pip install cupy-cuda11x  # adjust cuda version as needed
```

## Usage

### Basic run (noisy sine wave)
```bash
python main.py
```

### Custom signal type
```bash
python main.py --signal sine --duration 3.0
python main.py --signal multi_tone
python main.py --signal wav --file path/to/audio.wav
python main.py --signal csv --file path/to/sensor_data.csv
```

### With benchmark
```bash
python main.py --benchmark
```

### More sub-streams
```bash
python main.py --streams 16 --mode multi
```

## Dashboard

The Plotly dashboard shows:
- Original signal waveform
- FFT frequency spectrum
- Adaptive bit-width per chunk
- Signal complexity score per chunk
- Compression ratio per chunk
- SNR per chunk
- Benchmark comparison (if `--benchmark` flag used)

## Kaggle / GPU

On Kaggle:
1. Enable GPU in notebook settings
2. `!pip install cupy-cuda11x`
3. Run normally — the system auto-detects CuPy and switches to GPU backend

## Tech Stack

- `numpy` / `cupy` — vectorized signal processing
- `scipy` — DSP filters (Butterworth)
- `plotly` — interactive dashboard
- `concurrent.futures` — parallel stream processing
- `pandas` — CSV dataset loading
