# main.py
# Entry point — ties the full DSP Memory Compactor pipeline together

import argparse
import numpy as np

from signal_generator import get_signal
from dsp_processor import analyze_signal
from parallel_stream import run_parallel, aggregate_metrics
from dashboard import build_dashboard
from utils import get_backend_name, to_numpy


def parse_args():
    parser = argparse.ArgumentParser(description="DSP Smart Memory Compactor")
    parser.add_argument(
        "--signal", type=str, default="noisy",
        choices=["sine", "noisy", "multi_tone", "wav", "csv"],
        help="Type of input signal (default: noisy)"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to .wav or .csv file (required if --signal is wav or csv)"
    )
    parser.add_argument(
        "--streams", type=int, default=8,
        help="Number of parallel sub-streams (default: 8)"
    )
    parser.add_argument(
        "--mode", type=str, default="multi",
        choices=["single", "multi"],
        help="Processing mode: single or multi threaded (default: multi)"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run 3-way benchmark (single CPU vs multi CPU vs GPU)"
    )
    parser.add_argument(
        "--duration", type=float, default=2.0,
        help="Signal duration in seconds for synthetic signals (default: 2.0)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*50)
    print("  DSP Smart Memory Compactor")
    print(f"  Backend: {get_backend_name()}")
    print("="*50 + "\n")

    # --- Step 1: Load Signal ---
    print(f"[main] Loading signal: {args.signal}")
    if args.signal in ("wav", "csv"):
        if args.file is None:
            raise ValueError("--file is required when using wav or csv signal type")
        signal, sample_rate = get_signal(mode=args.signal, filepath=args.file)
    else:
        signal, sample_rate = get_signal(
            mode=args.signal,
            duration=args.duration,
            sample_rate=44100,
        )
    print(f"[main] Signal loaded | Samples: {len(signal)} | SR: {sample_rate} Hz")

    # --- Step 2: Analyze Signal ---
    print("\n[main] Analyzing signal...")
    analysis = analyze_signal(signal, sample_rate)
    print(f"[main] SNR: {analysis['snr_db']} dB | "
          f"Spectral Entropy: {analysis['spectral_entropy']}")

    # --- Step 3: Process in parallel streams ---
    print(f"\n[main] Processing {args.streams} sub-streams [{args.mode} mode]...")
    results, elapsed_ms = run_parallel(signal, sample_rate, args.streams, args.mode)

    # --- Step 4: Aggregate metrics ---
    metrics = aggregate_metrics(results)
    print(f"\n[main] Done in {elapsed_ms:.2f} ms")
    print(f"[main] Avg bit width: {metrics['avg_bit_width']} bits | "
          f"Compression: {metrics['overall_compression']}x")

    # --- Step 5 (optional): Benchmark ---
    benchmark_results = None
    if args.benchmark:
        from benchmark import run_benchmark
        signal_np = to_numpy(signal)
        benchmark_results = run_benchmark(signal_np, sample_rate, args.streams)

    # --- Step 6: Dashboard ---
    print("\n[main] Launching dashboard...")
    build_dashboard(signal, sample_rate, results, metrics, benchmark_results)


if __name__ == "__main__":
    main()
