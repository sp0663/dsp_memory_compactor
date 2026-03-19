# dashboard.py
# Plotly interactive dashboard — signal view, FFT, adaptive bit-width, benchmark

import numpy as np
from utils import to_numpy


def build_dashboard(signal, sample_rate, stream_results, metrics, benchmark_results=None):
    """
    Build and display the full Plotly dashboard.

    signal           : original signal array
    sample_rate      : int
    stream_results   : list of per-chunk dicts from parallel_stream
    metrics          : aggregated metrics dict from aggregate_metrics()
    benchmark_results: optional dict from benchmark.run_benchmark()
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from dsp_processor import compute_fft

    signal_np = to_numpy(signal)
    t = np.linspace(0, len(signal_np) / sample_rate, len(signal_np))

    # Determine subplot layout
    has_benchmark = benchmark_results is not None
    rows = 3 if not has_benchmark else 4
    subplot_titles = [
        "Original Signal",
        "FFT — Frequency Spectrum",
        "Adaptive Bit-Width per Chunk",
        "Signal Complexity per Chunk",
    ]
    if has_benchmark:
        subplot_titles.append("Benchmark: Processing Time (ms)")

    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    # --- Plot 1: Original Signal ---
    fig.add_trace(go.Scatter(
        x=t[:5000], y=signal_np[:5000],
        mode='lines', name='Signal',
        line=dict(color='#00bcd4', width=1)
    ), row=1, col=1)

    # --- Plot 2: FFT Spectrum ---
    freqs, mags = compute_fft(signal, sample_rate)
    freqs_np = to_numpy(freqs)
    mags_np = to_numpy(mags)

    fig.add_trace(go.Scatter(
        x=freqs_np, y=mags_np,
        mode='lines', name='FFT Magnitude',
        line=dict(color='#ff9800', width=1),
        fill='tozeroy'
    ), row=1, col=2)

    # --- Plot 3: Bit-width per chunk ---
    chunk_indices = [r["chunk_index"] for r in stream_results]
    bit_widths = [r["bit_width"] for r in stream_results]

    fig.add_trace(go.Bar(
        x=chunk_indices, y=bit_widths,
        name='Bit Width',
        marker_color='#4caf50',
        showlegend=True,
    ), row=2, col=1)

    fig.update_yaxes(title_text="Bits", range=[0, 18], row=2, col=1)

    # --- Plot 4: Complexity per chunk ---
    complexities = [r["complexity"] for r in stream_results]

    fig.add_trace(go.Scatter(
        x=chunk_indices, y=complexities,
        mode='lines+markers', name='Complexity',
        line=dict(color='#e91e63', width=2),
        marker=dict(size=6)
    ), row=2, col=2)

    fig.update_yaxes(title_text="Complexity Score", range=[0, 1], row=2, col=2)

    # --- Plot 5: Compression ratio per chunk ---
    compression_ratios = [r["compression_ratio"] for r in stream_results]

    fig.add_trace(go.Bar(
        x=chunk_indices, y=compression_ratios,
        name='Compression Ratio',
        marker_color='#9c27b0',
    ), row=3, col=1)

    fig.update_yaxes(title_text="Ratio (x)", row=3, col=1)

    # --- Plot 6: SNR per chunk ---
    snr_values = [r["snr_db"] for r in stream_results]

    fig.add_trace(go.Scatter(
        x=chunk_indices, y=snr_values,
        mode='lines+markers', name='SNR (dB)',
        line=dict(color='#2196f3', width=2),
    ), row=3, col=2)

    fig.update_yaxes(title_text="SNR (dB)", row=3, col=2)

    # --- Plot 7 (optional): Benchmark ---
    if has_benchmark:
        from benchmark import get_benchmark_labels
        labels, values = get_benchmark_labels(benchmark_results)
        colors = ['#607d8b', '#4caf50', '#ff5722'][:len(labels)]

        fig.add_trace(go.Bar(
            x=labels, y=values,
            name='Processing Time',
            marker_color=colors,
            text=[f"{v:.1f} ms" for v in values],
            textposition='auto',
        ), row=4, col=1)

        fig.update_yaxes(title_text="Time (ms)", row=4, col=1)

    # --- Layout ---
    fig.update_layout(
        title=dict(
            text="DSP Smart Memory Compactor — Live Dashboard",
            font=dict(size=20)
        ),
        height=300 * rows,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", y=-0.05),
        margin=dict(t=80, b=60, l=60, r=40),
    )

    # Add summary annotation
    summary_text = (
        f"Avg Bit Width: {metrics['avg_bit_width']} bits | "
        f"Overall Compression: {metrics['overall_compression']}x | "
        f"Chunks: {metrics['num_chunks']}"
    )
    fig.add_annotation(
        text=summary_text,
        xref="paper", yref="paper",
        x=0.5, y=1.02,
        showarrow=False,
        font=dict(size=12, color="#aaaaaa"),
        xanchor="center"
    )

    fig.show()
    print("[dashboard] Dashboard displayed successfully.")
