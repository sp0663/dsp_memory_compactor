# adaptive_bitwidth.py
# Core algorithm — dynamically decides bit precision based on signal complexity

# Bit width levels supported
BIT_WIDTH_OPTIONS = [4, 6, 8, 10, 12, 16]

# Thresholds for decision making (tunable)
SNR_THRESHOLDS = {
    "very_high" : 40,   # dB — very clean signal
    "high"      : 25,
    "medium"    : 15,
    "low"       : 5,
}

ENTROPY_THRESHOLDS = {
    "low"    : 0.3,   # simple signal
    "medium" : 0.6,
    "high"   : 0.85,  # very complex/noisy
}


def decide_bit_width(snr_db, spectral_entropy):
    """
    Core adaptive algorithm.
    Takes SNR and spectral entropy, returns the optimal bit width.

    Logic:
    - High SNR + Low entropy  → signal is simple → use fewer bits (save memory)
    - Low SNR + High entropy  → signal is complex/noisy → use more bits (preserve quality)

    Returns: (bit_width: int, reason: str)
    """

    # Compute a combined complexity score (0 = simple, 1 = complex)
    # SNR contribution: higher SNR = lower complexity
    if snr_db >= SNR_THRESHOLDS["very_high"]:
        snr_score = 0.0
    elif snr_db >= SNR_THRESHOLDS["high"]:
        snr_score = 0.25
    elif snr_db >= SNR_THRESHOLDS["medium"]:
        snr_score = 0.5
    elif snr_db >= SNR_THRESHOLDS["low"]:
        snr_score = 0.75
    else:
        snr_score = 1.0

    # Entropy contribution: higher entropy = higher complexity
    if spectral_entropy <= ENTROPY_THRESHOLDS["low"]:
        entropy_score = 0.0
    elif spectral_entropy <= ENTROPY_THRESHOLDS["medium"]:
        entropy_score = 0.4
    elif spectral_entropy <= ENTROPY_THRESHOLDS["high"]:
        entropy_score = 0.75
    else:
        entropy_score = 1.0

    # Weighted complexity score
    complexity = 0.5 * snr_score + 0.5 * entropy_score

    # Map complexity to bit width
    if complexity <= 0.2:
        bit_width = 4
        reason = "Very simple signal — minimal precision needed"
    elif complexity <= 0.4:
        bit_width = 6
        reason = "Simple signal — low precision sufficient"
    elif complexity <= 0.6:
        bit_width = 8
        reason = "Moderate complexity — standard precision"
    elif complexity <= 0.8:
        bit_width = 10
        reason = "Complex signal — high precision needed"
    else:
        bit_width = 16
        reason = "Very complex/noisy signal — maximum precision"

    return bit_width, reason, round(complexity, 3)


def adapt_stream(analysis_results):
    """
    Given a list of per-chunk analysis results, return bit width decisions for each chunk.
    analysis_results: list of dicts from dsp_processor.analyze_signal()
    Returns: list of dicts with bit_width, reason, complexity per chunk
    """
    decisions = []
    for i, analysis in enumerate(analysis_results):
        snr = analysis.get("snr_db", 20)
        entropy = analysis.get("spectral_entropy", 0.5)
        bit_width, reason, complexity = decide_bit_width(snr, entropy)
        decisions.append({
            "chunk_index" : i,
            "snr_db"      : snr,
            "entropy"     : entropy,
            "complexity"  : complexity,
            "bit_width"   : bit_width,
            "reason"      : reason,
        })
        print(f"[adaptive] Chunk {i:03d} | SNR: {snr:6.1f} dB | "
              f"Entropy: {entropy:.3f} | Complexity: {complexity:.3f} "
              f"→ {bit_width}-bit ({reason})")
    return decisions


def summarize_decisions(decisions):
    """Print a summary of bit width distribution across all chunks."""
    from collections import Counter
    bw_counts = Counter(d["bit_width"] for d in decisions)
    total = len(decisions)
    print("\n--- Adaptive Bit-Width Summary ---")
    for bw in sorted(bw_counts):
        pct = 100 * bw_counts[bw] / total
        print(f"  {bw:2d}-bit : {bw_counts[bw]:4d} chunks ({pct:.1f}%)")
    avg_bw = sum(d["bit_width"] for d in decisions) / total
    print(f"  Average bit width : {avg_bw:.2f} bits")
    print(f"  vs baseline 16-bit savings: {100*(1 - avg_bw/16):.1f}% memory saved")
    print("----------------------------------\n")
