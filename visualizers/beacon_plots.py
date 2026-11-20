"""Dark-mode Forensic Visualizations for Suspected C2 Beaconing Channels.

Generates:
  1. Callback Interval Timeline & Delta-t Distribution (`beacon_iat.png`)
  2. FFT Power Spectral Density Chart (`fft_spectrum.png`)
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from features.beaconing_signal import BeaconingAnalysisResult

# Forensic dark-mode palette
_BG_DARK = "#0d1117"
_PANEL_DARK = "#161b22"
_GRID_COLOR = "#30363d"
_TEXT_MAIN = "#e6edf3"
_TEXT_MUTED = "#8b949e"
_CYAN = "#58a6ff"
_CORAL = "#ff7b72"
_AMBER = "#d29922"
_GREEN = "#3fb950"


def _apply_dark_style(ax: plt.Axes) -> None:
    ax.set_facecolor(_PANEL_DARK)
    ax.tick_params(colors=_TEXT_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(_GRID_COLOR)
    ax.grid(True, linestyle="--", linewidth=0.6, color=_GRID_COLOR, alpha=0.6)


def plot_beacon_iat(
    result: BeaconingAnalysisResult,
    out_path: str | Path,
    confidence: Optional[float] = None,
    dpi: int = 150,
) -> Path:
    """Plot Callback Interval Timeline & Delta-t Distribution in dark forensic theme."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    timestamps = result.timestamps
    iats = result.iats
    rel_t = timestamps[1:] - timestamps[0] if len(timestamps) > 1 else np.array([0.0])

    median_iat = result.features["iat_median"]
    q25, q75 = (
        np.percentile(iats, [25.0, 75.0])
        if len(iats) >= 2
        else (median_iat, median_iat)
    )
    cv = result.features["iat_cv"]
    jitter_pct = result.features["estimated_jitter_pct"]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.8),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
        facecolor=_BG_DARK,
    )

    # Panel 1: Callback Interval Timeline
    _apply_dark_style(ax1)
    ax1.axhspan(q25, q75, color=_AMBER, alpha=0.14, label="IQR Band (25th-75th pct)")
    ax1.axhline(median_iat, color=_AMBER, linestyle="--", linewidth=1.3, label=f"Median: {median_iat:.2f}s")
    ax1.plot(rel_t, iats, color=_CYAN, linewidth=1.5, alpha=0.9, marker="o", markersize=4.5, label="Observed Δt")

    ax1.set_title("Callback Interval Timeline (Δt vs. Elapsed Time)", color=_TEXT_MAIN, fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Elapsed Capture Time (seconds)", color=_TEXT_MUTED, fontsize=9.5)
    ax1.set_ylabel("Inter-Arrival Time Δt (seconds)", color=_TEXT_MUTED, fontsize=9.5)
    ax1.legend(facecolor=_BG_DARK, edgecolor=_GRID_COLOR, labelcolor=_TEXT_MAIN, fontsize=8.5, loc="upper right")

    # Panel 2: Delta-t Distribution
    _apply_dark_style(ax2)
    bins = min(24, max(8, len(iats) // 2 + 3))
    counts, _, _ = ax2.hist(iats, bins=bins, color=_CORAL, alpha=0.72, edgecolor=_BG_DARK, linewidth=0.9)
    max_count = float(np.max(counts)) if len(counts) > 0 else 1.0
    ax2.set_ylim(0.0, max_count * 1.65)
    ax2.axvline(median_iat, color=_AMBER, linestyle="--", linewidth=1.5, label=f"Median: {median_iat:.2f}s")

    stats_text = (
        f"Mean Δt:  {result.features['iat_mean']:.2f} s\n"
        f"Std Dev:  {result.features['iat_std']:.2f} s\n"
        f"CV:       {cv:.3f}\n"
        f"MAD:      {result.features['iat_mad']:.2f} s\n"
        f"Est Jitter: ~{jitter_pct:.1f}%"
    )
    ax2.text(
        0.96,
        0.96,
        stats_text,
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=_TEXT_MAIN,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor=_BG_DARK, edgecolor=_GRID_COLOR, alpha=0.92),
    )

    ax2.set_title("Δt Distribution & Dispersion", color=_TEXT_MAIN, fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Inter-Arrival Time Δt (seconds)", color=_TEXT_MUTED, fontsize=9.5)
    ax2.set_ylabel("Callback Count", color=_TEXT_MUTED, fontsize=9.5)

    conf_str = f" | Confidence: {confidence * 100:.1f}%" if confidence is not None else ""
    sni_str = f" | SNI: {result.sni}" if result.sni else " | SNI: <missing>"
    fig.suptitle(
        f"Beacon Timing Analysis — {result.flow_id}{sni_str}{conf_str}",
        color=_TEXT_MAIN,
        fontsize=10.8,
        fontweight="bold",
        y=0.97,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=dpi, facecolor=_BG_DARK)
    plt.close(fig)
    return out


def plot_fft_spectrum(
    result: BeaconingAnalysisResult,
    out_path: str | Path,
    confidence: Optional[float] = None,
    dpi: int = 150,
) -> Path:
    """Plot dark-mode FFT Power Spectral Density highlighting recovered beacon period."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    spec = result.spectral
    freqs = spec.freqs
    power = spec.power

    # Focus on positive frequencies up to 0.25 Hz (periods >= 4s) or max valid
    mask = (freqs > 0) & (freqs <= max(0.25, spec.peak_freq * 2.5 if spec.peak_freq > 0 else 0.25))
    plot_f = freqs[mask] if np.any(mask) else freqs[freqs > 0]
    plot_p = power[mask] if np.any(mask) else power[freqs > 0]

    fig, ax = plt.subplots(figsize=(10.5, 4.8), facecolor=_BG_DARK)
    _apply_dark_style(ax)

    if len(plot_f) > 0:
        ax.fill_between(plot_f, plot_p, color=_CYAN, alpha=0.18)
        ax.plot(plot_f, plot_p, color=_CYAN, linewidth=1.6, label="Hann-Windowed PSD")
        max_p = float(np.max(plot_p)) if len(plot_p) > 0 else 1.0
        ax.set_ylim(0.0, max_p * 1.25)

        if spec.peak_freq > 0:
            ax.axvline(spec.peak_freq, color=_CORAL, linestyle="--", linewidth=1.2, alpha=0.85)
            ax.scatter([spec.peak_freq], [spec.peak_power], color=_CORAL, s=75, zorder=5, label="Dominant Peak")

            ax.annotate(
                f"Peak: {spec.peak_freq:.4f} Hz\n"
                f"Est. Sleep T ≈ {spec.estimated_period_sec:.2f} s\n"
                f"SNR: {spec.snr:.2f} | Entropy: {spec.spectral_entropy:.3f}",
                xy=(spec.peak_freq, spec.peak_power),
                xytext=(0.42, 0.74),
                textcoords="axes fraction",
                color=_TEXT_MAIN,
                fontsize=9,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=_BG_DARK, edgecolor=_CORAL, alpha=0.92),
                arrowprops=dict(arrowstyle="->", color=_CORAL, lw=1.2),
            )

    ax.set_title("FFT Power Spectral Density — Periodic Beacon Recovery", color=_TEXT_MAIN, fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xlabel("Frequency (Hz)  [Sleep Interval T = 1 / f]", color=_TEXT_MUTED, fontsize=9.5)
    ax.set_ylabel("Spectral Power |X(f)|²", color=_TEXT_MUTED, fontsize=9.5)
    ax.legend(facecolor=_BG_DARK, edgecolor=_GRID_COLOR, labelcolor=_TEXT_MAIN, fontsize=8.5, loc="upper right")

    conf_str = f" | C2 Confidence: {confidence * 100:.1f}%" if confidence is not None else ""
    fig.suptitle(
        f"Spectral Frequency Signature — {result.flow_id}{conf_str}",
        color=_TEXT_MAIN,
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=dpi, facecolor=_BG_DARK)
    plt.close(fig)
    return out


def image_to_base64(image_path: str | Path) -> str:
    """Encode a PNG image file into a base64 data URI string for HTML embedding."""
    raw = Path(image_path).read_bytes()
    return base64.b64encode(raw).decode("ascii")
