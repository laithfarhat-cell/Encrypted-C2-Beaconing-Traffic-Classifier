"""Signal Processing & Temporal Beaconing Engine.

Transforms grouped connection sequences `(src_ip -> dst_ip:dst_port)` into
periodic spectral metrics (FFT SNR, Spectral Entropy, recovered period),
inter-arrival dispersion statistics (CV, MAD, Bowley skewness, inferred jitter),
payload symmetry ratios, and TLS handshake anomaly features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import hann

from core.flow_extractor import EndpointFlowGroup

FEATURE_COLUMNS: List[str] = [
    # Temporal & Spectral FFT features
    "fft_peak_freq",
    "fft_estimated_period_sec",
    "fft_peak_power",
    "fft_snr",
    "fft_spectral_entropy",
    # Inter-arrival statistical dispersion features
    "iat_mean",
    "iat_std",
    "iat_median",
    "iat_cv",
    "iat_mad",
    "iat_mad_ratio",
    "iat_bowley_skew",
    "estimated_jitter_pct",
    # Payload symmetry & regularity features
    "up_bytes_mean",
    "up_bytes_cv",
    "down_bytes_mean",
    "down_bytes_cv",
    "up_down_byte_ratio",
    "small_payload_dominance",
    "session_count",
    "duration_sec",
    # TLS handshake metadata & anomaly indicators
    "sni_present",
    "sni_entropy",
    "missing_alpn",
    "cipher_count",
    "extension_count",
    "low_extension_profile",
    "self_signed_cert",
]


@dataclass
class SpectralSummary:
    """Outputs of uniform-grid Hann-windowed rFFT periodicity analysis."""

    peak_freq: float
    estimated_period_sec: float
    peak_power: float
    snr: float
    spectral_entropy: float
    freqs: np.ndarray
    power: np.ndarray


@dataclass
class BeaconingAnalysisResult:
    """Complete feature vector + diagnostic arrays for a single endpoint flow group."""

    flow_id: str
    src_ip: str
    dst_ip: str
    dst_port: int
    protocol: str
    features: Dict[str, float]
    timestamps: np.ndarray
    iats: np.ndarray
    spectral: SpectralSummary
    sni: Optional[str]
    ja3_hash: str
    ja4_fingerprint: str
    alpn_protocols: List[str]
    label: Optional[int] = None


def compute_fft_periodicity(
    timestamps: np.ndarray,
    grid_dt_sec: float = 1.0,
) -> SpectralSummary:
    """Resample irregular timestamps onto a uniform grid and compute Hann-windowed rFFT.

    Args:
        timestamps: Sorted 1D array of session start epoch timestamps (seconds).
        grid_dt_sec: Discrete uniform sampling interval in seconds.

    Returns:
        SpectralSummary containing dominant frequency, estimated sleep period,
        spectral SNR, normalized spectral entropy, and full frequency/power arrays.
    """
    if len(timestamps) < 3:
        empty_f = np.array([0.0], dtype=np.float64)
        empty_p = np.array([0.0], dtype=np.float64)
        return SpectralSummary(
            peak_freq=0.0,
            estimated_period_sec=0.0,
            peak_power=0.0,
            snr=0.0,
            spectral_entropy=1.0,
            freqs=empty_f,
            power=empty_p,
        )

    t_rel = timestamps - timestamps[0]
    total_dur = float(t_rel[-1])
    if total_dur <= 0.0:
        empty_f = np.array([0.0], dtype=np.float64)
        empty_p = np.array([0.0], dtype=np.float64)
        return SpectralSummary(
            peak_freq=0.0,
            estimated_period_sec=0.0,
            peak_power=0.0,
            snr=0.0,
            spectral_entropy=1.0,
            freqs=empty_f,
            power=empty_p,
        )

    # Adapt dt safely so grid size remains well-resolved and memory-bounded
    dt = max(float(grid_dt_sec), total_dur / 8192.0)
    if total_dur / dt < 128:
        dt = max(0.1, total_dur / 256.0)

    n_bins = int(np.ceil(total_dur / dt)) + 1
    n_bins = max(n_bins, 64)
    grid = np.zeros(n_bins, dtype=np.float64)

    indices = np.clip(np.round(t_rel / dt).astype(int), 0, n_bins - 1)
    np.add.at(grid, indices, 1.0)

    # Smooth impulse train mildly relative to median IAT to preserve fundamental
    # peak power under random jitter while attenuating high-frequency aliasing.
    iats = np.diff(t_rel)
    pos_iats = iats[iats > 0]
    median_iat = float(np.median(pos_iats)) if len(pos_iats) > 0 else total_dur / len(timestamps)
    sigma_bins = np.clip((0.12 * median_iat) / dt, 0.8, 6.0)
    signal = gaussian_filter1d(grid, sigma=sigma_bins, mode="constant")

    # Zero-center (remove DC component) and apply Hann window
    signal = signal - np.mean(signal)
    window = hann(len(signal), sym=False)
    windowed = signal * window

    spectrum = rfft(windowed)
    freqs = rfftfreq(len(windowed), d=dt)
    power = (np.abs(spectrum) ** 2) / max(np.sum(window ** 2), 1e-12)

    # Exclude DC bin (f=0) and ultra-low frequencies near 1 cycle over whole window
    min_valid_freq = max(1.0 / max(total_dur * 0.65, 1.0), freqs[1] if len(freqs) > 1 else 0.0)
    max_valid_freq = min(0.5 / dt, 1.0 / 2.0)  # ignore sub-2s noise if dt is small
    valid_mask = (freqs >= min_valid_freq) & (freqs <= max_valid_freq)

    if not np.any(valid_mask):
        valid_mask = freqs > 0

    valid_freqs = freqs[valid_mask]
    valid_power = power[valid_mask]

    if len(valid_power) == 0 or np.sum(valid_power) <= 1e-15:
        return SpectralSummary(
            peak_freq=0.0,
            estimated_period_sec=0.0,
            peak_power=0.0,
            snr=0.0,
            spectral_entropy=1.0,
            freqs=freqs,
            power=power,
        )

    peak_local_idx = int(np.argmax(valid_power))
    peak_freq = float(valid_freqs[peak_local_idx])
    peak_power = float(valid_power[peak_local_idx])
    estimated_period = float(1.0 / peak_freq) if peak_freq > 0 else 0.0

    # Spectral SNR: ratio of peak power to mean background spectral power
    # excluding immediate +/- 2 neighboring bins of the fundamental lobe
    bg_mask = np.ones(len(valid_power), dtype=bool)
    lo = max(0, peak_local_idx - 2)
    hi = min(len(valid_power), peak_local_idx + 3)
    bg_mask[lo:hi] = False
    bg_power = valid_power[bg_mask] if np.any(bg_mask) else valid_power
    noise_floor = float(np.mean(bg_power))
    snr = float(peak_power / max(noise_floor, 1e-12))
    snr = min(snr, 1e4)

    # Normalized Spectral Entropy in [0, 1]
    total_p = float(np.sum(valid_power))
    probs = valid_power / total_p
    probs_nz = probs[probs > 0]
    raw_entropy = float(-np.sum(probs_nz * np.log2(probs_nz)))
    max_entropy = float(np.log2(len(valid_power))) if len(valid_power) > 1 else 1.0
    spectral_entropy = float(np.clip(raw_entropy / max(max_entropy, 1e-12), 0.0, 1.0))

    return SpectralSummary(
        peak_freq=peak_freq,
        estimated_period_sec=estimated_period,
        peak_power=peak_power,
        snr=snr,
        spectral_entropy=spectral_entropy,
        freqs=freqs,
        power=power,
    )


def _bowley_skewness(values: np.ndarray) -> float:
    """Compute Bowley quartile skewness: (Q3 + Q1 - 2*Q2) / (Q3 - Q1)."""
    if len(values) < 3:
        return 0.0
    q1, q2, q3 = np.percentile(values, [25.0, 50.0, 75.0])
    iqr = q3 - q1
    if iqr <= 1e-12:
        return 0.0
    return float((q3 + q1 - 2.0 * q2) / iqr)


def extract_group_features(
    group: EndpointFlowGroup,
    grid_dt_sec: float = 1.0,
    small_payload_max_bytes: int = 1500,
) -> BeaconingAnalysisResult:
    """Extract full temporal, spectral, payload, and TLS feature set for an EndpointFlowGroup."""
    sessions = group.sorted_sessions()
    timestamps = np.array([s.start_time for s in sessions], dtype=np.float64)
    iats = np.diff(timestamps) if len(timestamps) >= 2 else np.array([0.0], dtype=np.float64)

    # Guard against negative/zero deltas in synthetic edge cases
    iats_clean = np.maximum(iats, 0.0)

    iat_mean = float(np.mean(iats_clean))
    iat_std = float(np.std(iats_clean))
    iat_median = float(np.median(iats_clean))
    iat_cv = float(iat_std / iat_mean) if iat_mean > 1e-9 else 0.0

    iat_mad = float(np.median(np.abs(iats_clean - iat_median)))
    iat_mad_ratio = float(iat_mad / iat_median) if iat_median > 1e-9 else 0.0
    iat_bowley_skew = _bowley_skewness(iats_clean)

    # For uniform +/- j% jitter around T, IQR / median == j
    q25, q75 = np.percentile(iats_clean, [25.0, 75.0]) if len(iats_clean) >= 2 else (iat_median, iat_median)
    estimated_jitter_pct = float(np.clip(100.0 * (q75 - q25) / max(iat_median, 1e-9), 0.0, 200.0))

    spectral = compute_fft_periodicity(timestamps, grid_dt_sec=grid_dt_sec)

    # Payload symmetry & regularity
    up_bytes = np.array([s.upstream_bytes for s in sessions], dtype=np.float64)
    down_bytes = np.array([s.downstream_bytes for s in sessions], dtype=np.float64)

    up_mean = float(np.mean(up_bytes)) if len(up_bytes) > 0 else 0.0
    up_std = float(np.std(up_bytes)) if len(up_bytes) > 0 else 0.0
    up_cv = float(up_std / up_mean) if up_mean > 1e-9 else 0.0

    down_mean = float(np.mean(down_bytes)) if len(down_bytes) > 0 else 0.0
    down_std = float(np.std(down_bytes)) if len(down_bytes) > 0 else 0.0
    down_cv = float(down_std / down_mean) if down_mean > 1e-9 else 0.0

    total_up = float(np.sum(up_bytes))
    total_down = float(np.sum(down_bytes))
    up_down_ratio = float(total_up / max(total_down, 1.0))

    small_mask = (up_bytes <= small_payload_max_bytes) & (down_bytes <= small_payload_max_bytes)
    small_payload_dominance = float(np.mean(small_mask)) if len(sessions) > 0 else 0.0

    duration_sec = float(timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0.0

    # TLS indicators
    tls = group.representative_tls()
    ext_count = len(tls.extensions)
    cipher_count = len(tls.cipher_suites)
    any_self_signed = any(s.tls.self_signed_cert or s.tls.cert_anomaly for s in sessions)

    features: Dict[str, float] = {
        "fft_peak_freq": spectral.peak_freq,
        "fft_estimated_period_sec": spectral.estimated_period_sec,
        "fft_peak_power": spectral.peak_power,
        "fft_snr": spectral.snr,
        "fft_spectral_entropy": spectral.spectral_entropy,
        "iat_mean": iat_mean,
        "iat_std": iat_std,
        "iat_median": iat_median,
        "iat_cv": iat_cv,
        "iat_mad": iat_mad,
        "iat_mad_ratio": iat_mad_ratio,
        "iat_bowley_skew": iat_bowley_skew,
        "estimated_jitter_pct": estimated_jitter_pct,
        "up_bytes_mean": up_mean,
        "up_bytes_cv": up_cv,
        "down_bytes_mean": down_mean,
        "down_bytes_cv": down_cv,
        "up_down_byte_ratio": up_down_ratio,
        "small_payload_dominance": small_payload_dominance,
        "session_count": float(len(sessions)),
        "duration_sec": duration_sec,
        "sni_present": 1.0 if tls.sni_present else 0.0,
        "sni_entropy": float(tls.sni_entropy),
        "missing_alpn": 1.0 if tls.missing_alpn else 0.0,
        "cipher_count": float(cipher_count),
        "extension_count": float(ext_count),
        "low_extension_profile": 1.0 if (tls.client_hello_seen and ext_count < 6) else 0.0,
        "self_signed_cert": 1.0 if any_self_signed else 0.0,
    }

    return BeaconingAnalysisResult(
        flow_id=group.flow_id,
        src_ip=group.src_ip,
        dst_ip=group.dst_ip,
        dst_port=group.dst_port,
        protocol=group.protocol,
        features=features,
        timestamps=timestamps,
        iats=iats_clean,
        spectral=spectral,
        sni=tls.sni,
        ja3_hash=tls.ja3_hash,
        ja4_fingerprint=tls.ja4_fingerprint,
        alpn_protocols=list(tls.alpn_protocols),
        label=group.label,
    )


def extract_features_batch(
    groups: Iterable[EndpointFlowGroup],
    grid_dt_sec: float = 1.0,
    small_payload_max_bytes: int = 1500,
) -> Tuple[pd.DataFrame, List[BeaconingAnalysisResult]]:
    """Convert an iterable of `EndpointFlowGroup` into a DataFrame + result list."""
    results: List[BeaconingAnalysisResult] = []
    rows: List[Dict[str, float]] = []

    for group in groups:
        res = extract_group_features(
            group,
            grid_dt_sec=grid_dt_sec,
            small_payload_max_bytes=small_payload_max_bytes,
        )
        results.append(res)
        rows.append({col: res.features[col] for col in FEATURE_COLUMNS})

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    return df, results
