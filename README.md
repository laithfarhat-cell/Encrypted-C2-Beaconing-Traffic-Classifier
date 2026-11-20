# Encrypted C2 & Beaconing Traffic Classifier

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-Calibrated%20XGBoost-FF6600?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Scapy](https://img.shields.io/badge/Packet%20Engine-Scapy%20PcapReader-2E8B57)](https://scapy.net/)
[![SciPy FFT](https://img.shields.io/badge/Signal%20Processing-SciPy%20rFFT-8CA1AF?logo=scipy&logoColor=white)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](pyproject.toml)

A non-intrusive, production-ready network threat-hunting and machine learning pipeline for detecting covert Command-and-Control (C2) beaconing channels inside encrypted TLS/HTTPS traffic **without performing TLS decryption or MitM interception**.

---

## Architecture Pipeline

```text
  .pcap / .pcapng / .jsonl
            │
            ▼
┌────────────────────────────────────────────────────────┐
│ 1. Streaming Flow Aggregator & TLS Fingerprinter       │
│    • scapy.utils.PcapReader (memory-safe streaming)    │
│    • Group outbound sessions by (src, dst, port, proto)│
│    • Parse ClientHello / ServerHello / X.509 Certs     │
│    • Compute JA3, JA4-style fingerprint, ALPN, SNI H() │
└────────────────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│ 2. Signal Processing & Temporal Beaconing Engine       │
│    • Δt sequence & uniform-grid Hann-windowed rFFT     │
│    • Spectral SNR, Spectral Entropy, recovered period  │
│    • IAT CV, MAD, Bowley Skewness, inferred Jitter %   │
│    • Payload symmetry & small-heartbeat dominance      │
└────────────────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│ 3. Calibrated XGBoost Classifier & TreeSHAP Explainer  │
│    • Stratified CV + Platt Calibration + Threshold Opt │
│    • Per-alert TreeSHAP feature attributions           │
└────────────────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│ 4. Threat Hunting Dashboard, Plots & SIEM Reports      │
│    • Rich terminal table of ranked C2 suspects         │
│    • Dark-mode PNGs: beacon_iat.png & fft_spectrum.png │
│    • ECS JSON (c2_detections.json) & Standalone HTML   │
└────────────────────────────────────────────────────────┘
```

---

## Key Capabilities

- **Memory-Safe Streaming Extraction (`core/flow_extractor.py`)**: Streams large `.pcap`/`.pcapng` files packet-by-packet via `PcapReader` and also ingests normalized `.jsonl` flow logs. Extracts unencrypted TLS ClientHello metadata (JA3, JA4-style string, GREASE-filtered cipher/extension lists, ALPN, SNI Shannon entropy) and self-signed certificate indicators when ServerHello/Certificate records are observable.
- **Jitter-Resilient FFT Beaconing Engine (`features/beaconing_signal.py`)**: Resamples irregular connection initiations onto a uniform time grid, applies Gaussian kernel smoothing + Hann windowing + `scipy.fft.rfft`, and extracts Spectral SNR, Spectral Entropy, and estimated sleep period $T_{\text{est}}$ alongside statistical dispersion ($CV$, MAD, Bowley skewness, inferred jitter %) and payload symmetry metrics.
- **Calibrated XGBoost + TreeSHAP (`models/c2_classifier.py`)**: Handles class imbalance (`scale_pos_weight`), calibrates probabilities via stratified `CalibratedClassifierCV`, auto-tunes the decision threshold on out-of-fold predictions, and produces per-alert TreeSHAP attributions (`pred_contribs=True`).
- **Forensic Plots & Reports (`visualizers/beacon_plots.py`, `reporters/exporter.py`)**: Generates dark-mode `matplotlib` charts (`beacon_iat.png`, `fft_spectrum.png`), Elastic Common Schema (ECS) JSON (`c2_detections.json`), and standalone HTML threat-hunting reports with embedded plots.

---

## Quickstart

### 1. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Synthetic Training & Hunt PCAPs

```bash
python main.py generate-sim --out-dir ./sample_pcaps --num-benign 36 --num-c2 24
```

### 3. Train & Evaluate the Calibrated XGBoost Model

```bash
python main.py train -i ./sample_pcaps --model models/c2_xgb.joblib
python main.py evaluate --model models/c2_xgb.joblib
```

### 4. Hunt for Encrypted C2 Beacons

```bash
python main.py hunt \
  -f ./sample_pcaps/hunt_capture.pcap \
  --model models/c2_xgb.joblib \
  --plot \
  --out-dir ./reports \
  --html ./reports/report.html \
  --json ./reports/c2_detections.json
```

### 5. Run Test Suite

```bash
pytest -v
```

---

## Project Structure

```text
├── config.yaml                  # Pipeline thresholds & model hyperparameters
├── core/
│   └── flow_extractor.py        # Streaming PcapReader, JSONL loader, TLS JA3/JA4/SNI/Cert parser
├── features/
│   └── beaconing_signal.py      # FFT spectral SNR/entropy, IAT dispersion, payload symmetry
├── models/
│   └── c2_classifier.py         # Calibrated XGBoost trainer, threshold tuner & TreeSHAP explainer
├── visualizers/
│   └── beacon_plots.py          # Dark-mode beacon_iat.png & fft_spectrum.png generators
├── reporters/
│   └── exporter.py              # SIEM ECS JSON & standalone HTML report exporters
├── scripts/
│   └── generate_c2_pcaps.py     # Synthetic benign HTTPS & jittered C2 beacon PCAP generator
├── tests/
│   └── test_c2_classifier.py    # Unit & end-to-end integration tests
├── main.py                      # Rich threat-hunting CLI (generate-sim, train, hunt, evaluate)
├── THEORY.md                    # Mathematical foundations & operational limitations
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

---

## Docker Usage

```bash
docker build -t c2-hunter .
docker run --rm -v "$(pwd)/sample_pcaps:/data" c2-hunter generate-sim --out-dir /data
```

See [THEORY.md](THEORY.md) for the mathematical derivation of FFT periodicity detection under jitter, dispersion estimators, TLS ClientHello fingerprinting, and encrypted traffic analysis limitations.
