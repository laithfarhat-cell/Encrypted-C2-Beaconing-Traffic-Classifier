# Theoretical Foundations & Signal Analysis Methodology

This document formalizes the signal processing, statistical dispersion, TLS metadata fingerprinting, and calibrated gradient-boosted classification methods implemented in the **Encrypted C2 & Beaconing Traffic Classifier**.

---

## 1. Non-Intrusive Encrypted Traffic Threat Model

Modern Command-and-Control (C2) channels routinely encapsulate communications inside **TLS/HTTPS** to evade signature-based Deep Packet Inspection (DPI). Performing TLS Man-in-the-Middle (MitM) decryption introduces privacy, compliance, and certificate-pinning risks. Instead, our pipeline relies exclusively on **unencrypted transport and handshake side-channels**:

1. **Connection Initiation Timestamps** $\{t_0, t_1, \dots, t_{N-1}\}$ grouped by `(src_ip, dst_ip, dst_port, protocol)`
2. **Packet Size & Directional Symmetry** $(B_{\text{up}}, B_{\text{down}})$
3. **Unencrypted TLS Handshake Metadata** (ClientHello extensions, cipher suites, ALPN, SNI, and ServerHello/Certificate chains)

---

## 2. Periodicity Detection Under Jitter via FFT Spectral Analysis

### 2.1 Resampling Irregular Callback Timestamps

A C2 agent configured with sleep interval $T_s$ and random uniform jitter factor $j \in [0, 0.30]$ initiates connections at irregular timestamps:

$$t_k = t_{k-1} + \Delta t_k, \quad \Delta t_k \sim \mathcal{U}\big(T_s(1 - j),\; T_s(1 + j)\big)$$

Because standard Discrete Fourier Transform algorithms require uniform sampling, we map connection start timestamps onto a discrete uniform time grid with step $\Delta \tau$ (default `1.0` s):

$$g[n] = \sum_{k=0}^{N-1} \delta\!\left[n - \operatorname{round}\!\left(\frac{t_k - t_0}{\Delta \tau}\right)\right]$$

To preserve fundamental periodic energy under random phase jitter while attenuating high-frequency Dirac comb harmonics, we convolve $g[n]$ with a narrow Gaussian kernel whose standard deviation scales with the median inter-arrival interval ($\sigma \approx 0.12 \cdot \operatorname{median}(\Delta t)$), subtract the DC mean $\bar{x}$, and apply a **Hann window** $w[n]$ to suppress spectral leakage:

$$\tilde{x}[n] = (x[n] - \bar{x}) \cdot w[n], \quad w[n] = \frac{1}{2}\left(1 - \cos\frac{2\pi n}{M - 1}\right)$$

### 2.2 Power Spectral Density, SNR, and Spectral Entropy

Using `scipy.fft.rfft`, we compute the one-sided Power Spectral Density (PSD):

$$P(f_m) = \frac{|X(f_m)|^2}{\sum_{n=0}^{M-1} w[n]^2}, \quad f_m = \frac{m}{M \Delta \tau}$$

Excluding DC ($f=0$) and ultra-low frequencies near the capture duration boundary, we identify the dominant frequency peak $f_{\text{peak}} = \arg\max_f P(f)$ and recover the estimated sleep interval:

$$T_{\text{est}} = \frac{1}{f_{\text{peak}}}$$

Two spectral discriminators separate periodic heartbeats from stochastic human browsing:

- **Spectral Signal-to-Noise Ratio (SNR)**: Ratio of dominant peak power to mean background power outside the fundamental lobe ($|m - m_{\text{peak}}| > 2$):

$$\text{SNR} = \frac{P(f_{\text{peak}})}{\mu_{\text{bg}}(P)}$$

- **Normalized Spectral Entropy**: Shannon entropy of the normalized PSD distribution $p_m = P(f_m) / \sum_k P(f_k)$:

$$H_{\text{spec}} = -\frac{1}{\log_2 K} \sum_{m=1}^{K} p_m \log_2 p_m \;\in [0, 1]$$

Periodic C2 beacons concentrate spectral energy into a sharp narrowband spike ($\text{SNR} \gg 1$, low $H_{\text{spec}}$), whereas bursty human HTTPS sessions distribute energy across a broadband spectrum (high $H_{\text{spec}} \to 1$).

---

## 3. Inter-Arrival Dispersion & Jitter Estimation

Let $\Delta t_k = t_k - t_{k-1}$ for $k = 1, \dots, N-1$:

| Metric | Formula | Behavioral Interpretation |
| :--- | :--- | :--- |
| **Coefficient of Variation (CV)** | $CV = \sigma_{\Delta t} / \mu_{\Delta t}$ | Low ($0.0 - 0.25$) for jittered C2; high ($\ge 0.8$) for log-normal/Pareto human bursts |
| **Median Absolute Deviation (MAD)** | $\operatorname{median}(\|\Delta t_k - \tilde{\mu}\|)$ | Outlier-resistant dispersion measure |
| **Bowley Skewness** | $\frac{Q_3 + Q_1 - 2 Q_2}{Q_3 - Q_1}$ | Near $0$ for symmetric uniform jitter; right-skewed for heavy-tailed human idle pauses |
| **Inferred Uniform Jitter (%)** | $100 \times \frac{Q_3 - Q_1}{Q_2}$ | For $\Delta t \sim \mathcal{U}(T_s(1-j), T_s(1+j))$, $\text{IQR} = j \cdot T_s$, directly recovering $j$ |

### Payload Symmetry & Heartbeat Dominance

Human HTTPS browsing exhibits strong downstream asymmetry ($B_{\text{down}} \gg B_{\text{up}}$) and heavy-tailed variance. Empty or small C2 check-in heartbeats exhibit high **small-payload dominance**, low payload CV, and an upstream-to-downstream byte ratio closer to unity.

---

## 4. Unencrypted TLS ClientHello Fingerprinting (JA3 / JA4) & SNI Heuristics

Although application payloads are encrypted, the **TLS ClientHello** is transmitted in plaintext:

- **JA3 Fingerprint**: MD5 hash of `TLSVersion,Ciphers,Extensions,EllipticCurves,ECPointFormats` after stripping RFC 8701 GREASE values (`0x?a?a`).
- **JA4-Inspired Fingerprint**: `{proto}{tls_ver}{sni_flag}{cipher_cnt}{ext_cnt}{alpn}_{sha256_ciphers[:12]}_{sha256_exts[:12]}`, preserving protocol/ALPN semantics and sorting cipher/extension lists for resilience against TLS extension randomization.
- **Protocol Anomalies**:
  - **Missing ALPN**: Modern browsers advertise `h2` or `http/1.1`; raw HTTP/TLS libraries or custom C2 implants frequently omit ALPN (`missing_alpn = 1`).
  - **Low Extension Profile**: Minimal TLS libraries expose fewer than 6 extensions vs. 10–16 in modern browsers.
  - **SNI Shannon Entropy**: $H(\text{SNI}) = -\sum p_c \log_2 p_c$ highlights pseudo-random DGA domains or omitted SNIs (`sni_present = 0`).
  - **Certificate Indicators**: When ServerHello and Certificate (`0x0b`) handshake records appear in TLS 1.2 captures, X.509 `issuer == subject` flags self-signed infrastructure.

---

## 5. Calibrated XGBoost & TreeSHAP Attribution

Raw tree-ensemble margins are often uncalibrated under class imbalance (`scale_pos_weight`). We wrap `XGBClassifier` inside **Stratified Cross-Validated Platt Scaling (`CalibratedClassifierCV`)**, tune the operating decision threshold on out-of-fold probabilities to maximize $F_1$, and compute exact per-prediction **TreeSHAP** values (`pred_contribs=True`) so analysts receive transparent per-alert attributions.

---

## 6. Operational Limitations & Blind Spots

1. **TLS 1.3 Encrypted Certificates**: TLS 1.3 encrypts the `Certificate` handshake message; self-signed certificate detection only applies when TLS 1.2 (or unencrypted certificate exchange) is negotiated.
2. **Encrypted ClientHello (ECH)**: Widespread ECH deployment obscures outer SNI values; temporal FFT and payload symmetry features remain effective even when SNI is unavailable.
3. **Long-Sleep / Low-Rate Beacons**: Beacons with very long sleep intervals (e.g., hours/days) require capture horizons spanning at least $5\text{–}10 \times T_s$ to achieve sufficient spectral resolution.
4. **Legitimate Polling Daemons**: Software updaters, NTP/OCSP/keepalive agents, and telemetry shippers also exhibit periodicity; TLS fingerprint context (ALPN, SNI, JA4) and analyst verification remain essential.
