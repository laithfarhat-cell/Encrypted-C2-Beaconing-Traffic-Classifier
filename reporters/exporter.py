"""SIEM-Ready ECS JSON and Standalone HTML Threat Hunting Report Exporters."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from features.beaconing_signal import BeaconingAnalysisResult
from models.c2_classifier import PredictionResult
from visualizers.beacon_plots import image_to_base64


@dataclass
class DetectionRecord:
    """Combined flow analysis + calibrated XGBoost prediction + optional plot paths."""

    analysis: BeaconingAnalysisResult
    prediction: PredictionResult
    iat_plot_path: Optional[Path] = None
    fft_plot_path: Optional[Path] = None


def _iso_utc(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def export_ecs_json(
    detections: List[DetectionRecord],
    output_path: str | Path,
    only_flagged: bool = False,
) -> Path:
    """Export SIEM-ready Elastic Common Schema (ECS) JSON alerts (`c2_detections.json`)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    events: List[Dict[str, Any]] = []
    for rec in detections:
        if only_flagged and not rec.prediction.is_beacon:
            continue

        a = rec.analysis
        p = rec.prediction
        last_ts = float(a.timestamps[-1]) if len(a.timestamps) > 0 else datetime.now(timezone.utc).timestamp()

        event: Dict[str, Any] = {
            "@timestamp": _iso_utc(last_ts),
            "ecs": {"version": "8.11.0"},
            "event": {
                "kind": "alert" if p.is_beacon else "event",
                "category": ["network", "intrusion_detection"],
                "type": ["connection", "indicator"],
                "severity": int(round(p.probability * 100)),
                "outcome": "suspect_c2_beacon" if p.is_beacon else "benign_https",
            },
            "source": {"ip": a.src_ip},
            "destination": {"ip": a.dst_ip, "port": a.dst_port},
            "network": {
                "transport": a.protocol.lower(),
                "protocol": "tls",
                "direction": "outbound",
            },
            "tls": {
                "client": {
                    "server_name": a.sni,
                    "ja3": a.ja3_hash or None,
                    "ja4": a.ja4_fingerprint or None,
                    "alpn": a.alpn_protocols,
                }
            },
            "threat": {
                "framework": "MITRE ATT&CK",
                "tactic": {"name": "Command and Control"} if p.is_beacon else None,
            },
            "c2_classifier": {
                "classification": p.label,
                "confidence": round(p.probability, 4),
                "threshold": round(p.threshold, 4),
                "session_count": int(a.features["session_count"]),
                "estimated_period_sec": round(a.features["fft_estimated_period_sec"], 2),
                "estimated_jitter_pct": round(a.features["estimated_jitter_pct"], 2),
                "fft_snr": round(a.features["fft_snr"], 3),
                "fft_spectral_entropy": round(a.features["fft_spectral_entropy"], 4),
                "iat_cv": round(a.features["iat_cv"], 4),
                "up_down_byte_ratio": round(a.features["up_down_byte_ratio"], 4),
                "top_attributions": [
                    {
                        "feature": attr.feature,
                        "value": round(attr.value, 4),
                        "shap_contribution": round(attr.shap_value, 4),
                    }
                    for attr in p.top_attributions
                ],
            },
        }
        events.append(event)

    out.write_text(json.dumps(events, indent=2), encoding="utf-8")
    return out


def export_html_report(
    detections: List[DetectionRecord],
    output_path: str | Path,
    capture_source: str = "capture.pcap",
    threshold: float = 0.5,
) -> Path:
    """Export standalone dark-themed HTML Threat Hunting Report with embedded plots."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    sorted_recs = sorted(detections, key=lambda r: r.prediction.probability, reverse=True)
    flagged = [r for r in sorted_recs if r.prediction.is_beacon]
    benign_count = len(sorted_recs) - len(flagged)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    rows_html: List[str] = []
    plots_html: List[str] = []

    for idx, rec in enumerate(sorted_recs, start=1):
        a = rec.analysis
        p = rec.prediction
        badge_cls = "badge-danger" if p.is_beacon else "badge-ok"
        badge_txt = "C2 BEACON" if p.is_beacon else "BENIGN HTTPS"

        sni_disp = html.escape(a.sni) if a.sni else '<span class="muted">&lt;missing&gt;</span>'
        ja4_disp = html.escape(a.ja4_fingerprint or "N/A")
        ja3_disp = html.escape(a.ja3_hash or "N/A")
        alpn_disp = html.escape(", ".join(a.alpn_protocols)) if a.alpn_protocols else '<span class="warn">none</span>'

        attr_chips = "".join(
            f'<span class="chip {"chip-pos" if attr.shap_value > 0 else "chip-neg"}">'
            f"{html.escape(attr.format_short())}</span>"
            for attr in p.top_attributions
        )

        rows_html.append(
            f"""
            <tr>
              <td><span class="badge {badge_cls}">{badge_txt}</span></td>
              <td class="mono">{html.escape(a.src_ip)} &rarr; {html.escape(a.dst_ip)}:{a.dst_port}</td>
              <td>{sni_disp}</td>
              <td class="mono small">JA4: {ja4_disp}<br>JA3: {ja3_disp}<br>ALPN: {alpn_disp}</td>
              <td class="num">{int(a.features['session_count'])}</td>
              <td class="num">{a.features['fft_estimated_period_sec']:.1f}s</td>
              <td class="num">{a.features['estimated_jitter_pct']:.1f}%</td>
              <td class="num">{a.features['fft_snr']:.2f}</td>
              <td class="num"><strong>{p.probability * 100:.1f}%</strong></td>
              <td>{attr_chips}</td>
            </tr>
            """
        )

        if rec.iat_plot_path and rec.iat_plot_path.exists() or rec.fft_plot_path and rec.fft_plot_path.exists():
            iat_img = (
                f'<img src="data:image/png;base64,{image_to_base64(rec.iat_plot_path)}" alt="Beacon IAT Plot">'
                if rec.iat_plot_path and rec.iat_plot_path.exists()
                else ""
            )
            fft_img = (
                f'<img src="data:image/png;base64,{image_to_base64(rec.fft_plot_path)}" alt="FFT Spectrum Plot">'
                if rec.fft_plot_path and rec.fft_plot_path.exists()
                else ""
            )
            plots_html.append(
                f"""
                <div class="plot-card">
                  <h3>Suspect #{idx}: {html.escape(a.flow_id)} &mdash; Confidence {p.probability * 100:.1f}%</h3>
                  <p class="muted">SNI: {sni_disp} | Est. Sleep: {a.features['fft_estimated_period_sec']:.2f}s |
                     Est. Jitter: {a.features['estimated_jitter_pct']:.1f}% | IAT CV: {a.features['iat_cv']:.3f}</p>
                  <div class="plot-grid">
                    {iat_img}
                    {fft_img}
                  </div>
                </div>
                """
            )

    plots_section = (
        f"""
        <section>
          <h2>Forensic Signal &amp; Periodicity Plots</h2>
          {"".join(plots_html)}
        </section>
        """
        if plots_html
        else ""
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Encrypted C2 &amp; Beaconing Threat Hunting Report</title>
  <style>
    :root {{
      --bg: #0d1117;
      --panel: #161b22;
      --border: #30363d;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --danger: #ff7b72;
      --ok: #3fb950;
      --warn: #d29922;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 28px;
      background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.5;
    }}
    header {{
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px; margin-bottom: 24px;
    }}
    h1 {{ margin: 0 0 6px 0; font-size: 1.65rem; color: var(--text); }}
    h2 {{ margin: 28px 0 12px 0; font-size: 1.2rem; color: var(--accent); }}
    .meta {{ color: var(--muted); font-size: 0.9rem; }}
    .kpi-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px; margin-bottom: 24px;
    }}
    .kpi {{
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 8px; padding: 14px 18px;
    }}
    .kpi-label {{ color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }}
    .kpi-value {{ font-size: 1.65rem; font-weight: 700; margin-top: 4px; }}
    .kpi-danger {{ color: var(--danger); }}
    .kpi-ok {{ color: var(--ok); }}
    table {{
      width: 100%; border-collapse: collapse;
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 8px; overflow: hidden; font-size: 0.88rem;
    }}
    th, td {{
      padding: 10px 12px; text-align: left;
      border-bottom: 1px solid var(--border); vertical-align: top;
    }}
    th {{ background: #1c2128; color: var(--muted); font-weight: 600; font-size: 0.8rem; text-transform: uppercase; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .small {{ font-size: 0.78rem; }}
    .muted {{ color: var(--muted); }}
    .warn {{ color: var(--warn); font-weight: 600; }}
    .badge {{
      display: inline-block; padding: 3px 8px; border-radius: 999px;
      font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; white-space: nowrap;
    }}
    .badge-danger {{ background: rgba(255,123,114,0.16); color: var(--danger); border: 1px solid rgba(255,123,114,0.45); }}
    .badge-ok {{ background: rgba(63,185,80,0.14); color: var(--ok); border: 1px solid rgba(63,185,80,0.4); }}
    .chip {{
      display: inline-block; padding: 2px 6px; margin: 2px 3px 2px 0;
      border-radius: 4px; font-family: monospace; font-size: 0.75rem;
    }}
    .chip-pos {{ background: rgba(255,123,114,0.12); color: #ffa198; border: 1px solid rgba(255,123,114,0.3); }}
    .chip-neg {{ background: rgba(88,166,255,0.12); color: #79c0ff; border: 1px solid rgba(88,166,255,0.3); }}
    .plot-card {{
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 8px; padding: 16px; margin-bottom: 18px;
    }}
    .plot-card h3 {{ margin: 0 0 6px 0; font-size: 1.02rem; }}
    .plot-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 14px; margin-top: 12px;
    }}
    .plot-grid img {{ width: 100%; border-radius: 6px; border: 1px solid var(--border); }}
  </style>
</head>
<body>
  <header>
    <h1>Encrypted C2 &amp; Beaconing Threat Hunting Report</h1>
    <div class="meta">Capture Source: <strong>{html.escape(capture_source)}</strong> &bull;
      Generated: {generated_at} &bull; Decision Threshold: {threshold:.2f}</div>
  </header>

  <section class="kpi-grid">
    <div class="kpi"><div class="kpi-label">Endpoint Pairs Analyzed</div><div class="kpi-value">{len(sorted_recs)}</div></div>
    <div class="kpi"><div class="kpi-label">Suspected C2 Beacons</div><div class="kpi-value kpi-danger">{len(flagged)}</div></div>
    <div class="kpi"><div class="kpi-label">Benign HTTPS Flows</div><div class="kpi-value kpi-ok">{benign_count}</div></div>
    <div class="kpi"><div class="kpi-label">Model Attribution</div><div class="kpi-value" style="font-size:1.1rem;padding-top:6px;">Calibrated XGBoost + TreeSHAP</div></div>
  </section>

  <section>
    <h2>Classified Endpoint Flow Pairs</h2>
    <table>
      <thead>
        <tr>
          <th>Verdict</th>
          <th>Endpoint Pair</th>
          <th>TLS SNI</th>
          <th>TLS Fingerprint (JA4 / JA3 / ALPN)</th>
          <th class="num">Sessions</th>
          <th class="num">Est. Period</th>
          <th class="num">Jitter %</th>
          <th class="num">FFT SNR</th>
          <th class="num">Confidence</th>
          <th>Top TreeSHAP Attributions</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows_html)}
      </tbody>
    </table>
  </section>

  {plots_section}
</body>
</html>
"""
    out.write_text(doc, encoding="utf-8")
    return out
