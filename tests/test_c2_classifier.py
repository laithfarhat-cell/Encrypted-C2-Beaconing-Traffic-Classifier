"""Comprehensive unit & integration test suite for Encrypted C2 Classifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.flow_extractor import (
    EndpointFlowGroup,
    SessionRecord,
    TLSHandshakeMetadata,
    compute_shannon_entropy,
    extract_flows_from_pcap,
    load_jsonl_flows,
    parse_tls_handshake_records,
)
from features.beaconing_signal import (
    FEATURE_COLUMNS,
    compute_fft_periodicity,
    extract_features_batch,
    extract_group_features,
)
from main import main as cli_main
from models.c2_classifier import C2BeaconClassifier
from reporters.exporter import DetectionRecord, export_ecs_json, export_html_report
from scripts.generate_c2_pcaps import (
    FlowSpec,
    build_self_signed_cert_der,
    build_tls_certificate_record,
    build_tls_client_hello,
    build_tls_server_hello,
    write_synthetic_pcap,
)
from visualizers.beacon_plots import plot_beacon_iat, plot_fft_spectrum


def test_tls_client_hello_and_certificate_parsing() -> None:
    ch = build_tls_client_hello(
        sni="api.example.com",
        alpn_protocols=["h2", "http/1.1"],
        ciphers=[0x0A0A, 0x1301, 0x1302, 0xC02F],
        extensions=[0x0017, 0x000A, 0x000B, 0x002B],
        supported_tls13=True,
    )
    meta = parse_tls_handshake_records(ch)

    assert meta.client_hello_seen is True
    assert meta.sni == "api.example.com"
    assert meta.sni_present is True
    assert meta.sni_entropy > 2.0
    assert meta.alpn_protocols == ["h2", "http/1.1"]
    assert meta.missing_alpn is False
    # GREASE 0x0a0a must be excluded
    assert 0x0A0A not in meta.cipher_suites
    assert len(meta.cipher_suites) == 3
    assert len(meta.ja3_hash) == 32
    assert meta.ja4_fingerprint.startswith("t13d03")

    # ServerHello + self-signed certificate
    cert_der = build_self_signed_cert_der("c2-node.local")
    srv_records = build_tls_server_hello() + build_tls_certificate_record(cert_der)
    meta = parse_tls_handshake_records(srv_records, meta=meta)
    assert meta.server_hello_seen is True
    assert meta.cert_observed is True
    assert meta.self_signed_cert is True
    assert meta.cert_anomaly is True


def test_fft_periodicity_recovery_with_jitter() -> None:
    rng = np.random.default_rng(123)
    sleep_sec = 30.0
    jitter_pct = 0.25
    timestamps = [1000.0]
    for _ in range(35):
        dt = sleep_sec * (1.0 + rng.uniform(-jitter_pct, jitter_pct))
        timestamps.append(timestamps[-1] + dt)

    spec = compute_fft_periodicity(np.array(timestamps), grid_dt_sec=1.0)
    # Recovered period should be within +/- 15% of true 30s sleep despite 25% jitter
    assert abs(spec.estimated_period_sec - sleep_sec) < 4.5
    assert spec.snr > 4.0
    assert spec.spectral_entropy < 0.85


def test_bursty_traffic_dispersion_vs_beacon() -> None:
    rng = np.random.default_rng(7)
    beacon_sessions = []
    t = 100.0
    for idx in range(20):
        beacon_sessions.append(
            SessionRecord(
                src_ip="10.0.0.5",
                dst_ip="198.51.100.10",
                src_port=40000 + idx,
                dst_port=443,
                protocol="TCP",
                start_time=t,
                end_time=t + 0.1,
                upstream_bytes=220,
                downstream_bytes=240,
                tls=TLSHandshakeMetadata(client_hello_seen=True, cipher_suites=[0xC02F], extensions=[0x0017]),
            )
        )
        t += 20.0 * (1.0 + rng.uniform(-0.15, 0.15))

    beacon_group = EndpointFlowGroup("10.0.0.5", "198.51.100.10", 443, "TCP", sessions=beacon_sessions)
    res_beacon = extract_group_features(beacon_group)

    bursty_sessions = []
    t = 100.0
    for idx in range(20):
        bursty_sessions.append(
            SessionRecord(
                src_ip="10.0.0.5",
                dst_ip="104.16.1.1",
                src_port=50000 + idx,
                dst_port=443,
                protocol="TCP",
                start_time=t,
                end_time=t + 0.5,
                upstream_bytes=550,
                downstream_bytes=int(rng.lognormal(9.0, 1.0)),
                tls=TLSHandshakeMetadata(
                    client_hello_seen=True,
                    sni="cdn.example.net",
                    sni_present=True,
                    sni_entropy=compute_shannon_entropy("cdn.example.net"),
                    alpn_protocols=["h2"],
                    cipher_suites=list(range(14)),
                    extensions=list(range(10)),
                ),
            )
        )
        t += 1.0 if idx % 3 != 0 else float(rng.uniform(45.0, 140.0))

    bursty_group = EndpointFlowGroup("10.0.0.5", "104.16.1.1", 443, "TCP", sessions=bursty_sessions)
    res_bursty = extract_group_features(bursty_group)

    assert res_beacon.features["iat_cv"] < res_bursty.features["iat_cv"]
    assert res_beacon.features["fft_snr"] > res_bursty.features["fft_snr"]
    assert res_beacon.features["small_payload_dominance"] > res_bursty.features["small_payload_dominance"]
    assert res_beacon.features["missing_alpn"] == 1.0
    assert res_bursty.features["missing_alpn"] == 0.0
    assert set(res_beacon.features.keys()) == set(FEATURE_COLUMNS)


def test_jsonl_flow_ingestion(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "flows.jsonl"
    lines = []
    for i in range(8):
        lines.append(
            json.dumps(
                {
                    "timestamp": 1700000000.0 + i * 30.0,
                    "src_ip": "10.1.1.10",
                    "dst_ip": "203.0.113.50",
                    "dst_port": 8443,
                    "protocol": "TCP",
                    "up_bytes": 210,
                    "down_bytes": 190,
                    "sni": None,
                    "alpn": [],
                    "cipher_count": 4,
                    "extension_count": 3,
                    "self_signed_cert": True,
                    "label": 1,
                }
            )
        )
    jsonl_path.write_text("\n".join(lines), encoding="utf-8")

    groups = load_jsonl_flows(jsonl_path, min_sessions=5)
    assert len(groups) == 1
    grp = next(iter(groups.values()))
    assert grp.session_count == 8
    assert grp.label == 1
    assert grp.representative_tls().self_signed_cert is True


def test_end_to_end_pcap_train_hunt_plot_and_export(tmp_path: Path) -> None:
    pcap_path = tmp_path / "mixed_test.pcap"
    specs = []
    for i in range(8):
        specs.append(FlowSpec(f"10.0.1.{10+i}", f"172.64.1.{10+i}", 443, 0, "benign", num_sessions=14))
    for i in range(6):
        specs.append(
            FlowSpec(
                f"10.0.2.{10+i}",
                f"198.51.100.{20+i}",
                443,
                1,
                "beacon",
                sleep_sec=20.0 + i * 5.0,
                jitter_pct=0.20,
                num_sessions=16,
                sni=None,
                include_self_signed_cert=(i % 2 == 0),
            )
        )

    labels = write_synthetic_pcap(pcap_path, specs, seed=99)
    groups = extract_flows_from_pcap(pcap_path, min_sessions=6)
    assert len(groups) == 14

    ordered_groups = list(groups.values())
    df, analyses = extract_features_batch(ordered_groups)
    y = [labels[g.flow_id] for g in ordered_groups]

    clf = C2BeaconClassifier(n_estimators=50, max_depth=3, cv_folds=3, random_state=42)
    clf.fit(df, y)
    assert clf.cv_metrics_ is not None
    assert clf.cv_metrics_.roc_auc >= 0.90
    assert clf.cv_metrics_.f1 >= 0.85

    model_file = tmp_path / "model.joblib"
    clf.save(model_file)
    loaded = C2BeaconClassifier.load(model_file)

    preds = loaded.explain_batch(df, top_k=5)
    assert len(preds) == 14
    assert all(len(p.top_attributions) == 5 for p in preds)

    # Check plots and reports on top suspect
    detections = [DetectionRecord(analysis=a, prediction=p) for a, p in zip(analyses, preds)]
    detections.sort(key=lambda d: d.prediction.probability, reverse=True)
    top = detections[0]
    assert top.prediction.is_beacon is True

    top.iat_plot_path = plot_beacon_iat(top.analysis, tmp_path / "beacon_iat.png", confidence=top.prediction.probability)
    top.fft_plot_path = plot_fft_spectrum(top.analysis, tmp_path / "fft_spectrum.png", confidence=top.prediction.probability)
    assert top.iat_plot_path.exists() and top.iat_plot_path.stat().st_size > 1000
    assert top.fft_plot_path.exists() and top.fft_plot_path.stat().st_size > 1000

    json_out = export_ecs_json(detections, tmp_path / "c2_detections.json")
    parsed_json = json.loads(json_out.read_text(encoding="utf-8"))
    assert len(parsed_json) == 14
    assert parsed_json[0]["c2_classifier"]["classification"] == "C2_Beacon"
    assert len(parsed_json[0]["c2_classifier"]["top_attributions"]) == 5

    html_out = export_html_report(detections, tmp_path / "report.html", capture_source=str(pcap_path))
    html_text = html_out.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html_text
    assert "C2 BEACON" in html_text


def test_cli_full_pipeline(tmp_path: Path) -> None:
    sim_dir = tmp_path / "sim"
    model_path = tmp_path / "c2_xgb.joblib"
    out_dir = tmp_path / "artifacts"
    html_path = tmp_path / "report.html"
    json_path = tmp_path / "c2_detections.json"

    assert cli_main(["generate-sim", "--out-dir", str(sim_dir), "--num-benign", "12", "--num-c2", "10", "--seed", "11"]) == 0
    assert cli_main(["train", "-i", str(sim_dir), "--model", str(model_path)]) == 0
    assert model_path.exists()

    assert cli_main(["evaluate", "--model", str(model_path)]) == 0
    assert (
        cli_main(
            [
                "hunt",
                "-f",
                str(sim_dir / "hunt_capture.pcap"),
                "--model",
                str(model_path),
                "--plot",
                "--out-dir",
                str(out_dir),
                "--html",
                str(html_path),
                "--json",
                str(json_path),
            ]
        )
        == 0
    )
    assert html_path.exists()
    assert json_path.exists()
    assert (out_dir / "beacon_iat.png").exists()
    assert (out_dir / "fft_spectrum.png").exists()
