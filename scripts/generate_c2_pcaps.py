"""Synthetic PCAP & Periodic Beacon Traffic Generator.

Produces realistic `.pcap` captures containing dummy TLS/HTTPS handshake and
application-data packets for defensive ML classifier training and validation:
  - Benign Enterprise HTTPS: bursty log-normal/Pareto inter-arrival intervals,
    modern browser TLS ClientHello fingerprints (GREASE, ALPN `h2`, rich
    extensions), low-entropy enterprise SNIs, and asymmetric heavy-tailed
    downstream payloads.
  - Simulated Periodic Beacons: fixed sleep intervals (e.g. 15s, 30s, 45s, 60s)
    with configurable uniform jitter (0% to 30%), uniform small heartbeat
    payloads, minimal/anomalous TLS ClientHello profiles (missing ALPN, low
    extension count, high-entropy or absent SNI), and optional self-signed certs.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from scapy.layers.inet import IP, TCP
from scapy.layers.l2 import Ether
from scapy.packet import Raw
from scapy.utils import PcapWriter

_BENIGN_SNIS = [
    "api.github.com",
    "cdn.cloudflare-edge.net",
    "login.microsoftonline.com",
    "assets.office365.com",
    "telemetry.datadoghq.com",
    "slack-files.com",
    "storage.googleapis.com",
    "updates.mozilla.org",
    "pypi.org",
    "registry.npmjs.org",
]

_HIGH_ENTROPY_SNIS = [
    "xk9q2m7v4p8z1r.cdn-sync-node.net",
    "a7f9c2e4b8d103.edge-relay.org",
    "qz8w3x9m2k7p5n.cloud-metrics.info",
    "v4b8n2m9x7k3q1.static-assets.biz",
]

_BROWSER_CIPHERS = [
    0x0A0A,  # GREASE
    0x1301, 0x1302, 0x1303,
    0xC02B, 0xC02F, 0xC02C, 0xC030,
    0xCCA9, 0xCCA8, 0xC013, 0xC014,
    0x009C, 0x009D, 0x002F, 0x0035,
]

_BEACON_CIPHERS = [
    0xC02F, 0xC030, 0xC013, 0xC014, 0x002F,
]


@dataclass
class FlowSpec:
    """Configuration for a synthetic endpoint flow series."""

    src_ip: str
    dst_ip: str
    dst_port: int
    label: int  # 0 = Benign_HTTPS, 1 = C2_Beacon
    profile_name: str
    sleep_sec: float = 30.0
    jitter_pct: float = 0.20
    num_sessions: int = 24
    sni: Optional[str] = None
    include_self_signed_cert: bool = False


def build_tls_client_hello(
    sni: Optional[str] = "api.github.com",
    alpn_protocols: Optional[List[str]] = None,
    ciphers: Optional[List[int]] = None,
    extensions: Optional[List[int]] = None,
    tls_version: int = 0x0303,
    supported_tls13: bool = True,
) -> bytes:
    """Construct an RFC-compliant unencrypted TLS ClientHello record."""
    if ciphers is None:
        ciphers = _BROWSER_CIPHERS
    if extensions is None:
        extensions = [0x0017, 0xFF01, 0x000A, 0x000B, 0x0023, 0x000D, 0x0033, 0x002D]

    ext_chunks: List[bytes] = []

    # SNI (0x0000)
    if sni:
        sni_bytes = sni.encode("ascii")
        sni_entry = struct.pack("!BH", 0, len(sni_bytes)) + sni_bytes
        sni_list = struct.pack("!H", len(sni_entry)) + sni_entry
        ext_chunks.append(struct.pack("!HH", 0x0000, len(sni_list)) + sni_list)

    # ALPN (0x0010)
    if alpn_protocols:
        alpn_body = b"".join(struct.pack("!B", len(p)) + p.encode("ascii") for p in alpn_protocols)
        alpn_list = struct.pack("!H", len(alpn_body)) + alpn_body
        ext_chunks.append(struct.pack("!HH", 0x0010, len(alpn_list)) + alpn_list)

    # Supported Groups (0x000a)
    if 0x000A in extensions:
        groups = [0x001D, 0x0017, 0x0018]
        g_bytes = b"".join(struct.pack("!H", g) for g in groups)
        g_list = struct.pack("!H", len(g_bytes)) + g_bytes
        ext_chunks.append(struct.pack("!HH", 0x000A, len(g_list)) + g_list)

    # EC Point Formats (0x000b)
    if 0x000B in extensions:
        pt_data = b"\x01\x00"
        ext_chunks.append(struct.pack("!HH", 0x000B, len(pt_data)) + pt_data)

    # Supported Versions (0x002b)
    if supported_tls13:
        ver_list = b"\x04\x03\x04\x03\x03"
        ext_chunks.append(struct.pack("!HH", 0x002B, len(ver_list)) + ver_list)

    # Remaining placeholder extensions
    for ext_id in extensions:
        if ext_id in (0x0000, 0x0010, 0x000A, 0x000B, 0x002B):
            continue
        ext_chunks.append(struct.pack("!HH", ext_id, 2) + b"\x00\x00")

    ext_bytes = b"".join(ext_chunks)
    cipher_bytes = b"".join(struct.pack("!H", c) for c in ciphers)

    random_bytes = os.urandom(32)
    session_id = b"\x00"
    compression = b"\x01\x00"

    body = (
        struct.pack("!H", tls_version)
        + random_bytes
        + session_id
        + struct.pack("!H", len(cipher_bytes))
        + cipher_bytes
        + compression
        + struct.pack("!H", len(ext_bytes))
        + ext_bytes
    )

    hs_header = struct.pack("!B", 0x01) + len(body).to_bytes(3, "big")
    handshake = hs_header + body
    record = struct.pack("!BHH", 0x16, 0x0301, len(handshake)) + handshake
    return record


def build_tls_server_hello(tls_version: int = 0x0303, cipher: int = 0xC02F) -> bytes:
    """Construct a minimal TLS ServerHello record."""
    body = (
        struct.pack("!H", tls_version)
        + os.urandom(32)
        + b"\x00"
        + struct.pack("!H", cipher)
        + b"\x00"
    )
    hs = struct.pack("!B", 0x02) + len(body).to_bytes(3, "big") + body
    return struct.pack("!BHH", 0x16, tls_version, len(hs)) + hs


def build_self_signed_cert_der(common_name: str = "localhost") -> bytes:
    """Generate a lightweight self-signed X.509 DER certificate for testing."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=1024)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.DER)


def build_tls_certificate_record(cert_der: bytes) -> bytes:
    """Wrap a DER certificate inside a TLS Handshake Certificate (0x0b) record."""
    leaf = len(cert_der).to_bytes(3, "big") + cert_der
    certs_list = len(leaf).to_bytes(3, "big") + leaf
    hs = struct.pack("!B", 0x0B) + len(certs_list).to_bytes(3, "big") + certs_list
    return struct.pack("!BHH", 0x16, 0x0303, len(hs)) + hs


def _build_app_data_record(payload_len: int) -> bytes:
    """Construct a dummy TLS Application Data (0x17) record of given length."""
    body_len = max(16, payload_len)
    return struct.pack("!BHH", 0x17, 0x0303, body_len) + (b"\x41" * body_len)


def _emit_tls_session_packets(
    writer: PcapWriter,
    src_ip: str,
    dst_ip: str,
    sport: int,
    dport: int,
    start_ts: float,
    client_hello: bytes,
    server_hello: bytes,
    up_payload_len: int,
    down_payload_len: int,
) -> None:
    """Write a complete synthetic TCP+TLS session sequence to PcapWriter."""
    src_mac = "00:11:22:33:44:55"
    dst_mac = "66:77:88:99:aa:bb"

    # 1. TCP SYN
    p1 = Ether(src=src_mac, dst=dst_mac) / IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=dport, flags="S", seq=1000)
    p1.time = start_ts
    writer.write(p1)

    # 2. TCP SYN-ACK
    p2 = Ether(src=dst_mac, dst=src_mac) / IP(src=dst_ip, dst=src_ip) / TCP(sport=dport, dport=sport, flags="SA", seq=2000, ack=1001)
    p2.time = start_ts + 0.008
    writer.write(p2)

    # 3. ClientHello
    p3 = Ether(src=src_mac, dst=dst_mac) / IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=dport, flags="PA", seq=1001, ack=2001) / Raw(load=client_hello)
    p3.time = start_ts + 0.015
    writer.write(p3)

    # 4. ServerHello (+ optional cert)
    p4 = Ether(src=dst_mac, dst=src_mac) / IP(src=dst_ip, dst=src_ip) / TCP(sport=dport, dport=sport, flags="PA", seq=2001, ack=1001 + len(client_hello)) / Raw(load=server_hello)
    p4.time = start_ts + 0.032
    writer.write(p4)

    # 5. Encrypted Application Data upstream
    up_rec = _build_app_data_record(up_payload_len)
    p5 = Ether(src=src_mac, dst=dst_mac) / IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=dport, flags="PA", seq=1001 + len(client_hello), ack=2001 + len(server_hello)) / Raw(load=up_rec)
    p5.time = start_ts + 0.048
    writer.write(p5)

    # 6. Encrypted Application Data downstream (chunked into <=1400B segments if large)
    remaining = max(16, down_payload_len)
    t_off = 0.065
    seq_srv = 2001 + len(server_hello)
    while remaining > 0:
        chunk = min(1400, remaining)
        down_rec = _build_app_data_record(chunk)
        p6 = Ether(src=dst_mac, dst=src_mac) / IP(src=dst_ip, dst=src_ip) / TCP(sport=dport, dport=sport, flags="PA", seq=seq_srv, ack=1001 + len(client_hello) + len(up_rec)) / Raw(load=down_rec)
        p6.time = start_ts + t_off
        writer.write(p6)
        seq_srv += len(down_rec)
        remaining -= chunk
        t_off += 0.005


def write_synthetic_pcap(
    out_pcap: str | Path,
    flow_specs: List[FlowSpec],
    base_time: float = 1763640000.0,
    seed: int = 42,
) -> Dict[str, int]:
    """Generate a synthetic `.pcap` containing all specified benign and beacon flows.

    Returns mapping of `"{src_ip}->{dst_ip}:{dst_port}/TCP"` to ground-truth label.
    """
    rng = np.random.default_rng(seed)
    out_path = Path(out_pcap)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cached_self_signed_cert = build_self_signed_cert_der("update-service.local")
    label_map: Dict[str, int] = {}

    # Pre-build all packet events in chronological order for clean PCAP timestamps
    events: List[Tuple[float, FlowSpec, int, bytes, bytes, int, int]] = []

    for spec in flow_specs:
        flow_id = f"{spec.src_ip}->{spec.dst_ip}:{spec.dst_port}/TCP"
        label_map[flow_id] = spec.label

        if spec.label == 1:
            # Periodic C2 Beacon profile
            sni = spec.sni
            alpn = ["http/1.1"] if rng.random() < 0.15 else []
            ch = build_tls_client_hello(
                sni=sni,
                alpn_protocols=alpn,
                ciphers=_BEACON_CIPHERS[: int(rng.integers(4, 6))],
                extensions=[0x0017, 0x000A, 0x000B][: int(rng.integers(2, 4))],
                supported_tls13=(rng.random() < 0.3),
            )
            sh = build_tls_server_hello(cipher=0xC02F)
            if spec.include_self_signed_cert:
                sh = sh + build_tls_certificate_record(cached_self_signed_cert)

            t = base_time + float(rng.uniform(0.0, 10.0))
            base_up = int(rng.integers(140, 340))
            base_down = int(rng.integers(110, 380))
            for i in range(spec.num_sessions):
                sport = 42000 + (i * 7 + int(rng.integers(0, 100))) % 20000
                up_sz = max(64, int(base_up + rng.integers(-24, 25)))
                down_sz = max(64, int(base_down + rng.integers(-30, 31)))
                events.append((t, spec, sport, ch, sh, up_sz, down_sz))
                jitter = float(rng.uniform(-spec.jitter_pct, spec.jitter_pct))
                t += max(1.0, spec.sleep_sec * (1.0 + jitter))
        else:
            # Benign Enterprise HTTPS profile (bursty heavy-tailed IAT, modern TLS)
            sni = spec.sni or str(rng.choice(_BENIGN_SNIS))
            ch = build_tls_client_hello(
                sni=sni,
                alpn_protocols=["h2", "http/1.1"],
                ciphers=_BROWSER_CIPHERS,
                extensions=[0x0017, 0xFF01, 0x000A, 0x000B, 0x0023, 0x000D, 0x0033, 0x002D, 0x0015, 0x001B],
                supported_tls13=True,
            )
            sh = build_tls_server_hello(tls_version=0x0303, cipher=0x1301)

            t = base_time + float(rng.uniform(0.0, 15.0))
            for i in range(spec.num_sessions):
                sport = 50000 + (i * 11 + int(rng.integers(0, 100))) % 15000
                up_sz = int(rng.lognormal(mean=6.2, sigma=0.75))
                down_sz = int(rng.lognormal(mean=8.8, sigma=1.1))
                events.append((t, spec, sport, ch, sh, up_sz, down_sz))
                # Mix bursts (0.5-3s) with idle pauses (10-120s) via log-normal / Pareto
                if rng.random() < 0.45:
                    dt = float(rng.uniform(0.8, 4.5))
                else:
                    dt = float(np.clip(rng.lognormal(mean=3.2, sigma=1.15), 4.0, 180.0))
                t += dt

    events.sort(key=lambda item: item[0])

    with PcapWriter(str(out_path), append=False, sync=False) as writer:
        for t, spec, sport, ch, sh, up_sz, down_sz in events:
            _emit_tls_session_packets(
                writer=writer,
                src_ip=spec.src_ip,
                dst_ip=spec.dst_ip,
                sport=sport,
                dport=spec.dst_port,
                start_ts=t,
                client_hello=ch,
                server_hello=sh,
                up_payload_len=up_sz,
                down_payload_len=down_sz,
            )

    return label_map


def generate_synthetic_campaign(
    out_dir: str | Path,
    num_benign: int = 36,
    num_c2: int = 24,
    seed: int = 42,
) -> Dict[str, object]:
    """Generate a multi-PCAP labeled dataset plus a mixed `hunt_capture.pcap`."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Dict[str, int]] = {}

    # 1. Generate labeled training/evaluation PCAPs (split into 3 batches for streaming realism)
    benign_specs: List[FlowSpec] = []
    for i in range(num_benign):
        benign_specs.append(
            FlowSpec(
                src_ip=f"10.10.{1 + (i // 20)}.{10 + (i % 200)}",
                dst_ip=f"172.64.{10 + (i // 50)}.{20 + (i % 200)}",
                dst_port=443,
                label=0,
                profile_name="benign_https",
                num_sessions=int(rng.integers(14, 30)),
                sni=_BENIGN_SNIS[i % len(_BENIGN_SNIS)],
            )
        )

    c2_specs: List[FlowSpec] = []
    sleep_choices = [15.0, 20.0, 30.0, 45.0, 60.0]
    jitter_choices = [0.05, 0.12, 0.20, 0.25, 0.30]
    for i in range(num_c2):
        use_sni = rng.random() < 0.65
        sni = _HIGH_ENTROPY_SNIS[i % len(_HIGH_ENTROPY_SNIS)] if use_sni else None
        c2_specs.append(
            FlowSpec(
                src_ip=f"10.10.99.{10 + (i % 100)}",
                dst_ip=f"185.220.{10 + (i // 30)}.{50 + i}",
                dst_port=int(rng.choice([443, 8443, 4443])),
                label=1,
                profile_name="c2_beacon",
                sleep_sec=float(sleep_choices[i % len(sleep_choices)]),
                jitter_pct=float(jitter_choices[i % len(jitter_choices)]),
                num_sessions=int(rng.integers(16, 28)),
                sni=sni,
                include_self_signed_cert=(rng.random() < 0.5),
            )
        )

    all_specs = benign_specs + c2_specs
    rng.shuffle(all_specs)

    n_batches = 3
    batches = [all_specs[i::n_batches] for i in range(n_batches)]
    for idx, batch in enumerate(batches, start=1):
        pcap_name = f"campaign_batch_{idx}.pcap"
        labels = write_synthetic_pcap(out / pcap_name, batch, seed=seed + idx)
        manifest[pcap_name] = labels

    # 2. Generate a clean mixed `hunt_capture.pcap` for standalone hunting demos
    hunt_specs = [
        FlowSpec("10.20.1.15", "140.82.121.6", 443, 0, "benign_github", num_sessions=22, sni="api.github.com"),
        FlowSpec("10.20.1.15", "104.16.132.229", 443, 0, "benign_cdn", num_sessions=20, sni="cdn.cloudflare-edge.net"),
        FlowSpec("10.20.1.18", "52.96.165.130", 443, 0, "benign_m365", num_sessions=18, sni="login.microsoftonline.com"),
        FlowSpec("10.20.1.42", "198.51.100.77", 443, 1, "cobalt_style_30s_j25", sleep_sec=30.0, jitter_pct=0.25, num_sessions=26, sni=None, include_self_signed_cert=True),
        FlowSpec("10.20.1.88", "203.0.113.199", 8443, 1, "sliver_style_45s_j15", sleep_sec=45.0, jitter_pct=0.15, num_sessions=22, sni="xk9q2m7v4p8z1r.cdn-sync-node.net", include_self_signed_cert=True),
    ]
    hunt_labels = write_synthetic_pcap(out / "hunt_capture.pcap", hunt_specs, seed=seed + 99)

    labels_path = out / "labels.json"
    labels_path.write_text(
        json.dumps({"training_batches": manifest, "hunt_capture.pcap": hunt_labels}, indent=2),
        encoding="utf-8",
    )

    return {
        "out_dir": str(out),
        "labels_file": str(labels_path),
        "training_pcaps": [str(out / f"campaign_batch_{i}.pcap") for i in range(1, n_batches + 1)],
        "hunt_pcap": str(out / "hunt_capture.pcap"),
        "num_benign": num_benign,
        "num_c2": num_c2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic benign & C2 beacon PCAP captures.")
    parser.add_argument("--out-dir", default="./sample_pcaps", help="Output directory for PCAPs and labels.json")
    parser.add_argument("--num-benign", type=int, default=36, help="Number of benign HTTPS endpoint groups")
    parser.add_argument("--num-c2", type=int, default=24, help="Number of simulated C2 beacon groups")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    args = parser.parse_args()

    summary = generate_synthetic_campaign(
        out_dir=args.out_dir,
        num_benign=args.num_benign,
        num_c2=args.num_c2,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
