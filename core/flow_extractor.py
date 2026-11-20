"""Streaming PCAP Flow Aggregator & Unencrypted TLS Metadata Fingerprinter.

Processes `.pcap` and `.pcapng` captures sequentially using `scapy.utils.PcapReader`
without loading full captures into memory, as well as normalized JSONL flow logs.
Aggregates packets into bidirectional connection sessions and groups repeated
callbacks by `(src_ip, dst_ip, dst_port, protocol)`. Parses unencrypted TLS
ClientHello, ServerHello, and Certificate handshake records to compute JA3/JA4-style
fingerprints, ALPN presence, SNI Shannon entropy, and self-signed cert indicators.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.packet import Raw
from scapy.utils import PcapReader

# RFC 8701 GREASE values: 0x0a0a, 0x1a1a, ..., 0xfafa
_GREASE_VALUES = {
    0x0A0A, 0x1A1A, 0x2A2A, 0x3A3A, 0x4A4A, 0x5A5A, 0x6A6A, 0x7A7A,
    0x8A8A, 0x9A9A, 0xAAAA, 0xBABA, 0xCACA, 0xDADA, 0xEAEA, 0xFAFA,
}

_TLS_VERSION_MAP = {
    0x0304: "13",
    0x0303: "12",
    0x0302: "11",
    0x0301: "10",
    0x0300: "s3",
}

_COMMON_TLS_PORTS = {443, 8443, 4443, 9443, 8080, 993, 995, 465, 636}
_MAX_HANDSHAKE_BUFFER = 32768


def is_grease(val: int) -> bool:
    """Return True if an integer is an RFC 8701 GREASE code point."""
    return val in _GREASE_VALUES


def compute_shannon_entropy(text: str) -> float:
    """Compute Shannon entropy (bits/char) of a string such as an SNI hostname."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return float(entropy)


@dataclass
class TLSHandshakeMetadata:
    """Unencrypted TLS handshake observables extracted from ClientHello/ServerHello/Cert."""

    client_hello_seen: bool = False
    server_hello_seen: bool = False
    tls_version: int = 0
    negotiated_version: int = 0
    cipher_suites: List[int] = field(default_factory=list)
    extensions: List[int] = field(default_factory=list)
    elliptic_curves: List[int] = field(default_factory=list)
    ec_point_formats: List[int] = field(default_factory=list)
    alpn_protocols: List[str] = field(default_factory=list)
    sni: Optional[str] = None
    sni_present: bool = False
    sni_entropy: float = 0.0
    ja3_string: str = ""
    ja3_hash: str = ""
    ja4_fingerprint: str = ""
    cert_observed: bool = False
    self_signed_cert: bool = False
    cert_anomaly: bool = False
    cert_subject: Optional[str] = None
    cert_issuer: Optional[str] = None

    @property
    def missing_alpn(self) -> bool:
        return self.client_hello_seen and len(self.alpn_protocols) == 0


@dataclass
class SessionRecord:
    """Single bidirectional connection session between client and server."""

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    start_time: float
    end_time: float
    upstream_bytes: int = 0
    downstream_bytes: int = 0
    upstream_packets: int = 0
    downstream_packets: int = 0
    upstream_payload_sizes: List[int] = field(default_factory=list)
    downstream_payload_sizes: List[int] = field(default_factory=list)
    tls: TLSHandshakeMetadata = field(default_factory=TLSHandshakeMetadata)

    # Bounded reassembly buffers for TLS record parsing (cleared when parsed)
    _client_buf: bytearray = field(default_factory=bytearray, repr=False)
    _server_buf: bytearray = field(default_factory=bytearray, repr=False)
    _client_done: bool = field(default=False, repr=False)
    _server_done: bool = field(default=False, repr=False)

    @property
    def total_bytes(self) -> int:
        return self.upstream_bytes + self.downstream_bytes

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class EndpointFlowGroup:
    """All outbound sessions between `(src_ip, dst_ip, dst_port, protocol)`."""

    src_ip: str
    dst_ip: str
    dst_port: int
    protocol: str
    sessions: List[SessionRecord] = field(default_factory=list)
    label: Optional[int] = None  # Optional ground-truth label (0=Benign, 1=C2)

    @property
    def flow_key(self) -> Tuple[str, str, int, str]:
        return (self.src_ip, self.dst_ip, self.dst_port, self.protocol)

    @property
    def flow_id(self) -> str:
        return f"{self.src_ip}->{self.dst_ip}:{self.dst_port}/{self.protocol}"

    @property
    def session_count(self) -> int:
        return len(self.sessions)

    def sorted_sessions(self) -> List[SessionRecord]:
        return sorted(self.sessions, key=lambda s: s.start_time)

    def representative_tls(self) -> TLSHandshakeMetadata:
        """Return the most informative TLS metadata observed across sessions."""
        for s in self.sessions:
            if s.tls.client_hello_seen:
                return s.tls
        return self.sessions[0].tls if self.sessions else TLSHandshakeMetadata()


def _parse_sni_extension(ext_data: bytes) -> Optional[str]:
    """Parse TLS Server Name Indication (extension 0x0000)."""
    if len(ext_data) < 5:
        return None
    list_len = struct.unpack("!H", ext_data[:2])[0]
    offset = 2
    end = min(len(ext_data), 2 + list_len)
    while offset + 3 <= end:
        name_type = ext_data[offset]
        name_len = struct.unpack("!H", ext_data[offset + 1 : offset + 3])[0]
        offset += 3
        if offset + name_len > end:
            break
        if name_type == 0:  # host_name
            try:
                return ext_data[offset : offset + name_len].decode("ascii", errors="ignore").strip(".")
            except Exception:
                return None
        offset += name_len
    return None


def _parse_alpn_extension(ext_data: bytes) -> List[str]:
    """Parse TLS Application-Layer Protocol Negotiation (extension 0x0010)."""
    if len(ext_data) < 2:
        return []
    list_len = struct.unpack("!H", ext_data[:2])[0]
    offset = 2
    end = min(len(ext_data), 2 + list_len)
    protocols: List[str] = []
    while offset < end:
        proto_len = ext_data[offset]
        offset += 1
        if offset + proto_len > end:
            break
        proto = ext_data[offset : offset + proto_len].decode("ascii", errors="ignore")
        if proto:
            protocols.append(proto)
        offset += proto_len
    return protocols


def _parse_supported_groups(ext_data: bytes) -> List[int]:
    """Parse TLS Supported Groups / Elliptic Curves (extension 0x000a)."""
    if len(ext_data) < 2:
        return []
    list_len = struct.unpack("!H", ext_data[:2])[0]
    groups: List[int] = []
    offset = 2
    end = min(len(ext_data), 2 + list_len)
    while offset + 2 <= end:
        val = struct.unpack("!H", ext_data[offset : offset + 2])[0]
        if not is_grease(val):
            groups.append(val)
        offset += 2
    return groups


def _parse_ec_point_formats(ext_data: bytes) -> List[int]:
    """Parse TLS EC Point Formats (extension 0x000b)."""
    if len(ext_data) < 1:
        return []
    list_len = ext_data[0]
    return list(ext_data[1 : 1 + list_len])


def _parse_supported_versions(ext_data: bytes) -> Optional[int]:
    """Parse TLS 1.3 Supported Versions extension (0x002b) in ClientHello."""
    if len(ext_data) < 1:
        return None
    list_len = ext_data[0]
    offset = 1
    end = min(len(ext_data), 1 + list_len)
    versions: List[int] = []
    while offset + 2 <= end:
        val = struct.unpack("!H", ext_data[offset : offset + 2])[0]
        if not is_grease(val):
            versions.append(val)
        offset += 2
    return max(versions) if versions else None


def _compute_ja3_and_ja4(meta: TLSHandshakeMetadata, protocol: str = "TCP") -> None:
    """Populate JA3 string/MD5 and JA4-inspired fingerprint on TLSHandshakeMetadata."""
    ciphers_str = "-".join(str(c) for c in meta.cipher_suites)
    exts_str = "-".join(str(e) for e in meta.extensions)
    curves_str = "-".join(str(g) for g in meta.elliptic_curves)
    pts_str = "-".join(str(p) for p in meta.ec_point_formats)

    meta.ja3_string = f"{meta.tls_version},{ciphers_str},{exts_str},{curves_str},{pts_str}"
    meta.ja3_hash = hashlib.md5(meta.ja3_string.encode("ascii")).hexdigest()

    proto_char = "q" if protocol.upper() == "UDP" else "t"
    eff_ver = meta.negotiated_version or meta.tls_version
    ver_tag = _TLS_VERSION_MAP.get(eff_ver, "00")
    sni_flag = "d" if meta.sni_present else "i"
    c_cnt = min(99, len(meta.cipher_suites))
    e_cnt = min(99, len(meta.extensions))

    if meta.alpn_protocols:
        first_alpn = meta.alpn_protocols[0]
        alpn_tag = (first_alpn[0] + first_alpn[-1]) if len(first_alpn) >= 2 else (first_alpn[0] + "0")
    else:
        alpn_tag = "00"

    sorted_ciphers = ",".join(f"{c:04x}" for c in sorted(meta.cipher_suites))
    # JA4 excludes SNI (0x0000) and ALPN (0x0010) from the hashed extension list
    filtered_exts = sorted(e for e in meta.extensions if e not in (0x0000, 0x0010))
    sorted_exts = ",".join(f"{e:04x}" for e in filtered_exts)

    c_hash = hashlib.sha256(sorted_ciphers.encode("ascii")).hexdigest()[:12] if sorted_ciphers else "0" * 12
    e_hash = hashlib.sha256(sorted_exts.encode("ascii")).hexdigest()[:12] if sorted_exts else "0" * 12

    meta.ja4_fingerprint = f"{proto_char}{ver_tag}{sni_flag}{c_cnt:02d}{e_cnt:02d}{alpn_tag}_{c_hash}_{e_hash}"


def _parse_client_hello_body(body: bytes, meta: TLSHandshakeMetadata, protocol: str = "TCP") -> None:
    """Parse TLS Handshake ClientHello body bytes."""
    if len(body) < 38:
        return
    client_ver = struct.unpack("!H", body[:2])[0]
    offset = 34  # 2 bytes version + 32 bytes random

    # Session ID
    if offset >= len(body):
        return
    sess_id_len = body[offset]
    offset += 1 + sess_id_len

    # Cipher Suites
    if offset + 2 > len(body):
        return
    cs_len = struct.unpack("!H", body[offset : offset + 2])[0]
    offset += 2
    if offset + cs_len > len(body):
        return
    ciphers: List[int] = []
    for i in range(0, cs_len - (cs_len % 2), 2):
        val = struct.unpack("!H", body[offset + i : offset + i + 2])[0]
        if not is_grease(val):
            ciphers.append(val)
    offset += cs_len

    # Compression methods
    if offset >= len(body):
        return
    comp_len = body[offset]
    offset += 1 + comp_len

    # Extensions
    extensions: List[int] = []
    curves: List[int] = []
    ec_points: List[int] = []
    alpn: List[str] = []
    sni: Optional[str] = None
    highest_supported_ver: Optional[int] = None

    if offset + 2 <= len(body):
        ext_total_len = struct.unpack("!H", body[offset : offset + 2])[0]
        offset += 2
        ext_end = min(len(body), offset + ext_total_len)
        while offset + 4 <= ext_end:
            ext_type, ext_len = struct.unpack("!HH", body[offset : offset + 4])
            offset += 4
            if offset + ext_len > ext_end:
                break
            ext_data = body[offset : offset + ext_len]
            offset += ext_len

            if is_grease(ext_type):
                continue
            extensions.append(ext_type)

            if ext_type == 0x0000:
                sni = _parse_sni_extension(ext_data)
            elif ext_type == 0x000A:
                curves = _parse_supported_groups(ext_data)
            elif ext_type == 0x000B:
                ec_points = _parse_ec_point_formats(ext_data)
            elif ext_type == 0x0010:
                alpn = _parse_alpn_extension(ext_data)
            elif ext_type == 0x002B:
                highest_supported_ver = _parse_supported_versions(ext_data)

    meta.client_hello_seen = True
    meta.tls_version = client_ver
    meta.negotiated_version = highest_supported_ver or client_ver
    meta.cipher_suites = ciphers
    meta.extensions = extensions
    meta.elliptic_curves = curves
    meta.ec_point_formats = ec_points
    meta.alpn_protocols = alpn
    meta.sni = sni
    meta.sni_present = bool(sni)
    meta.sni_entropy = compute_shannon_entropy(sni) if sni else 0.0
    _compute_ja3_and_ja4(meta, protocol=protocol)


def _parse_certificate_body(body: bytes, meta: TLSHandshakeMetadata) -> None:
    """Parse TLS 1.2 Certificate handshake body to detect self-signed/anomalous certs."""
    if len(body) < 6:
        return
    certs_len = int.from_bytes(body[:3], "big")
    offset = 3
    end = min(len(body), 3 + certs_len)
    if offset + 3 > end:
        return
    leaf_len = int.from_bytes(body[offset : offset + 3], "big")
    offset += 3
    if offset + leaf_len > end or leaf_len < 16:
        return
    der_bytes = body[offset : offset + leaf_len]
    try:
        cert = x509.load_der_x509_certificate(der_bytes, default_backend())
        meta.cert_observed = True
        meta.cert_subject = cert.subject.rfc4514_string()
        meta.cert_issuer = cert.issuer.rfc4514_string()
        is_self_signed = cert.subject == cert.issuer
        meta.self_signed_cert = is_self_signed
        validity_days = (cert.not_valid_after_utc - cert.not_valid_before_utc).days
        meta.cert_anomaly = is_self_signed or validity_days > 825 or validity_days < 1
    except Exception:
        # Malformed or synthetic minimal DER cert bytes -> flag anomaly if cert header seen
        meta.cert_observed = True
        meta.cert_anomaly = True


def parse_tls_handshake_records(
    payload: bytes,
    meta: Optional[TLSHandshakeMetadata] = None,
    protocol: str = "TCP",
) -> TLSHandshakeMetadata:
    """Parse one or more TLS handshake records from a raw TCP/UDP payload buffer."""
    if meta is None:
        meta = TLSHandshakeMetadata()

    offset = 0
    total = len(payload)
    while offset + 5 <= total:
        content_type = payload[offset]
        # TLS Handshake (0x16) and valid major TLS version (0x03)
        if content_type != 0x16 or payload[offset + 1] != 0x03:
            break
        rec_len = struct.unpack("!H", payload[offset + 3 : offset + 5])[0]
        rec_start = offset + 5
        rec_end = rec_start + rec_len
        if rec_end > total:
            break

        hs_offset = rec_start
        while hs_offset + 4 <= rec_end:
            hs_type = payload[hs_offset]
            hs_len = int.from_bytes(payload[hs_offset + 1 : hs_offset + 4], "big")
            body_start = hs_offset + 4
            body_end = body_start + hs_len
            if body_end > rec_end:
                break
            body = payload[body_start:body_end]
            if hs_type == 0x01 and not meta.client_hello_seen:
                _parse_client_hello_body(body, meta, protocol=protocol)
            elif hs_type == 0x02:
                meta.server_hello_seen = True
                if len(body) >= 2:
                    meta.negotiated_version = struct.unpack("!H", body[:2])[0]
            elif hs_type == 0x0B and not meta.cert_observed:
                _parse_certificate_body(body, meta)
            hs_offset = body_end

        offset = rec_end

    return meta


def _looks_like_tls_client_hello(payload: bytes) -> bool:
    return len(payload) >= 9 and payload[0] == 0x16 and payload[1] == 0x03 and payload[5] == 0x01


def _looks_like_tls_server_hello(payload: bytes) -> bool:
    return len(payload) >= 9 and payload[0] == 0x16 and payload[1] == 0x03 and payload[5] == 0x02


def extract_flows_from_pcap(
    pcap_path: str | Path,
    session_idle_gap_sec: float = 5.0,
    min_sessions: int = 1,
) -> Dict[Tuple[str, str, int, str], EndpointFlowGroup]:
    """Stream a `.pcap` / `.pcapng` file via `PcapReader` and group outbound flows.

    Args:
        pcap_path: Path to `.pcap` or `.pcapng` file.
        session_idle_gap_sec: Idle timeout splitting consecutive sessions on reused ports.
        min_sessions: Filter out endpoint groups with fewer than `min_sessions` callbacks.

    Returns:
        Dictionary mapping `(src_ip, dst_ip, dst_port, protocol)` to `EndpointFlowGroup`.
    """
    pcap_path = Path(pcap_path)
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    # Active 5-tuple sessions: (client_ip, server_ip, client_port, server_port, proto) -> SessionRecord
    active_sessions: Dict[Tuple[str, str, int, int, str], SessionRecord] = {}
    groups: Dict[Tuple[str, str, int, str], EndpointFlowGroup] = {}

    def _get_or_create_group(src_ip: str, dst_ip: str, dst_port: int, proto: str) -> EndpointFlowGroup:
        key = (src_ip, dst_ip, dst_port, proto)
        if key not in groups:
            groups[key] = EndpointFlowGroup(
                src_ip=src_ip,
                dst_ip=dst_ip,
                dst_port=dst_port,
                protocol=proto,
            )
        return groups[key]

    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            if IP in pkt:
                ip_layer = pkt[IP]
                src_ip = str(ip_layer.src)
                dst_ip = str(ip_layer.dst)
            elif IPv6 in pkt:
                ip_layer = pkt[IPv6]
                src_ip = str(ip_layer.src)
                dst_ip = str(ip_layer.dst)
            else:
                continue

            ts = float(pkt.time)
            payload = b""
            tcp_flags = 0

            if TCP in pkt:
                l4 = pkt[TCP]
                proto = "TCP"
                sport = int(l4.sport)
                dport = int(l4.dport)
                tcp_flags = int(l4.flags)
                if Raw in l4:
                    payload = bytes(l4[Raw].load)
                elif bytes(l4.payload):
                    payload = bytes(l4.payload)
            elif UDP in pkt:
                l4 = pkt[UDP]
                proto = "UDP"
                sport = int(l4.sport)
                dport = int(l4.dport)
                if Raw in l4:
                    payload = bytes(l4[Raw].load)
                elif bytes(l4.payload):
                    payload = bytes(l4.payload)
            else:
                continue

            fwd_tuple = (src_ip, dst_ip, sport, dport, proto)
            rev_tuple = (dst_ip, src_ip, dport, sport, proto)

            is_syn_only = proto == "TCP" and (tcp_flags & 0x02) != 0 and (tcp_flags & 0x10) == 0

            # Determine direction relative to existing active session
            if rev_tuple in active_sessions and not is_syn_only:
                sess = active_sessions[rev_tuple]
                is_upstream = False
            elif fwd_tuple in active_sessions:
                sess = active_sessions[fwd_tuple]
                # If a brand-new SYN arrives or idle gap exceeded, open a new session
                if is_syn_only or (ts - sess.end_time > session_idle_gap_sec and _looks_like_tls_client_hello(payload)):
                    sess = SessionRecord(
                        src_ip=src_ip,
                        dst_ip=dst_ip,
                        src_port=sport,
                        dst_port=dport,
                        protocol=proto,
                        start_time=ts,
                        end_time=ts,
                    )
                    active_sessions[fwd_tuple] = sess
                    _get_or_create_group(src_ip, dst_ip, dport, proto).sessions.append(sess)
                is_upstream = True
            else:
                # Infer client vs server on unseen 5-tuple
                is_client_origin = (
                    is_syn_only
                    or _looks_like_tls_client_hello(payload)
                    or (
                        not _looks_like_tls_server_hello(payload)
                        and (dport in _COMMON_TLS_PORTS or (sport >= 1024 and dport < sport))
                    )
                )
                if is_client_origin:
                    sess = SessionRecord(
                        src_ip=src_ip,
                        dst_ip=dst_ip,
                        src_port=sport,
                        dst_port=dport,
                        protocol=proto,
                        start_time=ts,
                        end_time=ts,
                    )
                    active_sessions[fwd_tuple] = sess
                    _get_or_create_group(src_ip, dst_ip, dport, proto).sessions.append(sess)
                    is_upstream = True
                else:
                    # Reverse packet seen first without SYN/ClientHello
                    sess = SessionRecord(
                        src_ip=dst_ip,
                        dst_ip=src_ip,
                        src_port=dport,
                        dst_port=sport,
                        protocol=proto,
                        start_time=ts,
                        end_time=ts,
                    )
                    active_sessions[rev_tuple] = sess
                    _get_or_create_group(dst_ip, src_ip, sport, proto).sessions.append(sess)
                    is_upstream = False

            sess.end_time = max(sess.end_time, ts)
            payload_len = len(payload)

            if is_upstream:
                sess.upstream_packets += 1
                sess.upstream_bytes += payload_len
                if payload_len > 0:
                    sess.upstream_payload_sizes.append(payload_len)
                    if not sess._client_done and len(sess._client_buf) < _MAX_HANDSHAKE_BUFFER:
                        sess._client_buf.extend(payload[: _MAX_HANDSHAKE_BUFFER - len(sess._client_buf)])
                        parse_tls_handshake_records(bytes(sess._client_buf), sess.tls, protocol=proto)
                        if sess.tls.client_hello_seen:
                            sess._client_done = True
                            sess._client_buf.clear()
            else:
                sess.downstream_packets += 1
                sess.downstream_bytes += payload_len
                if payload_len > 0:
                    sess.downstream_payload_sizes.append(payload_len)
                    if not sess._server_done and len(sess._server_buf) < _MAX_HANDSHAKE_BUFFER:
                        sess._server_buf.extend(payload[: _MAX_HANDSHAKE_BUFFER - len(sess._server_buf)])
                        parse_tls_handshake_records(bytes(sess._server_buf), sess.tls, protocol=proto)
                        if sess.tls.cert_observed:
                            sess._server_done = True
                            sess._server_buf.clear()

    if min_sessions > 1:
        groups = {k: g for k, g in groups.items() if g.session_count >= min_sessions}

    return groups


def load_jsonl_flows(
    jsonl_path: str | Path,
    min_sessions: int = 1,
) -> Dict[Tuple[str, str, int, str], EndpointFlowGroup]:
    """Load normalized JSONL flow logs into `EndpointFlowGroup` objects.

    Expected JSONL fields per line (flexible defaults supported):
      - `timestamp` (float epoch seconds) or `start_time`
      - `src_ip`, `dst_ip`, `dst_port`, `protocol` (default "TCP")
      - `src_port` (optional)
      - `up_bytes` / `upstream_bytes`, `down_bytes` / `downstream_bytes`
      - `sni`, `alpn` (list or str), `ja3_hash`, `ja4_fingerprint`,
        `cipher_count`, `extension_count`, `self_signed_cert`, `label` (optional)
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL flow file not found: {jsonl_path}")

    groups: Dict[Tuple[str, str, int, str], EndpointFlowGroup] = {}
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            rec: Dict[str, Any] = json.loads(line)
            src_ip = str(rec["src_ip"])
            dst_ip = str(rec["dst_ip"])
            dst_port = int(rec.get("dst_port", 443))
            proto = str(rec.get("protocol", "TCP")).upper()
            ts = float(rec.get("timestamp", rec.get("start_time", 0.0)))
            duration = float(rec.get("duration", 0.1))

            up_bytes = int(rec.get("up_bytes", rec.get("upstream_bytes", 0)))
            down_bytes = int(rec.get("down_bytes", rec.get("downstream_bytes", 0)))

            alpn_raw = rec.get("alpn", rec.get("alpn_protocols", []))
            if isinstance(alpn_raw, str):
                alpn_list = [alpn_raw] if alpn_raw else []
            else:
                alpn_list = list(alpn_raw or [])

            sni = rec.get("sni")
            cipher_count = int(rec.get("cipher_count", 0))
            ext_count = int(rec.get("extension_count", 0))

            tls_meta = TLSHandshakeMetadata(
                client_hello_seen=bool(rec.get("client_hello_seen", True)),
                tls_version=int(rec.get("tls_version", 0x0303)),
                cipher_suites=list(range(cipher_count)),
                extensions=list(range(ext_count)),
                alpn_protocols=alpn_list,
                sni=sni,
                sni_present=bool(sni),
                sni_entropy=compute_shannon_entropy(sni) if sni else 0.0,
                ja3_hash=str(rec.get("ja3_hash", "")),
                ja4_fingerprint=str(rec.get("ja4_fingerprint", "")),
                cert_observed=bool(rec.get("cert_observed", False)),
                self_signed_cert=bool(rec.get("self_signed_cert", False)),
                cert_anomaly=bool(rec.get("cert_anomaly", rec.get("self_signed_cert", False))),
            )

            sess = SessionRecord(
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=int(rec.get("src_port", 50000 + (line_no % 10000))),
                dst_port=dst_port,
                protocol=proto,
                start_time=ts,
                end_time=ts + duration,
                upstream_bytes=up_bytes,
                downstream_bytes=down_bytes,
                upstream_packets=int(rec.get("up_packets", 1)),
                downstream_packets=int(rec.get("down_packets", 1)),
                upstream_payload_sizes=[up_bytes] if up_bytes > 0 else [],
                downstream_payload_sizes=[down_bytes] if down_bytes > 0 else [],
                tls=tls_meta,
            )

            key = (src_ip, dst_ip, dst_port, proto)
            if key not in groups:
                groups[key] = EndpointFlowGroup(
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    dst_port=dst_port,
                    protocol=proto,
                    label=rec.get("label"),
                )
            groups[key].sessions.append(sess)

    if min_sessions > 1:
        groups = {k: g for k, g in groups.items() if g.session_count >= min_sessions}

    return groups
