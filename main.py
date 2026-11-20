"""Encrypted C2 & Beaconing Traffic Classifier CLI.

Subcommands:
  generate-sim   Generate synthetic benign HTTPS + periodic C2 beacon PCAPs
  train          Train calibrated XGBoost classifier from labeled PCAP/JSONL corpus
  hunt           Stream a PCAP/JSONL capture, score C2 suspects, plot & export reports
  evaluate       Display ROC-AUC, PR-AUC, Precision, Recall, and F1-Score metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.flow_extractor import (
    EndpointFlowGroup,
    extract_flows_from_pcap,
    load_jsonl_flows,
)
from features.beaconing_signal import extract_features_batch
from models.c2_classifier import C2BeaconClassifier, EvaluationMetrics
from reporters.exporter import DetectionRecord, export_ecs_json, export_html_report
from scripts.generate_c2_pcaps import generate_synthetic_campaign
from visualizers.beacon_plots import plot_beacon_iat, plot_fft_spectrum

console = Console()


def _load_config(config_path: Path = Path("config.yaml")) -> Dict[str, Any]:
    if config_path.exists():
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return {}


def _load_labeled_dataset(
    input_dir: Path,
    min_sessions: int = 4,
    session_idle_gap_sec: float = 5.0,
    grid_dt_sec: float = 1.0,
    small_payload_max_bytes: int = 1500,
) -> Tuple[pd.DataFrame, List[int], List[EndpointFlowGroup]]:
    """Load labeled EndpointFlowGroups from a directory of PCAPs or JSONL logs."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_dir}")

    all_groups: List[EndpointFlowGroup] = []
    labels_file = input_dir / "labels.json"
    label_manifest: Dict[str, Dict[str, int]] = {}
    if labels_file.exists():
        raw_manifest = json.loads(labels_file.read_text(encoding="utf-8"))
        if "training_batches" in raw_manifest:
            label_manifest.update(raw_manifest["training_batches"])
        else:
            label_manifest.update(raw_manifest)

    if input_dir.is_file():
        files = [input_dir]
    else:
        files = sorted(
            [
                p
                for p in input_dir.iterdir()
                if p.suffix.lower() in {".pcap", ".pcapng", ".jsonl"}
            ]
        )

    for fpath in files:
        # Skip mixed hunt capture during training if training batches exist
        if fpath.name == "hunt_capture.pcap" and "training_batches" in (
            json.loads(labels_file.read_text(encoding="utf-8")) if labels_file.exists() else {}
        ):
            continue

        if fpath.suffix.lower() == ".jsonl":
            groups = load_jsonl_flows(fpath, min_sessions=min_sessions)
            for grp in groups.values():
                if grp.label is None:
                    grp.label = 1 if "c2" in fpath.stem.lower() or "beacon" in fpath.stem.lower() else 0
                all_groups.append(grp)
        else:
            groups = extract_flows_from_pcap(
                fpath,
                session_idle_gap_sec=session_idle_gap_sec,
                min_sessions=min_sessions,
            )
            file_labels = label_manifest.get(fpath.name, {})
            for grp in groups.values():
                if grp.flow_id in file_labels:
                    grp.label = int(file_labels[grp.flow_id])
                elif "c2" in fpath.stem.lower() or "beacon" in fpath.stem.lower():
                    grp.label = 1
                elif "benign" in fpath.stem.lower():
                    grp.label = 0
                else:
                    continue
                all_groups.append(grp)

    if not all_groups:
        raise RuntimeError(
            f"No labeled flow groups extracted from {input_dir}. "
            "Ensure PCAPs have labels.json or benign/c2 filenames."
        )

    df, _ = extract_features_batch(
        all_groups,
        grid_dt_sec=grid_dt_sec,
        small_payload_max_bytes=small_payload_max_bytes,
    )
    y = [int(g.label) for g in all_groups]  # type: ignore[arg-type]
    return df, y, all_groups


def _print_metrics_panel(metrics: EvaluationMetrics, title: str) -> None:
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("ROC-AUC", f"{metrics.roc_auc:.4f}")
    table.add_row("PR-AUC (Avg Precision)", f"{metrics.pr_auc:.4f}")
    table.add_row("Precision", f"{metrics.precision:.4f}")
    table.add_row("Recall", f"{metrics.recall:.4f}")
    table.add_row("F1-Score", f"{metrics.f1:.4f}")
    table.add_row("Brier Score (Calibration)", f"{metrics.brier_score:.4f}")
    table.add_row("Tuned Decision Threshold", f"{metrics.threshold:.4f}")
    table.add_row("Support (Benign / C2)", f"{metrics.support_benign} / {metrics.support_c2}")
    table.add_row("Confusion Matrix [[TN,FP],[FN,TP]]", str(metrics.confusion_matrix))
    console.print(table)


def _cmd_generate_sim(args: argparse.Namespace) -> int:
    console.print(Panel.fit("[bold cyan]Generating Synthetic Benign & C2 Beacon PCAP Corpus[/bold cyan]"))
    summary = generate_synthetic_campaign(
        out_dir=args.out_dir,
        num_benign=args.num_benign,
        num_c2=args.num_c2,
        seed=args.seed,
    )
    console.print(f"[green]✓[/green] Created training PCAPs & hunt capture in: [bold]{summary['out_dir']}[/bold]")
    console.print(f"  • Labels manifest : {summary['labels_file']}")
    console.print(f"  • Hunt demo PCAP  : {summary['hunt_pcap']}")
    console.print(f"  • Endpoint groups : {summary['num_benign']} Benign HTTPS, {summary['num_c2']} C2 Beacons")
    return 0


def _cmd_train(args: argparse.Namespace, cfg: Dict[str, Any]) -> int:
    flow_cfg = cfg.get("flow_extraction", {})
    sig_cfg = cfg.get("signal_processing", {})
    mdl_cfg = cfg.get("model", {})

    input_path = Path(args.input)
    console.print(f"[cyan]Streaming labeled captures from[/cyan] [bold]{input_path}[/bold] ...")
    df, y, _ = _load_labeled_dataset(
        input_path,
        min_sessions=int(flow_cfg.get("min_sessions", 6)),
        session_idle_gap_sec=float(flow_cfg.get("session_idle_gap_sec", 5.0)),
        grid_dt_sec=float(sig_cfg.get("grid_dt_sec", 1.0)),
        small_payload_max_bytes=int(sig_cfg.get("small_payload_max_bytes", 1500)),
    )

    console.print(
        f"[cyan]Training calibrated XGBoost on {len(df)} endpoint flow groups[/cyan] "
        f"({sum(y)} C2_Beacon, {len(y) - sum(y)} Benign_HTTPS)..."
    )

    clf = C2BeaconClassifier(
        n_estimators=int(mdl_cfg.get("n_estimators", 200)),
        max_depth=int(mdl_cfg.get("max_depth", 5)),
        learning_rate=float(mdl_cfg.get("learning_rate", 0.06)),
        subsample=float(mdl_cfg.get("subsample", 0.85)),
        colsample_bytree=float(mdl_cfg.get("colsample_bytree", 0.85)),
        calibration_method=str(mdl_cfg.get("calibration_method", "sigmoid")),
        cv_folds=int(mdl_cfg.get("cv_folds", 5)),
        random_state=int(mdl_cfg.get("random_state", 42)),
    )
    clf.fit(df, y)
    saved_path = clf.save(args.model)

    console.print(f"[green]✓[/green] Model saved to [bold]{saved_path}[/bold]")
    if clf.cv_metrics_ is not None:
        _print_metrics_panel(clf.cv_metrics_, "Stratified Cross-Validation Metrics")
    return 0


def _cmd_evaluate(args: argparse.Namespace, cfg: Dict[str, Any]) -> int:
    clf = C2BeaconClassifier.load(args.model)
    if args.input:
        flow_cfg = cfg.get("flow_extraction", {})
        sig_cfg = cfg.get("signal_processing", {})
        df, y, _ = _load_labeled_dataset(
            Path(args.input),
            min_sessions=int(flow_cfg.get("min_sessions", 6)),
            session_idle_gap_sec=float(flow_cfg.get("session_idle_gap_sec", 5.0)),
            grid_dt_sec=float(sig_cfg.get("grid_dt_sec", 1.0)),
            small_payload_max_bytes=int(sig_cfg.get("small_payload_max_bytes", 1500)),
        )
        metrics = clf.evaluate(df, y)
        _print_metrics_panel(metrics, f"Evaluation Metrics ({args.input})")
    elif clf.cv_metrics_ is not None:
        _print_metrics_panel(clf.cv_metrics_, f"Stored Cross-Validation Metrics ({args.model})")
    else:
        console.print("[red]No evaluation dataset provided and no stored CV metrics found in model.[/red]")
        return 1
    return 0


def _cmd_hunt(args: argparse.Namespace, cfg: Dict[str, Any]) -> int:
    flow_cfg = cfg.get("flow_extraction", {})
    sig_cfg = cfg.get("signal_processing", {})
    rep_cfg = cfg.get("reporting", {})

    capture_path = Path(args.file)
    min_sessions = int(args.min_sessions or flow_cfg.get("min_sessions", 6))

    console.print(Panel.fit(f"[bold cyan]Encrypted C2 Threat Hunt[/bold cyan]\nCapture: [bold]{capture_path}[/bold]"))

    if capture_path.suffix.lower() == ".jsonl":
        groups_dict = load_jsonl_flows(capture_path, min_sessions=min_sessions)
    else:
        groups_dict = extract_flows_from_pcap(
            capture_path,
            session_idle_gap_sec=float(flow_cfg.get("session_idle_gap_sec", 5.0)),
            min_sessions=min_sessions,
        )

    groups = list(groups_dict.values())
    if not groups:
        console.print(f"[yellow]No endpoint pairs met minimum session count ({min_sessions}).[/yellow]")
        return 0

    df, analyses = extract_features_batch(
        groups,
        grid_dt_sec=float(sig_cfg.get("grid_dt_sec", 1.0)),
        small_payload_max_bytes=int(sig_cfg.get("small_payload_max_bytes", 1500)),
    )

    clf = C2BeaconClassifier.load(args.model)
    threshold = float(args.threshold) if args.threshold is not None else clf.threshold_
    preds = clf.explain_batch(
        df,
        top_k=int(rep_cfg.get("top_shap_features", 5)),
        threshold=threshold,
    )

    detections = [DetectionRecord(analysis=a, prediction=p) for a, p in zip(analyses, preds)]
    detections.sort(key=lambda d: d.prediction.probability, reverse=True)

    # Generate diagnostic plots for flagged suspects (or top suspect if --plot requested)
    out_dir = Path(args.out_dir)
    if args.plot or args.html:
        out_dir.mkdir(parents=True, exist_ok=True)
        suspects = [d for d in detections if d.prediction.is_beacon]
        plot_targets = suspects if suspects else detections[:1]
        dpi = int(rep_cfg.get("plot_dpi", 150))

        for idx, rec in enumerate(plot_targets, start=1):
            iat_name = "beacon_iat.png" if idx == 1 else f"beacon_iat_{idx}.png"
            fft_name = "fft_spectrum.png" if idx == 1 else f"fft_spectrum_{idx}.png"
            rec.iat_plot_path = plot_beacon_iat(
                rec.analysis,
                out_dir / iat_name,
                confidence=rec.prediction.probability,
                dpi=dpi,
            )
            rec.fft_plot_path = plot_fft_spectrum(
                rec.analysis,
                out_dir / fft_name,
                confidence=rec.prediction.probability,
                dpi=dpi,
            )

    # Rich terminal dashboard
    table = Table(
        title="Ranked Endpoint Callback Analysis",
        show_lines=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("Verdict", justify="center", style="bold", no_wrap=True)
    table.add_column("Src -> Dst:Port", overflow="fold")
    table.add_column("Period", justify="right", no_wrap=True)
    table.add_column("Jitter", justify="right", no_wrap=True)
    table.add_column("TLS SNI / JA4", overflow="fold")
    table.add_column("Conf.", justify="right", no_wrap=True)
    table.add_column("Top TreeSHAP Attributions", overflow="fold")

    for rec in detections:
        a = rec.analysis
        p = rec.prediction
        verdict = "[bold red]C2 BEACON[/bold red]" if p.is_beacon else "[green]Benign HTTPS[/green]"
        sni_txt = a.sni or "<missing>"
        ja4_short = a.ja4_fingerprint[:22] if a.ja4_fingerprint else "N/A"
        tls_col = f"{sni_txt}\n[dim]{ja4_short}[/dim]"
        conf_color = "red" if p.is_beacon else "green"
        conf_txt = f"[{conf_color}]{p.probability * 100:.1f}%[/{conf_color}]"
        attr_txt = "\n".join(attr.format_short() for attr in p.top_attributions[:3])

        table.add_row(
            verdict,
            f"{a.src_ip} -> {a.dst_ip}:{a.dst_port}",
            f"{a.features['fft_estimated_period_sec']:.1f}s",
            f"{a.features['estimated_jitter_pct']:.1f}%",
            tls_col,
            conf_txt,
            attr_txt,
        )

    console.print(table)

    flagged_cnt = sum(1 for d in detections if d.prediction.is_beacon)
    console.print(
        f"\n[bold]Summary:[/bold] {len(detections)} endpoint pairs analyzed | "
        f"[bold red]{flagged_cnt} C2 suspects flagged[/bold red] (threshold={threshold:.2f})"
    )

    if args.json:
        json_path = export_ecs_json(detections, args.json)
        console.print(f"[green]✓[/green] ECS JSON detections exported to [bold]{json_path}[/bold]")

    if args.html:
        html_path = export_html_report(
            detections,
            args.html,
            capture_source=str(capture_path),
            threshold=threshold,
        )
        console.print(f"[green]✓[/green] HTML Threat Hunting Report exported to [bold]{html_path}[/bold]")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="c2-hunter",
        description="Non-intrusive Encrypted C2 & Beaconing Traffic Classifier",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # generate-sim
    p_gen = sub.add_parser("generate-sim", help="Generate synthetic benign & C2 beacon PCAPs")
    p_gen.add_argument("--out-dir", default="./sample_pcaps", help="Output directory")
    p_gen.add_argument("--num-benign", type=int, default=36, help="Benign HTTPS endpoint pairs")
    p_gen.add_argument("--num-c2", type=int, default=24, help="Simulated C2 beacon pairs")
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed")

    # train
    p_train = sub.add_parser("train", help="Train calibrated XGBoost C2 classifier")
    p_train.add_argument("-i", "--input", required=True, help="Directory of labeled PCAPs/JSONL")
    p_train.add_argument("--model", default="models/c2_xgb.joblib", help="Output model path")

    # hunt
    p_hunt = sub.add_parser("hunt", help="Hunt for encrypted C2 beaconing in a PCAP/JSONL capture")
    p_hunt.add_argument("-f", "--file", required=True, help="Path to .pcap/.pcapng or .jsonl file")
    p_hunt.add_argument("--model", default="models/c2_xgb.joblib", help="Trained model .joblib path")
    p_hunt.add_argument("--plot", action="store_true", help="Generate dark-mode IAT and FFT PNG plots")
    p_hunt.add_argument("--html", default=None, help="Output standalone HTML report path")
    p_hunt.add_argument("--json", default="c2_detections.json", help="Output SIEM ECS JSON path")
    p_hunt.add_argument("--out-dir", default="./reports", help="Directory for generated plots")
    p_hunt.add_argument("--min-sessions", type=int, default=None, help="Minimum sessions per endpoint pair")
    p_hunt.add_argument("--threshold", type=float, default=None, help="Override calibrated decision threshold")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Display classifier evaluation metrics")
    p_eval.add_argument("--model", default="models/c2_xgb.joblib", help="Trained model .joblib path")
    p_eval.add_argument("-i", "--input", default=None, help="Optional labeled PCAP/JSONL directory")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = _load_config()

    if args.command == "generate-sim":
        return _cmd_generate_sim(args)
    if args.command == "train":
        return _cmd_train(args, cfg)
    if args.command == "hunt":
        return _cmd_hunt(args, cfg)
    if args.command == "evaluate":
        return _cmd_evaluate(args, cfg)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
