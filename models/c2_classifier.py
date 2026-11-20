"""Calibrated XGBoost C2 Classifier & TreeSHAP Feature Attribution.

Combines temporal FFT spectral metrics, IAT statistical dispersion, payload
symmetry, and TLS ClientHello anomaly features into a stratified cross-validated,
probability-calibrated XGBoost model (`Benign_HTTPS` vs. `C2_Beacon`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from features.beaconing_signal import FEATURE_COLUMNS


@dataclass
class FeatureAttribution:
    """Single feature TreeSHAP attribution for an individual prediction."""

    feature: str
    value: float
    shap_value: float

    def format_short(self) -> str:
        val_str = f"{self.value:.2f}" if abs(self.value - round(self.value)) > 1e-4 else f"{int(round(self.value))}"
        sign = "+" if self.shap_value >= 0 else ""
        return f"{self.feature}={val_str} ({sign}{self.shap_value:.2f})"


@dataclass
class PredictionResult:
    """Calibrated prediction + per-flow TreeSHAP attributions."""

    label: str  # "C2_Beacon" or "Benign_HTTPS"
    is_beacon: bool
    probability: float
    threshold: float
    top_attributions: List[FeatureAttribution] = field(default_factory=list)


@dataclass
class EvaluationMetrics:
    """Model evaluation summary metrics."""

    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    brier_score: float
    threshold: float
    support_benign: int
    support_c2: int
    confusion_matrix: List[List[int]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class C2BeaconClassifier:
    """Calibrated XGBoost classifier with automatic threshold tuning and TreeSHAP."""

    CLASS_NAMES = ("Benign_HTTPS", "C2_Beacon")

    def __init__(
        self,
        feature_columns: Optional[Sequence[str]] = None,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.06,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        calibration_method: str = "sigmoid",
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.feature_columns: List[str] = list(feature_columns or FEATURE_COLUMNS)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.calibration_method = calibration_method
        self.cv_folds = cv_folds
        self.random_state = random_state

        self.threshold_: float = 0.5
        self.scale_pos_weight_: float = 1.0
        self.calibrated_model_: Optional[CalibratedClassifierCV] = None
        self.explainer_xgb_: Optional[xgb.XGBClassifier] = None
        self.cv_metrics_: Optional[EvaluationMetrics] = None

    def _prepare_matrix(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_columns if c not in X.columns]
            if missing:
                raise ValueError(f"Missing required feature columns: {missing}")
            arr = X[self.feature_columns].to_numpy(dtype=np.float32)
        else:
            arr = np.asarray(X, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] != len(self.feature_columns):
                raise ValueError(
                    f"Expected {len(self.feature_columns)} features, got {arr.shape[1]}"
                )
        return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)

    def _build_base_xgb(self, scale_pos_weight: float, min_child_weight: float = 1.0) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=min_child_weight,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=-1,
        )

    @staticmethod
    def _tune_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
        """Find decision threshold in [0.15, 0.85] maximizing F1 score."""
        candidates = np.linspace(0.15, 0.85, 71)
        best_thr = 0.5
        best_f1 = -1.0
        for thr in candidates:
            preds = (probs >= thr).astype(int)
            score = f1_score(y_true, preds, zero_division=0)
            if score > best_f1 + 1e-9 or (abs(score - best_f1) <= 1e-9 and abs(thr - 0.5) < abs(best_thr - 0.5)):
                best_f1 = float(score)
                best_thr = float(thr)
        return round(best_thr, 4)

    def fit(self, X: pd.DataFrame | np.ndarray, y: Sequence[int] | np.ndarray) -> "C2BeaconClassifier":
        """Train stratified CV calibrated XGBoost classifier and tune decision threshold."""
        X_arr = self._prepare_matrix(X)
        y_arr = np.asarray(y, dtype=int)

        n_pos = int(np.sum(y_arr == 1))
        n_neg = int(np.sum(y_arr == 0))
        if n_pos == 0 or n_neg == 0:
            raise ValueError("Training data must include both Benign (0) and C2_Beacon (1) classes.")

        self.scale_pos_weight_ = float(n_neg / max(n_pos, 1))
        min_class = min(n_pos, n_neg)
        n_splits = max(2, min(self.cv_folds, min_class))

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        min_cw = 0.25 if len(y_arr) < 40 else 1.0
        base_est = self._build_base_xgb(self.scale_pos_weight_, min_child_weight=min_cw)

        # Calibrated model
        cal_method = self.calibration_method
        if len(y_arr) < 40 and cal_method == "isotonic":
            cal_method = "sigmoid"

        self.calibrated_model_ = CalibratedClassifierCV(
            estimator=base_est,
            method=cal_method,
            cv=cv,
        )

        # Out-of-fold probability predictions for unbiased threshold tuning & CV evaluation.
        # On larger corpora, calibrate inside each outer fold; on small datasets (< 30),
        # evaluate base_est directly across folds to avoid degenerate sub-3-sample splits.
        oof_estimator = (
            CalibratedClassifierCV(estimator=base_est, method=cal_method, cv=max(2, min(3, min_class)))
            if len(y_arr) >= 30
            else base_est
        )
        oof_probs = cross_val_predict(
            oof_estimator,
            X_arr,
            y_arr,
            cv=cv,
            method="predict_proba",
        )[:, 1]

        self.threshold_ = self._tune_threshold(y_arr, oof_probs)
        self.cv_metrics_ = self._compute_metrics(y_arr, oof_probs, self.threshold_)

        # Fit final calibrated model and full-data explainer XGBoost booster
        self.calibrated_model_.fit(X_arr, y_arr)
        self.explainer_xgb_ = self._build_base_xgb(self.scale_pos_weight_, min_child_weight=min_cw)
        self.explainer_xgb_.fit(X_arr, y_arr)

        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return calibrated class probabilities `[P(Benign_HTTPS), P(C2_Beacon)]`."""
        if self.calibrated_model_ is None:
            raise RuntimeError("Model has not been trained or loaded.")
        X_arr = self._prepare_matrix(X)
        return self.calibrated_model_.predict_proba(X_arr)

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Return binary predictions (0=Benign_HTTPS, 1=C2_Beacon)."""
        thr = self.threshold_ if threshold is None else float(threshold)
        probs = self.predict_proba(X)[:, 1]
        return (probs >= thr).astype(int)

    def explain_batch(
        self,
        X: pd.DataFrame | np.ndarray,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[PredictionResult]:
        """Predict calibrated probabilities and compute per-flow TreeSHAP attributions."""
        if self.calibrated_model_ is None or self.explainer_xgb_ is None:
            raise RuntimeError("Model has not been trained or loaded.")

        X_arr = self._prepare_matrix(X)
        probs = self.calibrated_model_.predict_proba(X_arr)[:, 1]
        thr = self.threshold_ if threshold is None else float(threshold)

        dmat = xgb.DMatrix(X_arr, feature_names=self.feature_columns)
        # Last column of pred_contribs is the bias (expected value)
        contribs = self.explainer_xgb_.get_booster().predict(dmat, pred_contribs=True)

        results: List[PredictionResult] = []
        for idx in range(X_arr.shape[0]):
            prob = float(probs[idx])
            is_beacon = prob >= thr
            row_vals = X_arr[idx]
            row_shap = contribs[idx, :-1]

            # Sort by absolute SHAP magnitude descending (or positive contributors for C2)
            order = np.argsort(np.abs(row_shap))[::-1][:top_k]
            attrs = [
                FeatureAttribution(
                    feature=self.feature_columns[j],
                    value=float(row_vals[j]),
                    shap_value=float(row_shap[j]),
                )
                for j in order
            ]
            results.append(
                PredictionResult(
                    label=self.CLASS_NAMES[1] if is_beacon else self.CLASS_NAMES[0],
                    is_beacon=is_beacon,
                    probability=prob,
                    threshold=thr,
                    top_attributions=attrs,
                )
            )
        return results

    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray,
        probs: np.ndarray,
        threshold: float,
    ) -> EvaluationMetrics:
        preds = (probs >= threshold).astype(int)
        cm = confusion_matrix(y_true, preds, labels=[0, 1]).tolist()
        roc_auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.0
        pr_auc = float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.0
        return EvaluationMetrics(
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            precision=float(precision_score(y_true, preds, zero_division=0)),
            recall=float(recall_score(y_true, preds, zero_division=0)),
            f1=float(f1_score(y_true, preds, zero_division=0)),
            brier_score=float(brier_score_loss(y_true, probs)),
            threshold=float(threshold),
            support_benign=int(np.sum(y_true == 0)),
            support_c2=int(np.sum(y_true == 1)),
            confusion_matrix=cm,
        )

    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Sequence[int] | np.ndarray,
        threshold: Optional[float] = None,
    ) -> EvaluationMetrics:
        """Evaluate model on a labeled dataset."""
        y_arr = np.asarray(y, dtype=int)
        probs = self.predict_proba(X)[:, 1]
        thr = self.threshold_ if threshold is None else float(threshold)
        return self._compute_metrics(y_arr, probs, thr)

    def save(self, path: str | Path) -> Path:
        """Persist trained classifier, calibrator, explainer, and CV metrics via joblib."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "C2BeaconClassifier":
        """Load a persisted `C2BeaconClassifier` artifact."""
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Model file not found: {in_path}")
        obj = joblib.load(in_path)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a C2BeaconClassifier: {type(obj)}")
        return obj
