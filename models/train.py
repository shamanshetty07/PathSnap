"""Train and compare 5 candidate-classification models with trip-grouped CV.

    python -m models.train --model all

Data: outputs/features/train only (student_test is never touched).
Rows with label_obs_has_label == 1; inputs = engineered features only
(spatial_/temporal_/network_/transition_). No keys, labels or route info.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features.feature_pipeline import load_split_features
from features.feature_validation import make_group_folds
from features.schema import feature_columns

SEED = 42
N_FOLDS = 5
SVM_MAX_TRAIN = 20_000      # RBF SVM is O(n^2-n^3): trained on a fixed subsample
MODELS_DIR = Path("outputs/models")
EXP_DIR = Path("outputs/experiments")
ALL = ["logistic_regression", "random_forest", "xgboost", "lightgbm", "svm_rbf"]


class PlattSVM(ClassifierMixin, BaseEstimator):
    """RBF SVM with sigmoid-calibrated probabilities (calibrated on a held-out slice of the
    training subsample, so no validation data is used)."""

    def __init__(self, max_train=SVM_MAX_TRAIN, seed=SEED):
        self.max_train, self.seed = max_train, seed

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        idx = rng.permutation(len(X))[: self.max_train]
        cut = int(0.8 * len(idx))
        tr, ca = idx[:cut], idx[cut:]
        self.svc = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced").fit(X[tr], y[tr])
        self.platt = LogisticRegression().fit(self.svc.decision_function(X[ca])[:, None], y[ca])
        self.classes_ = self.svc.classes_
        return self

    def predict_proba(self, X):
        return self.platt.predict_proba(self.svc.decision_function(X)[:, None])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_estimator(name: str, pos_weight: float):
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    if name == "logistic_regression":
        return LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=5,
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    if name == "xgboost":
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=pos_weight, tree_method="hist",
                             n_jobs=-1, random_state=SEED, eval_metric="logloss")
    if name == "lightgbm":
        return LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.1, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=pos_weight,
                              n_jobs=-1, random_state=SEED, verbose=-1)
    if name == "svm_rbf":
        return PlattSVM()
    raise ValueError(name)


def build_pipeline(name: str, pos_weight: float):
    """Same imputer (median + missing-indicators: NaN is informative here) for all models;
    scaling only for the scale-sensitive Logistic Regression and SVM."""
    steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if name in ("logistic_regression", "svm_rbf"):
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", build_estimator(name, pos_weight)))
    return Pipeline(steps)


def top1_accuracy(df_meta: pd.DataFrame, score: np.ndarray, y: np.ndarray) -> float:
    """Per observation (trip_id, observation_index): is the highest-scoring candidate the labelled one?"""
    d = df_meta[["trip_id", "observation_index"]].copy()
    d["s"], d["y"] = score, y
    d = d.sort_values(["trip_id", "observation_index", "s"], ascending=[True, True, False], kind="stable")
    top = d.drop_duplicates(["trip_id", "observation_index"])
    return float(top["y"].mean())


def metrics(meta, y, proba, thr=0.5) -> dict:
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "log_loss": log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7), labels=[0, 1]),
            "top1_obs_accuracy": top1_accuracy(meta, proba, y),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def load_training_data():
    df = load_split_features(split="train")
    df = df[df["label_obs_has_label"] == 1].reset_index(drop=True)
    feats = feature_columns(df)
    assert not any(c.startswith(("label_", "route")) for c in feats)
    return df, feats


def importances(name, pipe, feats):
    model = pipe.named_steps["model"]
    names = list(pipe.named_steps["imputer"].get_feature_names_out(feats))
    if name == "logistic_regression":
        v = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        v = model.feature_importances_
    else:
        return None
    return pd.DataFrame({"feature": names, "importance": v}).sort_values("importance", ascending=False)


def run(models: list[str]) -> pd.DataFrame:
    for d in (MODELS_DIR, EXP_DIR):
        d.mkdir(parents=True, exist_ok=True)
    df, feats = load_training_data()
    X = df[feats].to_numpy(np.float32)
    y = df["label_aligned"].to_numpy(int)
    groups = df["trip_id"].to_numpy()
    folds = make_group_folds(groups, N_FOLDS)          # identical folds for every model
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    print(f"train rows={len(df):,} trips={len(np.unique(groups))} features={len(feats)} "
          f"pos_rate={y.mean():.3f} folds={N_FOLDS}")
    json.dump({"features": feats, "n_rows": len(df), "pos_weight": pos_weight,
               "folds": [sorted(set(groups[va])) for _, va in folds]},
              open(EXP_DIR / "setup.json", "w"), indent=1)

    summary, fold_rows = [], []
    for name in models:
        print(f"== {name}")
        cm = np.zeros((2, 2), int)
        tr_t = inf_t = 0.0
        n_val = 0
        for k, (tr, va) in enumerate(folds):
            pipe = build_pipeline(name, pos_weight)
            t0 = time.time()
            pipe.fit(X[tr], y[tr])
            ft = time.time() - t0
            t0 = time.time()
            proba = pipe.predict_proba(X[va])[:, 1]
            it = time.time() - t0
            m = metrics(df.iloc[va], y[va], proba)
            m.update(model=name, fold=k, train_time_s=ft, inference_time_s=it, n_val=len(va))
            fold_rows.append(m)
            cm += np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
            tr_t += ft
            inf_t += it
            n_val += len(va)
            print(f"  fold {k}: f1={m['f1']:.3f} top1={m['top1_obs_accuracy']:.3f} "
                  f"logloss={m['log_loss']:.3f} train={ft:.1f}s")
        fm = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        row = {"model": name}
        for c in ("accuracy", "precision", "recall", "f1", "log_loss", "top1_obs_accuracy"):
            row[c] = fm[c].mean()
            row[c + "_std"] = fm[c].std()
        row["train_time_s_per_fold"] = tr_t / len(folds)
        row["inference_ms_per_1k_rows"] = 1000.0 * inf_t / n_val * 1000
        summary.append(row)
        pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
            EXP_DIR / f"confusion_matrix_{name}.csv")
        # final model on all training rows (never student_test)
        final = build_pipeline(name, pos_weight)
        t0 = time.time()
        final.fit(X, y)
        row["final_fit_time_s"] = time.time() - t0
        joblib.dump({"pipeline": final, "features": feats}, MODELS_DIR / f"{name}.joblib")
        imp = importances(name, final, feats)
        if imp is not None:
            imp.to_csv(EXP_DIR / f"feature_importance_{name}.csv", index=False)
        print(f"  final fit {row['final_fit_time_s']:.1f}s")

    cmp_df = pd.DataFrame(summary).sort_values("top1_obs_accuracy", ascending=False)
    fold_df = pd.DataFrame(fold_rows)
    # keep earlier models' results when run one at a time
    p_cmp, p_fold = EXP_DIR / "model_comparison.csv", EXP_DIR / "cv_metrics.csv"
    if models != ALL and p_cmp.exists():
        cmp_df = pd.concat([pd.read_csv(p_cmp)[lambda d: ~d.model.isin(models)], cmp_df])
        fold_df = pd.concat([pd.read_csv(p_fold)[lambda d: ~d.model.isin(models)], fold_df])
    cmp_df = cmp_df.sort_values("top1_obs_accuracy", ascending=False)
    cmp_df.to_csv(p_cmp, index=False)
    fold_df.to_csv(p_fold, index=False)
    write_report(cmp_df, len(df), len(feats), y.mean())
    return cmp_df


def write_report(cmp_df, n_rows, n_feat, pos_rate):
    t = cmp_df[["model", "accuracy", "precision", "recall", "f1", "log_loss", "top1_obs_accuracy",
                "train_time_s_per_fold", "inference_ms_per_1k_rows"]].round(4)
    best = cmp_df.iloc[0]
    L = ["# PathSnap experiment report", "",
         f"- Data: outputs/features/train, label_obs_has_label==1 -> {n_rows:,} rows, {n_feat} features, "
         f"positive rate {pos_rate:.3f}",
         f"- {N_FOLDS}-fold GroupKFold on trip_id (identical folds for all models); student_test unused",
         "- Preprocessing: median imputation + missing indicators (all models); StandardScaler for LR/SVM",
         "- Imbalance: class_weight='balanced' (LR, RF, SVM) / scale_pos_weight (XGB, LGBM); threshold 0.5",
         f"- SVM RBF trained on a fixed {SVM_MAX_TRAIN:,}-row subsample per fit (cost), Platt-calibrated",
         "", "## Cross-validation means", "", t.to_markdown(index=False), "",
         f"**Best by observation-level top-1 accuracy: {best['model']} ({best['top1_obs_accuracy']:.4f})**", "",
         "Confusion matrices (summed over folds): outputs/experiments/confusion_matrix_*.csv. "
         "Per-fold metrics: cv_metrics.csv."]
    (EXP_DIR / "experiment_report.md").write_text("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=["all"] + ALL)
    a = ap.parse_args(argv)
    cmp_df = run(ALL if a.model == "all" else [a.model])
    print(cmp_df[["model", "f1", "log_loss", "top1_obs_accuracy", "train_time_s_per_fold"]].to_string(index=False))


if __name__ == "__main__":
    main()
