"""Bayesian hyperparameter optimisation for PathSnap's Random Forest.

Run from the project root:
    venv/bin/python -m models.optimize --trials 30

Uses Optuna's TPE sampler and the same deterministic 5-fold GroupKFold
splits as models.train. Only outputs/features/train is loaded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import Pipeline

from features.feature_pipeline import load_split_features
from features.feature_validation import make_group_folds
from features.schema import feature_columns

SEED = 42
N_FOLDS = 5
OUTPUT_DIR = Path("outputs/experiments/bayesian_optimization")
MODELS_DIR = Path("outputs/models")
STUDY_DIR = OUTPUT_DIR / "study"
TRIAL_RESULTS_CSV = OUTPUT_DIR / "trial_results.csv"
BEST_HP_JSON = OUTPUT_DIR / "best_hyperparameters.json"
BEST_METRICS_JSON = OUTPUT_DIR / "best_metrics.json"
OPT_HISTORY_PNG = OUTPUT_DIR / "optimization_history.png"


def make_pipeline(params: dict[str, Any]) -> Pipeline:
    """Create a fresh leakage-safe preprocessing + Random Forest pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                RandomForestClassifier(
                    **params,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def top1_accuracy_from_proba(
    df_meta: pd.DataFrame, proba: np.ndarray, y: np.ndarray
) -> float:
    """Accuracy of the highest-scoring candidate for each GPS observation."""
    d = df_meta[["trip_id", "observation_index"]].copy()
    d["score"] = proba
    d["label"] = y
    d = d.sort_values(
        ["trip_id", "observation_index", "score"],
        ascending=[True, True, False],
        kind="stable",
    )
    top = d.drop_duplicates(["trip_id", "observation_index"], keep="first")
    return float(top["label"].mean())


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Valid RandomForest search space (compatible with current sklearn)."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 32, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
        "max_samples": trial.suggest_categorical(
            "max_samples", [None, 0.5, 0.7, 0.9]
        ),
        "class_weight": trial.suggest_categorical(
            "class_weight", ["balanced", "balanced_subsample", None]
        ),
    }


def objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Maximise mean fold-level observation Top-1 accuracy."""
    params = suggest_params(trial)
    fold_top1: list[float] = []
    fold_f1: list[float] = []
    fold_logloss: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        pipe = make_pipeline(params)
        pipe.fit(X[train_idx], y[train_idx])

        probabilities = pipe.predict_proba(X[val_idx])[:, 1]
        fold_meta = df.iloc[val_idx]

        fold_top1.append(
            top1_accuracy_from_proba(fold_meta, probabilities, y[val_idx])
        )
        fold_f1.append(
            f1_score(
                y[val_idx],
                (probabilities >= 0.5).astype(int),
                zero_division=0,
            )
        )
        fold_logloss.append(
            log_loss(
                y[val_idx],
                np.clip(probabilities, 1e-7, 1 - 1e-7),
                labels=[0, 1],
            )
        )

        trial.report(float(np.mean(fold_top1)), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_top1 = float(np.mean(fold_top1))
    trial.set_user_attr("mean_top1", mean_top1)
    trial.set_user_attr("mean_f1", float(np.mean(fold_f1)))
    trial.set_user_attr("mean_logloss", float(np.mean(fold_logloss)))
    trial.set_user_attr("fold_top1", fold_top1)
    trial.set_user_attr("fold_f1", fold_f1)
    trial.set_user_attr("fold_logloss", fold_logloss)
    return mean_top1


def evaluate_params(
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Evaluate a configuration with fresh pipelines on the fixed CV folds."""
    top1_scores: list[float] = []
    f1_scores: list[float] = []
    losses: list[float] = []
    train_times: list[float] = []

    for train_idx, val_idx in folds:
        pipe = make_pipeline(params)
        started = time.perf_counter()
        pipe.fit(X[train_idx], y[train_idx])
        train_times.append(time.perf_counter() - started)

        probabilities = pipe.predict_proba(X[val_idx])[:, 1]
        top1_scores.append(
            top1_accuracy_from_proba(df.iloc[val_idx], probabilities, y[val_idx])
        )
        f1_scores.append(
            f1_score(
                y[val_idx],
                (probabilities >= 0.5).astype(int),
                zero_division=0,
            )
        )
        losses.append(
            log_loss(
                y[val_idx],
                np.clip(probabilities, 1e-7, 1 - 1e-7),
                labels=[0, 1],
            )
        )

    return {
        "mean_top1": float(np.mean(top1_scores)),
        "std_top1": float(np.std(top1_scores, ddof=1)),
        "mean_f1": float(np.mean(f1_scores)),
        "std_f1": float(np.std(f1_scores, ddof=1)),
        "mean_logloss": float(np.mean(losses)),
        "std_logloss": float(np.std(losses, ddof=1)),
        "total_cv_train_time_s": float(np.sum(train_times)),
        "fold_top1": top1_scores,
        "fold_f1": f1_scores,
        "fold_logloss": losses,
    }


def export_trials(study: optuna.Study) -> None:
    """Export complete study history, including trials from previous runs."""
    rows = []
    for trial in study.trials:
        row = {
            "trial_id": trial.number,
            "state": trial.state.name,
            "objective_top1": trial.value,
            **trial.params,
            "mean_top1": trial.user_attrs.get("mean_top1"),
            "mean_f1": trial.user_attrs.get("mean_f1"),
            "mean_logloss": trial.user_attrs.get("mean_logloss"),
            "fold_top1": json.dumps(trial.user_attrs.get("fold_top1", [])),
            "fold_f1": json.dumps(trial.user_attrs.get("fold_f1", [])),
            "fold_logloss": json.dumps(trial.user_attrs.get("fold_logloss", [])),
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(TRIAL_RESULTS_CSV, index=False)


def plot_optimization_history(study: optuna.Study) -> None:
    complete = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
    ]
    if not complete:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(
        [trial.number for trial in complete],
        [trial.value for trial in complete],
        "o-",
        alpha=0.7,
    )
    best_so_far = np.maximum.accumulate([trial.value for trial in complete])
    plt.plot(
        [trial.number for trial in complete],
        best_so_far,
        "--",
        label="Best so far",
    )
    plt.xlabel("Trial number")
    plt.ylabel("Mean observation-level Top-1 accuracy")
    plt.title("Random Forest Bayesian Optimisation (TPE)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OPT_HISTORY_PNG, dpi=150)
    plt.close()


def run_optimization(trials: int = 30) -> tuple[optuna.Study, dict[str, Any]]:
    if trials < 1:
        raise ValueError("--trials must be at least 1")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Training split only. Unlabelled observations are excluded as in models.train.
    df = load_split_features(split="train")
    if df.empty:
        raise RuntimeError("No training features found in outputs/features/train.")
    df = df[df["label_obs_has_label"] == 1].reset_index(drop=True)
    feats = feature_columns(df)
    if not feats:
        raise RuntimeError("No engineered model features were found.")
    forbidden = [
        col for col in feats
        if col.startswith(("label_", "route_", "reference_")) or "route" in col
    ]
    if forbidden:
        raise RuntimeError(f"Potential label/route leakage in model features: {forbidden}")

    X = df[feats].to_numpy(dtype=np.float32)
    y = df["label_aligned"].to_numpy(dtype=int)
    groups = df["trip_id"].to_numpy()
    folds = make_group_folds(groups, N_FOLDS)

    # Verify every fold contains both classes before starting expensive trials.
    for fold_idx, (_, val_idx) in enumerate(folds):
        if len(np.unique(y[val_idx])) < 2:
            raise RuntimeError(
                f"Validation fold {fold_idx} contains only one class; "
                "F1/log-loss evaluation is not valid."
            )

    setup = {
        "features": feats,
        "n_rows": int(len(df)),
        "n_trips": int(df["trip_id"].nunique()),
        "positive_rate": float(y.mean()),
        "n_folds": N_FOLDS,
        "fold_validation_trips": [
            sorted(map(str, set(groups[val_idx]))) for _, val_idx in folds
        ],
        "student_test_used": False,
        "objective": "mean observation-level Top-1 accuracy",
    }
    (OUTPUT_DIR / "setup.json").write_text(json.dumps(setup, indent=2))

    study = optuna.create_study(
        study_name="bayesian_rf_optimization_v2",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
        storage=f"sqlite:///{STUDY_DIR / 'study.db'}",
        load_if_exists=True,
    )

    before_count = len(study.trials)
    study.optimize(
        lambda trial: objective(trial, X, y, df, folds),
        n_trials=trials,
        gc_after_trial=True,
        show_progress_bar=True,
    )
    new_trials = len(study.trials) - before_count
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not completed:
        raise RuntimeError("Optuna produced no completed trials; inspect the trial logs.")

    best_trial = study.best_trial
    best_params = dict(best_trial.params)

    # Evaluate baseline and tuned model on exactly the same group folds.
    from models.train import build_pipeline as build_original_pipeline

    baseline_scores: dict[str, list[float]] = {
        "top1": [], "f1": [], "logloss": [], "train_time": []
    }
    for train_idx, val_idx in folds:
        baseline = build_original_pipeline("random_forest", pos_weight=1.0)
        started = time.perf_counter()
        baseline.fit(X[train_idx], y[train_idx])
        baseline_scores["train_time"].append(time.perf_counter() - started)
        p = baseline.predict_proba(X[val_idx])[:, 1]
        baseline_scores["top1"].append(
            top1_accuracy_from_proba(df.iloc[val_idx], p, y[val_idx])
        )
        baseline_scores["f1"].append(
            f1_score(y[val_idx], (p >= 0.5).astype(int), zero_division=0)
        )
        baseline_scores["logloss"].append(
            log_loss(y[val_idx], np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])
        )

    original_metrics = {
        "mean_top1": float(np.mean(baseline_scores["top1"])),
        "mean_f1": float(np.mean(baseline_scores["f1"])),
        "mean_logloss": float(np.mean(baseline_scores["logloss"])),
        "total_cv_train_time_s": float(np.sum(baseline_scores["train_time"])),
    }
    optimized_metrics = evaluate_params(best_params, X, y, df, folds)

    # Fit the deployable final pipeline on ALL labelled training rows.
    final_pipeline = make_pipeline(best_params)
    started = time.perf_counter()
    final_pipeline.fit(X, y)
    final_fit_time = time.perf_counter() - started
    joblib.dump(
        {"pipeline": final_pipeline, "features": feats},
        MODELS_DIR / "random_forest_optimized.joblib",
    )

    metrics_payload = {
        "original_random_forest_cv": original_metrics,
        "optimized_random_forest_cv": optimized_metrics,
        "best_trial_top1": float(best_trial.value),
        "best_trial_number": int(best_trial.number),
        "new_trials_this_run": int(new_trials),
        "total_trials_in_study": int(len(study.trials)),
        "final_full_training_fit_time_s": float(final_fit_time),
        "student_test_used": False,
    }
    (BEST_HP_JSON).write_text(json.dumps(best_params, indent=2))
    BEST_METRICS_JSON.write_text(json.dumps(metrics_payload, indent=2))
    export_trials(study)
    plot_optimization_history(study)

    print("\n=== Bayesian Optimisation Complete ===")
    print(f"New trials this run: {new_trials}")
    print(f"Total trials in study: {len(study.trials)}")
    print(f"Completed trials: {len(completed)}")
    print(f"Best trial: {best_trial.number}")
    print(f"Best CV Top-1: {best_trial.value:.4f}")
    print(f"Best hyperparameters: {best_params}")
    print("\nOriginal Random Forest (5-fold GroupKFold):")
    print(
        f"  Top-1={original_metrics['mean_top1']:.4f} "
        f"F1={original_metrics['mean_f1']:.4f} "
        f"Log loss={original_metrics['mean_logloss']:.4f}"
    )
    print("\nOptimised Random Forest (5-fold GroupKFold):")
    print(
        f"  Top-1={optimized_metrics['mean_top1']:.4f} "
        f"F1={optimized_metrics['mean_f1']:.4f} "
        f"Log loss={optimized_metrics['mean_logloss']:.4f}"
    )
    print(f"Final full-data fit time: {final_fit_time:.1f}s")
    print(f"Saved model: {MODELS_DIR / 'random_forest_optimized.joblib'}")
    return study, metrics_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter optimisation for PathSnap Random Forest."
    )
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args(argv)
    run_optimization(trials=args.trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
