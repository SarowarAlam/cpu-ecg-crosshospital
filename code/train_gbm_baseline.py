"""Gradient-boosting baseline for cross-hospital ECG classification.

Builds a handcrafted-feature matrix (features.py) from the three sources
(PTB-XL, CPSC-2018, Chapman-Shaoxing), using the shared label scheme
(labels.py) from the PhysioNet/CinC Challenge 2021 SNOMED-CT repackaging.
Trains a LightGBM classifier per source and evaluates both within-dataset
(stratified CV) and cross-dataset (train on one source, test on the other
two) -- the latter is the cross-hospital generalization test this paper's
title asks about.

Run order: download_data.py headers -> build-index -> signals, then this.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from common import CsvLogger, PAPER, RESULTS, Timer, machine_info, save_json, set_seed
from ecg_io import load_record
from features import extract_features
from labels import CLASSES

SEED = 0
WANTED_LEADS = ["I", "II", "V1"]
N_FOLDS = 5
SOURCES = ["cpsc_2018", "chapman_shaoxing", "ptb-xl"]

INDEX_PATH = PAPER / "data" / "label_index.csv"
RAW = PAPER / "data" / "raw"


def build_feature_matrix(run_dir: Path) -> pd.DataFrame:
    """Extracts features for every downloaded, labelled record. Cached to
    run_dir/features.parquet since extraction is the slow step."""
    cache = run_dir / "features.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    df = pd.read_csv(INDEX_PATH, dtype=str)
    df = df[df["label"].notna()].copy()

    rows = []
    n_missing = n_errors = 0
    for _, row in df.iterrows():
        mat_path = RAW / row["source"] / f"{row['record_id']}.mat"
        if not mat_path.exists():
            n_missing += 1
            continue
        try:
            signal, header = load_record(mat_path.with_suffix(""), WANTED_LEADS)
            feats = extract_features(signal, header.fs, WANTED_LEADS)
        except Exception as e:  # noqa: BLE001
            n_errors += 1
            print(f"  feature-extraction error on {row['source']}/{row['record_id']}: {e}")
            continue
        feats["source"] = row["source"]
        feats["label"] = row["label"]
        feats["record_id"] = row["record_id"]
        rows.append(feats)

    print(f"feature extraction: {len(rows)} ok, {n_missing} missing .mat, {n_errors} errors")
    out = pd.DataFrame(rows)
    run_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("source", "label", "record_id")]


def within_dataset_cv(df: pd.DataFrame, source: str, logger: CsvLogger) -> dict:
    sub = df[df["source"] == source]
    X = sub[feature_cols(sub)].fillna(0.0).values
    le = LabelEncoder().fit(CLASSES)
    y = le.transform(sub["label"].values)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    accs, f1s = [], []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        clf = lgb.LGBMClassifier(n_estimators=200, random_state=SEED, verbosity=-1)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        acc = balanced_accuracy_score(y[te], pred)
        f1 = f1_score(y[te], pred, average="macro")
        accs.append(acc)
        f1s.append(f1)
        logger.log(eval_type="within", train_source=source, test_source=source, fold=fold,
                    n_train=len(tr), n_test=len(te), balanced_accuracy=acc, macro_f1=f1)
    return {"source": source, "balanced_accuracy_mean": float(np.mean(accs)),
            "balanced_accuracy_std": float(np.std(accs)), "macro_f1_mean": float(np.mean(f1s))}


def cross_dataset_eval(df: pd.DataFrame, train_source: str, test_source: str, logger: CsvLogger) -> dict:
    tr_df = df[df["source"] == train_source]
    te_df = df[df["source"] == test_source]
    le = LabelEncoder().fit(CLASSES)

    X_tr = tr_df[feature_cols(tr_df)].fillna(0.0).values
    y_tr = le.transform(tr_df["label"].values)
    X_te = te_df[feature_cols(te_df)].fillna(0.0).values
    y_te = le.transform(te_df["label"].values)

    clf = lgb.LGBMClassifier(n_estimators=200, random_state=SEED, verbosity=-1)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    acc = balanced_accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro")
    logger.log(eval_type="cross", train_source=train_source, test_source=test_source, fold=-1,
                n_train=len(y_tr), n_test=len(y_te), balanced_accuracy=acc, macro_f1=f1)
    return {"train_source": train_source, "test_source": test_source,
            "balanced_accuracy": float(acc), "macro_f1": float(f1)}


def main():
    set_seed(SEED)
    timer = Timer()
    run_name = "gbm-baseline-01"
    run_dir = RESULTS / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    df = build_feature_matrix(run_dir)
    print(f"feature matrix: {df.shape}, by source/label:")
    print(df.groupby(["source", "label"]).size())

    logger = CsvLogger(run_dir / "eval_log.csv",
                        fields=["eval_type", "train_source", "test_source", "fold",
                                "n_train", "n_test", "balanced_accuracy", "macro_f1"])

    within_results = [within_dataset_cv(df, s, logger) for s in SOURCES if (df["source"] == s).any()]

    cross_results = []
    for train_source in SOURCES:
        for test_source in SOURCES:
            if train_source == test_source:
                continue
            if not (df["source"] == train_source).any() or not (df["source"] == test_source).any():
                continue
            cross_results.append(cross_dataset_eval(df, train_source, test_source, logger))

    logger.close()

    save_json(run_dir / "metrics.json", {
        "within_dataset": within_results,
        "cross_dataset": cross_results,
        "n_records_total": int(len(df)),
        "n_records_by_source": df.groupby("source").size().to_dict(),
        "classes": CLASSES,
        "elapsed_s": timer.lap(),
    })
    save_json(run_dir / "config.json", {
        "wanted_leads": WANTED_LEADS, "n_folds": N_FOLDS, "seed": SEED,
        "classifier": "LGBMClassifier(n_estimators=200)",
        "sources": SOURCES,
    })
    save_json(run_dir / "machine_info.json", machine_info())

    print("within-dataset:", within_results)
    print("cross-dataset:", cross_results)
    print(f"wrote results to {run_dir}")


if __name__ == "__main__":
    main()
