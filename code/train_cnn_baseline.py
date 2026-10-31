"""Lightweight-CNN baseline for cross-hospital ECG classification -- the
second half of this paper's comparison (gradient boosting is
train_gbm_baseline.py). Same five classes, same label source, same three
hospitals, same WANTED_LEADS as the GBM baseline, so the two are directly
comparable.

Architecture: cnn_model.LightX3ECG, scaled down from Le et al.'s LightX3ECG
(base_channels=16, 3 leads, 500 Hz raw signal resampled to 125 Hz) --
feasibility measured on real data during scouting, see README.md.

Compute-budget deviation from the GBM baseline's protocol (documented, not
silent): training a CNN for every one of 5 folds per source, as the GBM
baseline's 5-fold CV does, would 5x the training cost for no feasibility
margin left in the 7-8 h budget. Instead, within-dataset evaluation here
uses a single stratified 80/20 train/test split per source (same split
doubles as the early-stopping validation set, for the same reason). The
model trained on the 80% split of each source is then reused, unchanged,
for that source's row of the cross-dataset transfer table, matching the
GBM baseline's "train on one source's data, test on another's" protocol.

Run order: download_data.py headers -> build-index -> signals, then this
(reads the same data/label_index.csv and data/raw/ as the GBM baseline).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import scipy.signal
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from cnn_model import LightX3ECG
from common import CsvLogger, PAPER, RESULTS, Timer, machine_info, save_json, set_seed
from ecg_io import load_record
from labels import CLASSES

SEED = 0
WANTED_LEADS = ["I", "II", "V1"]
SOURCES = ["cpsc_2018", "chapman_shaoxing", "ptb-xl"]
ORIG_LEN = 5000             # 500 Hz, 10 s window (cropped/zero-padded)
DOWNSAMPLE_FACTOR = 4       # 500 Hz -> 125 Hz
BASE_CHANNELS = 16
BATCH_SIZE = 64
MAX_EPOCHS = 25
PATIENCE = 5
LR = 1e-3
WEIGHT_DECAY = 5e-5
TEST_FRACTION = 0.2

INDEX_PATH = PAPER / "data" / "label_index.csv"
RAW = PAPER / "data" / "raw"

torch.manual_seed(SEED)


class EcgDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_encoder: LabelEncoder):
        self.df = df.reset_index(drop=True)
        self.le = label_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        basepath = RAW / row["source"] / row["record_id"]
        signal, _ = load_record(basepath, WANTED_LEADS)  # (3, n_samples)

        n = signal.shape[1]
        if n >= ORIG_LEN:
            signal = signal[:, :ORIG_LEN]
        else:
            signal = np.pad(signal, ((0, 0), (0, ORIG_LEN - n)), mode="constant")
        signal = scipy.signal.resample_poly(signal, up=1, down=DOWNSAMPLE_FACTOR, axis=-1)

        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True)
        std[std < 1e-6] = 1.0
        signal = (signal - mean) / std
        signal = np.ascontiguousarray(signal, dtype=np.float32)

        label = int(self.le.transform([row["label"]])[0])
        return torch.from_numpy(signal), label


def run_epoch(model, loader, optimizer, criterion, train: bool):
    model.train(train)
    total_loss, all_pred, all_true = 0.0, [], []
    for x, y in loader:
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(y)
        all_pred.extend(out.argmax(dim=1).tolist())
        all_true.extend(y.tolist())
    n = len(all_true)
    acc = balanced_accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return total_loss / n, acc, f1


def train_one_source(df: pd.DataFrame, source: str, le: LabelEncoder, logger: CsvLogger, run_dir: Path):
    sub = df[df["source"] == source]
    train_df, test_df = train_test_split(
        sub, test_size=TEST_FRACTION, stratify=sub["label"], random_state=SEED,
    )
    train_loader = DataLoader(EcgDataset(train_df, le), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(EcgDataset(test_df, le), batch_size=BATCH_SIZE, shuffle=False)

    weights = compute_class_weight("balanced", classes=np.arange(len(CLASSES)),
                                    y=le.transform(train_df["label"].values))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

    model = LightX3ECG(base_channels=BASE_CHANNELS, num_classes=len(CLASSES))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_acc, best_state, epochs_no_improve = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, optimizer, criterion, train=True)
        val_loss, val_acc, val_f1 = run_epoch(model, test_loader, optimizer, criterion, train=False)
        logger.log(source=source, epoch=epoch, train_loss=train_loss, train_balanced_accuracy=train_acc,
                   val_loss=val_loss, val_balanced_accuracy=val_acc, val_macro_f1=val_f1)
        print(f"  [{source}] epoch {epoch}: train_loss={train_loss:.3f} val_loss={val_loss:.3f} "
              f"val_bal_acc={val_acc:.3f}")
        if val_acc > best_acc:
            best_acc, best_state, epochs_no_improve = val_acc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  [{source}] early stop at epoch {epoch} (best val_bal_acc={best_acc:.3f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    _, final_acc, final_f1 = run_epoch(model, test_loader, optimizer, criterion, train=False)
    return model, {"source": source, "balanced_accuracy": float(final_acc), "macro_f1": float(final_f1),
                   "n_train": len(train_df), "n_test": len(test_df)}


def eval_cross(model, df: pd.DataFrame, train_source: str, test_source: str, le: LabelEncoder, logger: CsvLogger):
    te_df = df[df["source"] == test_source]
    loader = DataLoader(EcgDataset(te_df, le), batch_size=BATCH_SIZE, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    _, acc, f1 = run_epoch(model, loader, optimizer=None, criterion=criterion, train=False)
    logger.log(eval_type="cross", train_source=train_source, test_source=test_source,
               n_test=len(te_df), balanced_accuracy=acc, macro_f1=f1)
    return {"train_source": train_source, "test_source": test_source,
            "balanced_accuracy": float(acc), "macro_f1": float(f1)}


def main():
    set_seed(SEED)
    timer = Timer()
    run_name = "cnn-baseline-01"
    run_dir = RESULTS / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INDEX_PATH, dtype=str)
    df = df[df["label"].notna()].copy()
    df = df[df.apply(lambda r: (RAW / r["source"] / f"{r['record_id']}.mat").exists(), axis=1)]
    print(f"records with downloaded signals: {len(df)}")
    print(df.groupby(["source", "label"]).size())

    le = LabelEncoder().fit(CLASSES)

    train_logger = CsvLogger(run_dir / "train_log.csv",
                              fields=["source", "epoch", "train_loss", "train_balanced_accuracy",
                                      "val_loss", "val_balanced_accuracy", "val_macro_f1"])
    cross_logger = CsvLogger(run_dir / "cross_eval_log.csv",
                              fields=["eval_type", "train_source", "test_source", "n_test",
                                      "balanced_accuracy", "macro_f1"])

    models, within_results = {}, []
    for source in SOURCES:
        if not (df["source"] == source).any():
            continue
        model, result = train_one_source(df, source, le, train_logger, run_dir)
        models[source] = model
        within_results.append(result)

    train_logger.close()

    cross_results = []
    for train_source in SOURCES:
        if train_source not in models:
            continue
        for test_source in SOURCES:
            if train_source == test_source or not (df["source"] == test_source).any():
                continue
            cross_results.append(eval_cross(models[train_source], df, train_source, test_source, le, cross_logger))
    cross_logger.close()

    save_json(run_dir / "metrics.json", {
        "within_dataset": within_results,
        "cross_dataset": cross_results,
        "n_records_total": int(len(df)),
        "n_records_by_source": df.groupby("source").size().to_dict(),
        "classes": CLASSES,
        "elapsed_s": timer.lap(),
    })
    save_json(run_dir / "config.json", {
        "wanted_leads": WANTED_LEADS, "sources": SOURCES, "seed": SEED,
        "architecture": "LightX3ECG (scaled down)", "base_channels": BASE_CHANNELS,
        "orig_len": ORIG_LEN, "downsample_factor": DOWNSAMPLE_FACTOR,
        "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "test_fraction": TEST_FRACTION,
        "within_dataset_protocol": "single stratified 80/20 split (not 5-fold CV, "
                                    "compute-budget deviation from the GBM baseline, see module docstring)",
    })
    save_json(run_dir / "machine_info.json", machine_info())

    print("within-dataset:", within_results)
    print("cross-dataset:", cross_results)
    print(f"wrote results to {run_dir}")


if __name__ == "__main__":
    main()
