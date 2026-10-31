"""Shared utilities for the ECG cross-hospital baselines: paths, seeding, CSV logging, machine info."""
from __future__ import annotations

import csv
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
RESULTS = PAPER / "results"
DATA = PAPER / "data"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


class CsvLogger:
    def __init__(self, path: Path, fields: list[str], append: bool = False):
        self.path = path
        self.fields = fields
        new = not (append and path.exists())
        self.f = open(path, "a" if append else "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        if new:
            self.w.writeheader()

    def log(self, **row):
        self.w.writerow({k: row.get(k, "") for k in self.fields})
        self.f.flush()

    def close(self):
        self.f.close()


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def machine_info() -> dict:
    info = {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count()}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import lightgbm
        info["lightgbm"] = lightgbm.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        info["torch"] = torch.__version__
    except Exception:  # noqa: BLE001
        pass
    return info


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def lap(self) -> float:
        t = time.perf_counter()
        d = t - self.t0
        self.t0 = t
        return d
