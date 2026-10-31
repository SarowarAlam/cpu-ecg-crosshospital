"""Read WFDB-style ECG records (PhysioNet/CinC Challenge 2021 repackaging):
a .hea header (ASCII) plus a .mat signal file with a single `val` array.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io

STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


@dataclass
class EcgHeader:
    record_name: str
    n_leads: int
    fs: float
    n_samples: int
    leads: list[str]
    age: str | None
    sex: str | None
    dx: list[str]


def parse_header(hea_path: Path) -> EcgHeader:
    lines = Path(hea_path).read_text().splitlines()
    head = lines[0].split()
    record_name, n_leads, fs, n_samples = head[0], int(head[1]), float(head[2]), int(head[3])

    leads = []
    for line in lines[1 : 1 + n_leads]:
        leads.append(line.split()[-1])

    age = sex = None
    dx: list[str] = []
    for line in lines[1 + n_leads :]:
        if line.startswith("#Age:") or line.startswith("# Age:"):
            age = line.split(":", 1)[1].strip()
        elif line.startswith("#Sex:") or line.startswith("# Sex:"):
            sex = line.split(":", 1)[1].strip()
        elif line.startswith("#Dx:") or line.startswith("# Dx:"):
            dx = [c.strip() for c in line.split(":", 1)[1].split(",") if c.strip()]

    return EcgHeader(record_name, n_leads, fs, n_samples, leads, age, sex, dx)


def load_signal(mat_path: Path) -> np.ndarray:
    """Returns (n_leads, n_samples) float32."""
    mat = scipy.io.loadmat(str(mat_path))
    return mat["val"].astype(np.float32)


def select_leads(signal: np.ndarray, header: EcgHeader, wanted: list[str]) -> np.ndarray:
    """Select leads by name; falls back to the standard 12-lead order if a header's
    lead names don't match exactly (seen on some non-standard records)."""
    try:
        idx = [header.leads.index(name) for name in wanted]
    except ValueError:
        idx = [STANDARD_LEADS.index(name) for name in wanted]
    return signal[idx, :]


def load_record(basepath: Path, wanted_leads: list[str]) -> tuple[np.ndarray, EcgHeader]:
    header = parse_header(basepath.with_suffix(".hea"))
    signal = load_signal(basepath.with_suffix(".mat"))
    return select_leads(signal, header, wanted_leads), header
