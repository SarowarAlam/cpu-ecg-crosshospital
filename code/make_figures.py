"""Figures for the manuscript, built only from results/*/metrics.json (Rule 4:
every number in a figure traces to a results file produced by a script in
code/). Writes PDF figures into ../figures/ for \\graphicspath in main.tex.

Run from papers/cpu-ecg-crosshospital/: python code/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
RESULTS = PAPER / "results"
FIGURES = PAPER / "figures"
FIGURES.mkdir(exist_ok=True)

SOURCES = ["cpsc_2018", "chapman_shaoxing", "ptb-xl"]
SOURCE_LABELS = {"cpsc_2018": "CPSC-2018", "chapman_shaoxing": "Chapman-Shaoxing", "ptb-xl": "PTB-XL"}

# Validated categorical palette (dataviz skill, references/palette.md), slots 1 and 2.
BLUE = "#2a78d6"   # gradient boosting
ORANGE = "#eb6834"  # lightweight CNN
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


def load_metrics(run_name: str) -> dict:
    return json.loads((RESULTS / run_name / "metrics.json").read_text())


def within_by_source(metrics: dict) -> dict:
    # the GBM baseline logs the 5-fold mean as "balanced_accuracy_mean";
    # the CNN baseline logs its single-split score as "balanced_accuracy".
    return {r["source"]: r.get("balanced_accuracy", r.get("balanced_accuracy_mean"))
            for r in metrics["within_dataset"]}


def cross_matrix(metrics: dict) -> np.ndarray:
    m = np.full((3, 3), np.nan)
    for r in metrics["cross_dataset"]:
        i = SOURCES.index(r["train_source"])
        j = SOURCES.index(r["test_source"])
        m[i, j] = r["balanced_accuracy"]
    return m


def fig_within_dataset(gbm: dict, cnn: dict):
    gbm_w = within_by_source(gbm)
    cnn_w = within_by_source(cnn)

    x = np.arange(len(SOURCES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 3.6))

    bars_gbm = ax.bar(x - width / 2, [gbm_w[s] for s in SOURCES], width,
                       label="Gradient boosting", color=BLUE)
    bars_cnn = ax.bar(x + width / 2, [cnn_w[s] for s in SOURCES], width,
                       label="Lightweight CNN", color=ORANGE)

    for bars in (bars_gbm, bars_cnn):
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8, color=INK, xytext=(0, 2),
                        textcoords="offset points")

    ax.set_xticks(x)
    ax.set_xticklabels([SOURCE_LABELS[s] for s in SOURCES])
    ax.set_ylabel("Within-dataset balanced accuracy")
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.15), ncol=2)

    fig.tight_layout()
    fig.savefig(FIGURES / "within_dataset_accuracy.pdf")
    plt.close(fig)


def fig_cross_dataset(gbm: dict, cnn: dict):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), constrained_layout=True)
    labels = [SOURCE_LABELS[s] for s in SOURCES]
    mats = {"Gradient boosting": cross_matrix(gbm), "Lightweight CNN": cross_matrix(cnn)}

    im = None
    for ax, (title, mat) in zip(axes, mats.items()):
        im = ax.imshow(mat, cmap="Blues", vmin=0.4, vmax=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_yticks(range(3))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Test source")
        ax.set_ylabel("Train source")
        for i in range(3):
            for j in range(3):
                if np.isnan(mat[i, j]):
                    ax.text(j, i, "--", ha="center", va="center", color=MUTED, fontsize=9)
                else:
                    color = "white" if mat[i, j] > 0.75 else INK
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                            color=color, fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.colorbar(im, ax=axes, label="Balanced accuracy", shrink=0.85, pad=0.02)
    fig.savefig(FIGURES / "cross_dataset_accuracy.pdf")
    plt.close(fig)


def main():
    gbm = load_metrics("gbm-baseline-01")
    cnn = load_metrics("cnn-baseline-01")
    fig_within_dataset(gbm, cnn)
    fig_cross_dataset(gbm, cnn)
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
