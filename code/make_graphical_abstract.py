"""Graphical abstract for JBHI submission, built only from results/*/metrics.json
(Rule 4: every number traces to a results file produced by a script in code/).
Writes a PNG into ../figures/ for upload to the submission portal.

Run from papers/cpu-ecg-crosshospital/: python code/make_graphical_abstract.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

PAPER = Path(__file__).resolve().parents[1]
RESULTS = PAPER / "results"
FIGURES = PAPER / "figures"
FIGURES.mkdir(exist_ok=True)

SOURCES = ["cpsc_2018", "chapman_shaoxing", "ptb-xl"]
SOURCE_LABELS = {"cpsc_2018": "CPSC-2018", "chapman_shaoxing": "Chapman-Shaoxing", "ptb-xl": "PTB-XL"}

# Validated categorical palette (dataviz skill), same slots as make_figures.py.
BLUE = "#2a78d6"    # gradient boosting
ORANGE = "#eb6834"  # lightweight CNN
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "text.color": INK,
})


def load_metrics(run_name: str) -> dict:
    return json.loads((RESULTS / run_name / "metrics.json").read_text())


def within_by_source(metrics: dict) -> dict:
    return {r["source"]: r.get("balanced_accuracy", r.get("balanced_accuracy_mean"))
            for r in metrics["within_dataset"]}


def cross_by_test(metrics: dict) -> dict:
    # mean cross-dataset accuracy when a given source is the held-out test set
    by_test = {s: [] for s in SOURCES}
    for r in metrics["cross_dataset"]:
        by_test[r["test_source"]].append(r["balanced_accuracy"])
    return {s: float(np.mean(v)) for s, v in by_test.items()}


def main():
    gbm = load_metrics("gbm-baseline-01")
    cnn = load_metrics("cnn-baseline-01")

    gbm_w, cnn_w = within_by_source(gbm), within_by_source(cnn)
    gbm_c, cnn_c = cross_by_test(gbm), cross_by_test(cnn)

    mean_within = {"Gradient boosting": float(np.mean(list(gbm_w.values()))),
                   "Lightweight CNN": float(np.mean(list(cnn_w.values())))}
    mean_cross = {"Gradient boosting": float(np.mean(list(gbm_c.values()))),
                  "Lightweight CNN": float(np.mean(list(cnn_c.values())))}

    fig = plt.figure(figsize=(12, 6.0), dpi=200, facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.32,
                           left=0.04, right=0.98, top=0.74, bottom=0.1)

    # --- Left panel: schematic (3 hospitals -> 2 models -> unseen hospital) ---
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_xlim(0, 10)
    ax0.set_ylim(0, 10)
    ax0.axis("off")
    ax0.set_facecolor(SURFACE)

    hosp_y = [8.3, 5.5, 2.7]
    for y, s in zip(hosp_y, SOURCES):
        ax0.add_patch(plt.Rectangle((0.2, y - 0.6), 2.4, 1.2, facecolor="white",
                                     edgecolor=MUTED, linewidth=1.2))
        ax0.text(1.4, y, SOURCE_LABELS[s], ha="center", va="center", fontsize=9.5, color=INK)
    ax0.text(1.4, 9.6, "Public 12-lead ECG sources", ha="center", va="center",
              fontsize=9.5, color=MUTED, style="italic")

    model_boxes = [("Gradient\nboosting", 6.9, BLUE), ("Lightweight\nCNN", 3.9, ORANGE)]
    for label, y, color in model_boxes:
        ax0.add_patch(plt.Rectangle((4.1, y - 0.75), 2.2, 1.5, facecolor=color,
                                     edgecolor="none", alpha=0.9))
        ax0.text(5.2, y, label, ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")

    for y0 in hosp_y:
        for _, y1, color in model_boxes:
            ax0.add_patch(FancyArrowPatch((2.6, y0), (4.1, y1), arrowstyle="-|>",
                                           mutation_scale=8, color=MUTED, linewidth=0.8, alpha=0.6))

    ax0.add_patch(plt.Rectangle((7.3, 4.1), 2.3, 1.8, facecolor="white",
                                 edgecolor=INK, linewidth=1.4, linestyle=(0, (4, 2))))
    ax0.text(8.45, 5.35, "Hospital not\nseen in training", ha="center", va="center",
              fontsize=9, color=INK)
    ax0.text(8.45, 4.55, "accuracy drop?", ha="center", va="center",
              fontsize=9, color=MUTED, style="italic")
    for _, y1, color in model_boxes:
        ax0.add_patch(FancyArrowPatch((6.3, y1), (7.3, 5.0), arrowstyle="-|>",
                                       mutation_scale=10, color=color, linewidth=1.6))

    # --- Right panel: mean within- vs cross-dataset accuracy ---
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor(SURFACE)
    models = ["Gradient boosting", "Lightweight CNN"]
    colors = [BLUE, ORANGE]
    x = np.arange(len(models))
    width = 0.32

    bars_w = ax1.bar(x - width / 2, [mean_within[m] for m in models], width,
                      label="Within-dataset", color=colors, alpha=1.0)
    bars_c = ax1.bar(x + width / 2, [mean_cross[m] for m in models], width,
                      label="Cross-dataset", color=colors, alpha=0.4)

    for bars in (bars_w, bars_c):
        for b in bars:
            ax1.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                         ha="center", va="bottom", fontsize=9.5, color=INK, xytext=(0, 3),
                         textcoords="offset points")

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.set_ylabel("Mean balanced accuracy", fontsize=10)
    ax1.set_ylim(0, 1.0)
    ax1.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax1.set_axisbelow(True)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)
    ax1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, fontsize=9.5)

    fig.suptitle("Do compact ECG classifiers transfer across hospitals?",
                 fontsize=14.5, fontweight="bold", y=0.97, color=INK)
    fig.text(0.5, 0.85, "Both the gradient-boosting and lightweight-CNN baselines lose accuracy"
              " on an unseen hospital's data", ha="center", fontsize=11, color=MUTED, style="italic")

    fig.savefig(FIGURES / "graphical_abstract.png", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {FIGURES / 'graphical_abstract.png'}")


if __name__ == "__main__":
    main()
