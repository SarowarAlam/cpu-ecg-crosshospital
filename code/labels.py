"""Shared label scheme across PTB-XL, CPSC-2018, and Chapman-Shaoxing.

Labels come from the PhysioNet/CinC Challenge 2021 repackaging of these three
datasets (Reyna et al. 2021, dataset DOI 10.13026/34va-7q14), which gives all
three a common SNOMED-CT `#Dx` coding scheme. The five diagnoses below were
chosen from the challenge's own scored-label table
(physionetchallenges/evaluation-2021/dx_mapping_scored.csv, verified
2026-10-30) as the ones with meaningfully large counts in all three source
databases' *own* columns (not CPSC_Extra, not Ningbo):

    code        | abbrev | CPSC | PTB_XL | Chapman_Shaoxing
    164889003   | AF     | 1221 | 1514   | 1780
    426783006   | NSR    | 918  | 18092  | 1826
    270492004   | IAVB   | 722  | 797    | 247
    164909002   | LBBB   | 236  | 536    | 205
    284470004   | PAC    | 616  | 398    | 258

Deliberately excluded for this first baseline: PVC (427172004), because the
official challenge data maps CPSC-2018's PVC label to the unscored code
164884008 (VEB) instead, an unsettled mapping choice flagged on the
PhysioNet challenge forum (see papers/cpu-ecg-crosshospital/README.md,
open item 2). Revisit once that choice is made explicitly.

A record is labelled only if exactly one of the five target codes appears in
its (possibly multi-label) #Dx field; records with zero or more than one are
dropped from this baseline as ambiguous, not forced into a class.
"""
from __future__ import annotations

TARGET_CODES: dict[str, str] = {
    "164889003": "AF",
    "426783006": "NSR",
    "270492004": "IAVB",
    "164909002": "LBBB",
    "284470004": "PAC",
}

CLASSES = sorted(TARGET_CODES.values())


def assign_label(dx_codes: list[str]) -> str | None:
    hits = {TARGET_CODES[c] for c in dx_codes if c in TARGET_CODES}
    if len(hits) == 1:
        return next(iter(hits))
    return None
