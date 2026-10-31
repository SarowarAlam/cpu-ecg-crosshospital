"""Download PTB-XL, CPSC-2018, and Chapman-Shaoxing from the PhysioNet/CinC
Challenge 2021 repackaging (Reyna et al. 2021, dataset DOI 10.13026/34va-7q14),
which gives all three a shared SNOMED-CT `#Dx` label scheme (verified
2026-10-30, see ../README.md). No bulk zip is offered for these nested
subfolders (checked: get-zip 404s on them), and PhysioNet appears to cap
concurrent connections per client -- plain per-file GET over several
thousand records is still the only option, so this runs in two phases to
keep it within a reasonable time:

  Phase 1 (headers): download every record's tiny .hea file only, across
  all ~39,000 records in the three sources. Builds a full per-record label
  index without paying for ~39,000 much-larger .mat signal downloads.

  Phase 2 (signals): download .mat signal files only for the records whose
  label (per labels.assign_label) is one of our five target classes,
  capped per class per dataset so the final corpus stays balanced and the
  download stays bounded.

Usage:
  python download_data.py headers [--only cpsc_2018,chapman_shaoxing,ptb-xl]
  python download_data.py build-index
  python download_data.py signals [--cap-per-class N]
"""
from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ecg_io import parse_header
from labels import assign_label

_thread_local = threading.local()


def _session() -> requests.Session:
    """One requests.Session per worker thread -- Session is not safe to share across threads."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


PAPER = Path(__file__).resolve().parents[1]
RAW = PAPER / "data" / "raw"
INDEX_PATH = PAPER / "data" / "label_index.csv"

BASE = "https://physionet.org/files/challenge-2021/1.0.3/training"
SOURCES = {
    "cpsc_2018": "cpsc_2018",
    "chapman_shaoxing": "chapman_shaoxing",
    "ptb-xl": "ptb-xl",
}
WORKERS = 8  # PhysioNet's server appears to throttle beyond ~8 concurrent connections

HREF_RE = re.compile(r'href="([^"]+)"')


def list_hrefs(url: str) -> list[str]:
    r = _session().get(url, timeout=30)
    r.raise_for_status()
    return HREF_RE.findall(r.text)


def list_record_ids(source_url: str) -> list[tuple[str, str]]:
    """Returns (group_name, record_id) pairs across all gN subfolders of a source."""
    groups = [h for h in list_hrefs(source_url) if h not in ("../",) and h.endswith("/")]
    out = []
    for g in groups:
        hrefs = list_hrefs(source_url + g)
        ids = sorted({h[:-4] for h in hrefs if h.endswith(".hea")})
        out.extend((g, rid) for rid in ids)
    return out


def download_one(url: str, dest: Path, retries: int = 2) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    for attempt in range(retries + 1):
        try:
            r = _session().get(url, timeout=30)
            if r.status_code != 200:
                continue
            dest.write_bytes(r.content)
            return True
        except Exception:  # noqa: BLE001
            if attempt == retries:
                return False
    return False


def run_jobs(name: str, jobs: list[tuple[str, Path]], workers: int = WORKERS) -> tuple[int, int]:
    t0 = time.perf_counter()
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download_one, url, dest): dest for url, dest in jobs}
        for i, fut in enumerate(as_completed(futs)):
            if fut.result():
                ok += 1
            else:
                fail += 1
            if (i + 1) % 1000 == 0:
                print(f"[{name}] {i+1}/{len(jobs)} files ({ok} ok, {fail} failed), "
                      f"{time.perf_counter()-t0:.0f}s elapsed", flush=True)
    print(f"[{name}] done: {ok} ok, {fail} failed, {time.perf_counter()-t0:.0f}s", flush=True)
    return ok, fail


def phase_headers(sources: dict[str, str], limit: int | None = None) -> None:
    for name, folder in sources.items():
        source_url = f"{BASE}/{folder}/"
        out_dir = RAW / name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{name}] listing records...", flush=True)
        records = list_record_ids(source_url)
        print(f"[{name}] {len(records)} records found", flush=True)
        if limit is not None:
            records = records[:limit]

        jobs = [(f"{source_url}{g}{rid}.hea", out_dir / f"{rid}.hea") for g, rid in records]
        run_jobs(f"{name}:headers", jobs)


def phase_build_index(sources: dict[str, str]) -> pd.DataFrame:
    rows = []
    for name in sources:
        out_dir = RAW / name
        for hea_path in sorted(out_dir.glob("*.hea")):
            try:
                header = parse_header(hea_path)
            except Exception as e:  # noqa: BLE001
                print(f"  parse error {hea_path.name}: {e}")
                continue
            label = assign_label(header.dx)
            rows.append({
                "source": name, "record_id": header.record_name, "group": _group_of(hea_path),
                "fs": header.fs, "n_samples": header.n_samples, "n_leads": header.n_leads,
                "age": header.age, "sex": header.sex, "dx": ",".join(header.dx), "label": label,
            })
    df = pd.DataFrame(rows)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INDEX_PATH, index=False)
    print(f"index written: {INDEX_PATH} ({len(df)} records)")
    print(df.groupby(["source", "label"]).size())
    return df


def _group_of(hea_path: Path) -> str:
    # headers are flattened into data/raw/<source>/ without the gN subfolder, so
    # we don't know the original group here; not needed for signal re-download
    # since we reconstruct the URL by searching all groups lazily (see phase_signals).
    return ""


def phase_signals(sources: dict[str, str], cap_per_class: int) -> None:
    if not INDEX_PATH.exists():
        raise SystemExit("run `build-index` first")
    df = pd.read_csv(INDEX_PATH, dtype=str)
    df = df[df["label"].notna()]

    parts = []
    for (src, label), g in df.groupby(["source", "label"]):
        parts.append(g.sample(n=min(len(g), cap_per_class), random_state=0))
    selected = pd.concat(parts, ignore_index=True)
    print(selected.groupby(["source", "label"]).size())

    for name, folder in sources.items():
        sub = selected[selected["source"] == name]
        if sub.empty:
            continue
        source_url = f"{BASE}/{folder}/"
        out_dir = RAW / name
        print(f"[{name}] resolving group folders for {len(sub)} selected records...", flush=True)
        groups = [h for h in list_hrefs(source_url) if h.endswith("/") and h != "../"]
        id_to_group: dict[str, str] = {}
        wanted_ids = set(sub["record_id"])
        for g in groups:
            hrefs = list_hrefs(source_url + g)
            for h in hrefs:
                if h.endswith(".mat"):
                    rid = h[:-4]
                    if rid in wanted_ids:
                        id_to_group[rid] = g
            if len(id_to_group) == len(wanted_ids):
                break

        jobs = [(f"{source_url}{id_to_group[rid]}{rid}.mat", out_dir / f"{rid}.mat")
                for rid in sub["record_id"] if rid in id_to_group]
        missing = len(sub) - len(jobs)
        if missing:
            print(f"[{name}] warning: {missing} selected records not found in group listing")
        run_jobs(f"{name}:signals", jobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["headers", "build-index", "signals"])
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--limit-per-source", type=int, default=None)
    ap.add_argument("--cap-per-class", type=int, default=800)
    args = ap.parse_args()

    sources = SOURCES
    if args.only:
        wanted = set(args.only.split(","))
        sources = {k: v for k, v in SOURCES.items() if k in wanted}

    if args.phase == "headers":
        phase_headers(sources, args.limit_per_source)
    elif args.phase == "build-index":
        phase_build_index(sources)
    elif args.phase == "signals":
        phase_signals(sources, args.cap_per_class)


if __name__ == "__main__":
    main()
