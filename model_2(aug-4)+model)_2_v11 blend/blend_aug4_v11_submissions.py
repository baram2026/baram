"""Reproduce the leaderboard-tested Aug-4 + Model-2 v11 groupwise blend."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


TIME_COL = "forecast_kst_dtm"
KEYS = ["forecast_id", TIME_COL]
TARGETS = {1: "kpx_group_1", 2: "kpx_group_2", 3: "kpx_group_3"}
CAPACITIES = {1: 21600.0, 2: 21600.0, 3: 21000.0}
AUG4_WEIGHTS = {1: 0.35, 2: 0.80, 3: 1.00}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    missing = set(KEYS + list(TARGETS.values())) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame[KEYS].duplicated().any():
        raise ValueError(f"{path}: duplicated forecast_id/timestamp rows")
    frame[TIME_COL] = pd.to_datetime(frame[TIME_COL], errors="raise")
    return frame


def blend(aug4: pd.DataFrame, v11: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    if len(aug4) != len(v11):
        raise ValueError(f"Row-count mismatch: Aug-4={len(aug4)}, v11={len(v11)}")
    if not aug4[KEYS].astype(str).equals(v11[KEYS].astype(str)):
        raise ValueError("Aug-4 and v11 rows are not identically ordered")

    result = aug4.copy()
    diagnostics = []
    for group, target in TARGETS.items():
        aug4_weight = AUG4_WEIGHTS[group]
        v11_weight = 1.0 - aug4_weight
        raw = aug4_weight * aug4[target].astype(float) + v11_weight * v11[target].astype(float)
        result[target] = raw.clip(0.0, CAPACITIES[group] * 1.05)
        change = result[target] - aug4[target].astype(float)
        diagnostics.append({
            "group": group,
            "aug4_weight": aug4_weight,
            "v11_weight": v11_weight,
            "changed_rows": int(change.abs().gt(1e-9).sum()),
            "mean_change_kwh": float(change.mean()),
            "mean_abs_change_kwh": float(change.abs().mean()),
            "max_abs_change_kwh": float(change.abs().max()),
        })
    return result, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug4", type=Path, required=True, help="Original Aug-4 submission CSV")
    parser.add_argument("--v11", type=Path, required=True, help="Final Model-2 v11 submission CSV")
    parser.add_argument("--output", type=Path, default=Path("submission_aug4_v11_blend.csv"))
    args = parser.parse_args()

    aug4 = load_submission(args.aug4)
    v11 = load_submission(args.v11)
    result, diagnostics = blend(aug4, v11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")

    manifest = {
        "formula": {
            "group1": "0.35 * Aug4 + 0.65 * v11",
            "group2": "0.80 * Aug4 + 0.20 * v11",
            "group3": "1.00 * Aug4 + 0.00 * v11",
        },
        "inputs": {
            "aug4": str(args.aug4), "aug4_sha256": sha256(args.aug4),
            "v11": str(args.v11), "v11_sha256": sha256(args.v11),
        },
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "rows": len(result),
        "diagnostics": diagnostics,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(diagnostics).to_string(index=False))
    print("saved:", args.output)
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
