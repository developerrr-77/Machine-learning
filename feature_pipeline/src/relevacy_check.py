
"""
Fetch data from a Hopsworks Feature Group and assess its suitability for DeepAR model training.

Usage:
  - Set env vars HOPSWORKS_API_KEY and HOPSWORKS_PROJECT (or pass --project).
  - python hopsworks_deepar_relevance_check.py --fg-name demand_records --fg-version 1
Optional overrides:
  --item-col ITEM --time-col TIME --target-col TARGET
"""
from __future__ import annotations
import os
import sys
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd

# Hopsworks / HSFS
try:
    import hopsworks  # type: ignore
except Exception as e:
    hopsworks = None


@dataclass
class ColumnGuess:
    item_col: Optional[str]
    time_col: Optional[str]
    target_col: Optional[str]
    notes: List[str]


DEEPar_TARGET_CANDIDATES = [
    "target","y","value","values","label","sales","units","units_sold","qty","quantity","demand","count","volume"
]
ITEM_ID_CANDIDATES = [
    "item_id","item","id","series_id","product_id","product","sku","sku_id","pm_id","seller_sku","asin","upc","gtin"
]
TIME_CANDIDATES = [
    "timestamp","ts","time","datetime","date","ds","event_time","order_date","sale_date"
]


def guess_columns(df: pd.DataFrame) -> ColumnGuess:
    cols = [c.lower() for c in df.columns]
    notes: List[str] = []

    def find_first(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in cols:
                return df.columns[cols.index(c)]
        return None

    item = find_first(ITEM_ID_CANDIDATES)
    # time: name first, else dtype
    time = find_first(TIME_CANDIDATES)
    if time is None:
        for c, dtype in df.dtypes.items():
            if np.issubdtype(dtype, np.datetime64):
                time = c
                notes.append(f"Detected datetime dtype for time column: {c}")
                break
    target = find_first(DEEPar_TARGET_CANDIDATES)

    # Try light heuristics if still missing
    if target is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            target = numeric_cols[-1]
            notes.append(f"Fallback target as last numeric column: {target}")

    return ColumnGuess(item, time, target, notes)


@dataclass
class CheckResult:
    passed: bool
    details: Dict[str, str]


def infer_frequency(g: pd.DataFrame, time_col: str) -> Optional[pd.Timedelta]:
    t = pd.to_datetime(g[time_col].sort_values())
    if t.size < 3:
        return None
    diffs = t.diff().dropna()
    if diffs.empty:
        return None
    # Use median to reduce outlier effect
    return diffs.median()


def evaluate_deepar_readiness(df: pd.DataFrame, item_col: str, time_col: str, target_col: str) -> Tuple[str, Dict[str, object]]:
    report: Dict[str, object] = {}

    # Basic cleaning
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=[time_col])
    dropped_time_na = n_before - len(df)

    # Basic stats
    report["rows"] = len(df)
    report["columns"] = list(df.columns)
    report["dropped_time_na"] = int(dropped_time_na)

    # Column validity
    issues: List[str] = []
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        try:
            df[target_col] = pd.to_numeric(df[target_col])
            report["target_casted_to_numeric"] = True
        except Exception:
            issues.append(f"Target column '{target_col}' is not numeric and could not be cast.")

    # Negative values
    negatives = (df[target_col] < 0).sum()
    report["negative_target_values"] = int(negatives)

    # Missing
    miss_rate = df[target_col].isna().mean()
    report["target_missing_rate"] = float(miss_rate)

    # Items
    n_items = df[item_col].nunique(dropna=True)
    report["n_items"] = int(n_items)

    # Length per item
    lengths = df.groupby(item_col)[target_col].size()
    report["series_length_min"] = int(lengths.min()) if not lengths.empty else 0
    report["series_length_p25"] = int(lengths.quantile(0.25)) if not lengths.empty else 0
    report["series_length_median"] = int(lengths.median()) if not lengths.empty else 0

    # Frequency and regularity
    freqs = (
        df.sort_values([item_col, time_col])
          .groupby(item_col)
          .apply(lambda g: infer_frequency(g, time_col))
    )
    freq_mode = None
    if isinstance(freqs, pd.Series) and len(freqs) > 0:
        # Convert to seconds for mode
        secs = freqs.dropna().apply(lambda x: x / np.timedelta64(1, 's'))
        if not secs.empty:
            freq_mode_seconds = secs.round().mode().iloc[0]
            freq_mode = pd.to_timedelta(freq_mode_seconds, unit='s')
    report["inferred_frequency_mode"] = str(freq_mode) if freq_mode is not None else None

    # Regularity score: share of diffs equal to per-item median
    def regularity_score(g: pd.DataFrame) -> float:
        t = pd.to_datetime(g[time_col].sort_values())
        d = t.diff().dropna()
        if d.empty:
            return 1.0
        med = d.median()
        return float((np.abs(d - med) < pd.Timedelta(seconds=1)).mean())

    reg_scores = df.groupby(item_col).apply(regularity_score)
    report["regularity_score_median"] = float(reg_scores.median()) if not reg_scores.empty else 1.0

    # Coverage of shared calendar if frequency inferred
    coverage = None
    if freq_mode is not None and pd.notna(freq_mode):
        # build per-item expected index and compare
        gaps = []
        for k, g in df.groupby(item_col):
            g = g.sort_values(time_col)
            if len(g) < 2:
                continue
            rng = pd.date_range(start=g[time_col].iloc[0], end=g[time_col].iloc[-1], freq=freq_mode)
            present = pd.Index(g[time_col])
            miss = (~rng.isin(present)).mean() if len(rng) > 0 else 0.0
            gaps.append(miss)
        if gaps:
            coverage = 1.0 - float(np.median(gaps))
    report["calendar_coverage_median"] = coverage

    # Outliers via IQR per item
    def outlier_rate(g: pd.DataFrame) -> float:
        x = g[target_col].astype(float)
        q1, q3 = np.nanpercentile(x, 25), np.nanpercentile(x, 75)
        iqr = q3 - q1
        if iqr == 0:
            return 0.0
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        return float(((x < lower) | (x > upper)).mean())

    out_rates = df.groupby(item_col).apply(outlier_rate)
    report["outlier_rate_median"] = float(out_rates.median()) if not out_rates.empty else 0.0

    # Verdict logic
    # Heuristics for DeepAR readiness
    #  - Have >= 50 items or >= 1 long series
    #  - Median series length >= 24
    #  - Target numeric with low missing (<5%) and few negatives
    #  - Reasonable regularity (>=0.8) or known frequency
    readiness_flags = []
    if n_items >= 50 or report["series_length_median"] >= 100:
        readiness_flags.append("enough_data")
    if report["series_length_median"] >= 24:
        readiness_flags.append("enough_history")
    if miss_rate <= 0.05:
        readiness_flags.append("low_missing")
    if negatives == 0:
        readiness_flags.append("no_negatives")
    if report["regularity_score_median"] >= 0.8 or report["inferred_frequency_mode"] is not None:
        readiness_flags.append("regular_frequency")

    score = len(readiness_flags) / 5.0
    report["readiness_flags"] = readiness_flags
    report["readiness_score_0_1"] = round(float(score), 2)
    report["verdict"] = (
        "Ready for DeepAR" if score >= 0.6 else
        "Probably OK with preprocessing" if score >= 0.4 else
        "Not ready"
    )

    # Key recommendations
    recs: List[str] = []
    if miss_rate > 0:
        recs.append("Impute or drop missing target values per series.")
    if negatives > 0:
        recs.append("Clip or investigate negative target values.")
    if report["regularity_score_median"] < 0.8:
        recs.append("Resample to a fixed frequency per item and fill gaps with zero or NA.")
    if report["series_length_median"] < 24:
        recs.append("Collect more history per series. DeepAR benefits from longer sequences.")
    if n_items < 10:
        recs.append("Add more items or switch to a single-series model.")
    report["recommendations"] = recs

    # Preview of inferred columns
    report["columns_used"] = {
        "item_col": item_col,
        "time_col": time_col,
        "target_col": target_col,
    }

    # Small sample preview
    report["sample_head"] = df.sort_values([item_col, time_col]).head(10).to_dict(orient="list")

    summary = (
        f"Rows={report['rows']}, Items={n_items}, MedianLen={report['series_length_median']}, "
        f"NegTarget={report['negative_target_values']}, MissRate={miss_rate:.3f}, "
        f"RegScoreMed={report['regularity_score_median']:.2f}, Freq={report['inferred_frequency_mode']}, "
        f"Score={report['readiness_score_0_1']}, Verdict={report['verdict']}"
    )

    return summary, report


def fetch_feature_group(name: str, version: int, project: Optional[str]) -> pd.DataFrame:
    if hopsworks is None:
        raise RuntimeError("hopsworks package not installed. pip install hopsworks")

    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if api_key is None:
        raise RuntimeError("Set HOPSWORKS_API_KEY in environment.")

    project_name = project or os.environ.get("HOPSWORKS_PROJECT_NAME")
    project_obj = hopsworks.login(project=project_name, api_key_value=api_key)
    fs = project_obj.get_feature_store()

    fg = fs.get_feature_group(name=name, version=version)

    # Try to get pandas DataFrame regardless of engine
    try:
        df = fg.read(read_options={"dataframe_type": "pandas"})
    except Exception:
        df = fg.read()
        try:
            df = df.to_pandas()
        except Exception:
            # As a last resort, convert via Arrow
            df = pd.DataFrame(df.collect())
    # Normalize columns by stripping spaces
    df.columns = [c.strip() for c in df.columns]
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fg-name", default="demand_records")
    p.add_argument("--fg-version", type=int, default=1)
    p.add_argument("--project", default=None)
    p.add_argument("--item-col", default=None)
    p.add_argument("--time-col", default=None)
    p.add_argument("--target-col", default=None)
    p.add_argument("--out-json", default="deepar_readiness_report.json")
    args = p.parse_args()

    df = fetch_feature_group(args.fg_name, args.fg_version, args.project)

    guess = guess_columns(df)
    item_col = args.item_col or guess.item_col
    time_col = args.time_col or guess.time_col
    target_col = args.target_col or guess.target_col

    if not all([item_col, time_col, target_col]):
        missing = [k for k,v in {
            'item_col': item_col, 'time_col': time_col, 'target_col': target_col
        }.items() if v is None]
        raise SystemExit(f"Could not infer required columns: {', '.join(missing)}. Use overrides.")

    summary, report = evaluate_deepar_readiness(df, item_col, time_col, target_col)

    # Persist report
    import json
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("Inference notes:")
    for n in guess.notes:
        print(" -", n)
    print("\nColumns used:", report["columns_used"]) 
    print("\nSummary:")
    print(summary)
    print(f"\nDetailed report saved to {args.out_json}")


if __name__ == "__main__":
    main()
