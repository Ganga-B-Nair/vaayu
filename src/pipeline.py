"""
Vaayu — next-day air quality advisory forecasting for Indian cities.

Two stages, same pattern as before: `explore` shows you the data and the
decisions you have to make; `run` executes the pipeline you specified and
saves artefacts the dashboard reads.

    python -m src.pipeline explore
    python -m src.pipeline run

THE QUESTION
    Can tomorrow's air quality be predicted well enough, today, to issue a
    public health advisory? CPCB advisories bite at AQI > 200 ("Poor"),
    where sensitive groups are told to limit outdoor exposure. So the target
    is binary and operational: will tomorrow breach 200?

WHY A PERSISTENCE BASELINE IS NOT OPTIONAL
    Air quality is highly autocorrelated: tomorrow usually resembles today.
    "Tomorrow = today" is therefore a strong baseline, and any model that
    cannot beat it has demonstrated nothing. Reporting 85% accuracy without
    reporting what persistence scores is the single most common way air
    quality forecasting papers mislead. This pipeline always reports both.

TEMPORAL SPLIT, AND WHY THE TEST SET STOPS AT 24 MARCH 2020
    India's COVID lockdown began 25 March 2020 and changed emissions
    abruptly and enormously. Including it in the test set would measure the
    model's response to an unprecedented regime change rather than its
    forecasting skill. The lockdown is analysed separately, as a natural
    experiment, in its own stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# CPCB AQI bands
AQI_MAX_VALID = 500          # the CPCB scale tops out here
ADVISORY_THRESHOLD = 200     # "Poor" — where advisories begin
LOCKDOWN_START = pd.Timestamp("2020-03-25")
TRAIN_END = pd.Timestamp("2018-12-31")
TEST_END = pd.Timestamp("2020-03-24")   # day before lockdown

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]


def _p(rel: str) -> Path:
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------
# load + quality
# --------------------------------------------------------------------------

def load_city_day() -> pd.DataFrame:
    df = pd.read_csv(_p("data/raw/city_day.csv"), parse_dates=["Date"])
    return df.sort_values(["City", "Date"]).reset_index(drop=True)


def quality_flags(df: pd.DataFrame) -> dict:
    """Physically impossible or out-of-scale values. Each is a real defect
    in CPCB's published data, and each needs a stated decision."""
    pm_inversion = (df["PM2.5"] > df["PM10"]) & df["PM10"].notna() & df["PM2.5"].notna()
    return {
        "rows": len(df),
        "cities": int(df["City"].nunique()),
        "date_min": str(df["Date"].min().date()),
        "date_max": str(df["Date"].max().date()),
        # PM2.5 is a SUBSET of PM10 by definition (particles under 2.5um are
        # also under 10um). PM2.5 > PM10 is physically impossible.
        "pm25_exceeds_pm10": int(pm_inversion.sum()),
        # The CPCB AQI scale is defined up to 500.
        "aqi_above_scale_max": int((df["AQI"] > AQI_MAX_VALID).sum()),
        "aqi_observed_max": float(df["AQI"].max()),
        "pm10_at_1000_ceiling": int((df["PM10"] == 1000.0).sum()),
        "pm25_at_1000_ceiling": int((df["PM2.5"] == 1000.0).sum()),
        "aqi_missing_pct": round(float(df["AQI"].isna().mean() * 100), 1),
        "pm25_missing_pct": round(float(df["PM2.5"].isna().mean() * 100), 1),
    }


def coverage_by_city(df: pd.DataFrame, col: str = "AQI") -> pd.DataFrame:
    g = (df.groupby("City")
         .agg(n_days=("Date", "size"),
              present=(col, lambda s: int(s.notna().sum())),
              first=("Date", "min"), last=("Date", "max"))
         .assign(coverage=lambda x: (x["present"] / x["n_days"]).round(3))
         .sort_values("coverage", ascending=False))
    return g


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Null out impossible values rather than dropping rows — a bad PM10
    reading should not cost you that day's AQI observation."""
    d = df.copy()
    log = {}

    bad = (d["PM2.5"] > d["PM10"]) & d["PM10"].notna() & d["PM2.5"].notna()
    log["pm_inversion_nulled"] = int(bad.sum())
    d.loc[bad, ["PM2.5", "PM10"]] = np.nan

    over = d["AQI"] > AQI_MAX_VALID
    log["aqi_over_scale_nulled"] = int(over.sum())
    d.loc[over, ["AQI", "AQI_Bucket"]] = np.nan

    # Xylene is >60% missing at city level and >78% at station level.
    # Carrying it would mean imputing more than we observe.
    log["dropped_columns"] = ["Xylene"]
    d = d.drop(columns=["Xylene"], errors="ignore")

    return d, log


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

def build_features(df: pd.DataFrame, min_coverage: float = 0.8) -> tuple[pd.DataFrame, dict]:
    """Lagged, rolling and calendar features, built PER CITY.

    Every feature is computed from information available on day t to predict
    day t+1. Nothing from t+1 or later touches the feature set — that is the
    whole discipline of forecasting, and getting it wrong inflates your
    metrics into meaninglessness.
    """
    cov = coverage_by_city(df)
    keep = cov[cov["coverage"] >= min_coverage].index.tolist()
    d = df[df["City"].isin(keep)].copy()

    frames = []
    for city, g in d.groupby("City", sort=False):
        g = g.sort_values("Date").copy()
        # a complete daily index so gaps stay visible as NaN rather than
        # silently collapsing (which would make a 5-day gap look like 1 day)
        g = g.set_index("Date").reindex(
            pd.date_range(g["Date"].min(), g["Date"].max(), freq="D"))
        g["City"] = city
        g.index.name = "Date"

        for col in ["AQI", "PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]:
            if col not in g:
                continue
            for lag in (1, 2, 3, 7):
                g[f"{col}_lag{lag}"] = g[col].shift(lag)
            g[f"{col}_roll3"] = g[col].shift(1).rolling(3, min_periods=2).mean()
            g[f"{col}_roll7"] = g[col].shift(1).rolling(7, min_periods=4).mean()
        g["AQI_delta1"] = g["AQI"].shift(1) - g["AQI"].shift(2)
        g["AQI_roll7_std"] = g["AQI"].shift(1).rolling(7, min_periods=4).std()

        frames.append(g.reset_index())

    out = pd.concat(frames, ignore_index=True)
    out["dayofweek"] = out["Date"].dt.dayofweek
    out["month"] = out["Date"].dt.month
    # cyclical encoding: December and January are adjacent, 12 and 1 are not
    doy = out["Date"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # TARGET: does TOMORROW breach the advisory threshold?
    out["target_aqi_next"] = out.groupby("City")["AQI"].shift(-1)
    out["target"] = (out["target_aqi_next"] > ADVISORY_THRESHOLD).astype("float")
    out.loc[out["target_aqi_next"].isna(), "target"] = np.nan

    return out, {"cities_kept": len(keep), "cities_dropped": int(df["City"].nunique() - len(keep)),
                 "dropped": sorted(set(df["City"].unique()) - set(keep)),
                 "min_coverage": min_coverage}


FEATURE_COLS = None  # set at train time


def feature_columns(df: pd.DataFrame) -> list[str]:
    bad = {"Date", "City", "AQI_Bucket", "target", "target_aqi_next", "AQI"}
    return [c for c in df.columns
            if c not in bad and pd.api.types.is_numeric_dtype(df[c])]


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

def persistence_baseline(d: pd.DataFrame) -> np.ndarray:
    """Tomorrow will be like today. The bar any model must clear."""
    return (d["AQI"] > ADVISORY_THRESHOLD).astype(int).to_numpy()


def evaluate(y_true, y_pred, y_prob=None) -> dict:
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score, confusion_matrix)
    out = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    # For an advisory system a missed alert costs more than a false one.
    out["missed_alerts"] = int(fn)
    out["false_alarms"] = int(fp)
    return out


def temporal_split(df: pd.DataFrame):
    d = df.dropna(subset=["target", "AQI"]).copy()
    train = d[d["Date"] <= TRAIN_END]
    test = d[(d["Date"] > TRAIN_END) & (d["Date"] <= TEST_END)]
    return train, test


# --------------------------------------------------------------------------
# lockdown natural experiment
# --------------------------------------------------------------------------

def lockdown_analysis(df: pd.DataFrame, pollutant: str = "PM2.5") -> pd.DataFrame:
    """Compare Apr-May 2020 (full lockdown) against the same months in the
    two preceding years, per city.

    Using the same calendar months removes seasonality, which is the
    dominant signal in Indian air quality — comparing April 2020 to
    January 2020 would mostly measure the end of winter, not the lockdown.

    This is a REAL event with a documented date. No synthetic component.
    """
    d = df[df["Date"].dt.month.isin([4, 5])].copy()
    d["year"] = d["Date"].dt.year
    d = d[d["year"].isin([2018, 2019, 2020])]

    piv = d.groupby(["City", "year"])[pollutant].median().unstack()
    piv = piv.dropna(subset=[2018, 2019, 2020])
    piv["baseline_18_19"] = piv[[2018, 2019]].mean(axis=1)
    piv["change_vs_2019_pct"] = ((piv[2020] / piv[2019] - 1) * 100).round(1)
    piv["change_vs_baseline_pct"] = ((piv[2020] / piv["baseline_18_19"] - 1) * 100).round(1)
    return piv.round(2).sort_values("change_vs_baseline_pct")


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def stage_explore() -> None:
    df = load_city_day()
    print("=== QUALITY FLAGS (raw) ===")
    q = quality_flags(df)
    for k, v in q.items():
        print(f"  {k:26s} {v}")
    print("""
  Two of these are worth a paragraph each:
   - PM2.5 > PM10 is physically impossible; PM2.5 particles are a subset
     of PM10 by definition. These rows are wrong, not extreme.
   - AQI above 500 is off the CPCB scale entirely.""")

    print("\n=== AQI COVERAGE BY CITY ===")
    cov = coverage_by_city(df)
    print(cov.to_string())
    print(f"\n  cities at >=80% coverage: {(cov['coverage'] >= 0.8).sum()} of {len(cov)}")
    print("  ^ YOU choose the cut. Too strict and you lose the south;")
    print("    too loose and you forecast from mostly-imputed history.")

    print("\n=== LOCKDOWN PREVIEW (median PM2.5, Apr-May) ===")
    d2, _ = clean(df)
    print(lockdown_analysis(d2).to_string())

    print("\nThen run:  python -m src.pipeline run")


def stage_run(min_coverage: float = 0.8) -> None:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    raw = load_city_day()
    df, clean_log = clean(raw)
    feats, feat_log = build_features(df, min_coverage)
    train, test = temporal_split(feats)

    cols = feature_columns(feats)
    print(f"\n=== SPLIT ===\n  features        {len(cols)}")
    print(f"  train  {train['Date'].min().date()} -> {train['Date'].max().date()}  {len(train):,} rows")
    print(f"  test   {test['Date'].min().date()} -> {test['Date'].max().date()}  {len(test):,} rows")
    print(f"  advisory days in train: {train['target'].mean():.1%}")
    print(f"  advisory days in test : {test['target'].mean():.1%}")

    Xtr, ytr = train[cols], train["target"].astype(int)
    Xte, yte = test[cols], test["target"].astype(int)

    results = {}
    print("\n=== MODELS (test set = 2019-01-01 to 2020-03-24, pre-lockdown) ===")

    base_pred = persistence_baseline(test)
    results["persistence"] = evaluate(yte, base_pred)
    print(f"  persistence  acc={results['persistence']['accuracy']:.4f} "
          f"f1={results['persistence']['f1']:.4f} "
          f"missed={results['persistence']['missed_alerts']}")

    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(max_iter=2000, class_weight="balanced"))
    lr.fit(Xtr, ytr)
    results["logistic"] = evaluate(yte, lr.predict(Xte), lr.predict_proba(Xte)[:, 1])
    print(f"  logistic     acc={results['logistic']['accuracy']:.4f} "
          f"f1={results['logistic']['f1']:.4f} "
          f"missed={results['logistic']['missed_alerts']}")

    try:
        from xgboost import XGBClassifier
        pos = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        xgb = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, scale_pos_weight=pos,
            eval_metric="logloss", random_state=42, n_jobs=4)
        xgb.fit(Xtr, ytr)
        prob = xgb.predict_proba(Xte)[:, 1]
        results["xgboost"] = evaluate(yte, (prob >= 0.5).astype(int), prob)
        print(f"  xgboost      acc={results['xgboost']['accuracy']:.4f} "
              f"f1={results['xgboost']['f1']:.4f} "
              f"missed={results['xgboost']['missed_alerts']}")
        joblib.dump({"model": xgb, "features": cols}, _p("models/xgb_advisory.pkl"))

        imp = (pd.DataFrame({"feature": cols, "gain": xgb.feature_importances_})
               .sort_values("gain", ascending=False).head(25))
        imp.to_csv(_p("data/processed/feature_importance.csv"), index=False)

        # SHAP: gain tells you what the model split on; SHAP tells you how
        # much each feature actually moved each prediction, with direction.
        try:
            import shap
            samp = Xte.sample(min(2000, len(Xte)), random_state=42)
            sv = shap.TreeExplainer(xgb).shap_values(samp)
            shap_imp = (pd.DataFrame({"feature": cols,
                                      "mean_abs_shap": np.abs(sv).mean(axis=0)})
                        .sort_values("mean_abs_shap", ascending=False))
            shap_imp.to_csv(_p("data/processed/shap_importance.csv"), index=False)
            print(f"  SHAP computed on {len(samp):,} test rows")
        except Exception as e:
            print(f"  SHAP skipped: {str(e)[:80]}")
    except ImportError:
        print("  xgboost not installed — skipped")

    # Did anything beat persistence?
    if "xgboost" in results:
        d_f1 = results["xgboost"]["f1"] - results["persistence"]["f1"]
        d_miss = results["persistence"]["missed_alerts"] - results["xgboost"]["missed_alerts"]
        print(f"\n  XGBoost vs persistence: F1 {d_f1:+.4f}, "
              f"{d_miss:+d} fewer missed alerts")
        print("  ^ if this is not clearly positive, say so. A model that does")
        print("    not beat 'tomorrow = today' has not earned its place.")

    lock = lockdown_analysis(df)
    lock.to_csv(_p("data/processed/lockdown.csv"))
    print(f"\n=== LOCKDOWN (real event, 25 Mar 2020) ===")
    print(lock[["baseline_18_19", 2020, "change_vs_baseline_pct"]].to_string())

    feats.to_parquet(_p("data/processed/features.parquet"), index=False)
    test.assign(pred=(results.get("xgboost") and prob >= 0.5)
                if "xgboost" in results else base_pred
                ).to_parquet(_p("data/processed/test_predictions.parquet"), index=False)
    coverage_by_city(raw).to_csv(_p("data/processed/coverage.csv"))

    payload = {
        "quality": quality_flags(raw),
        "cleaning": clean_log,
        "features": feat_log,
        "split": {
            "train_end": str(TRAIN_END.date()), "test_end": str(TEST_END.date()),
            "n_train": len(train), "n_test": len(test),
            "advisory_rate_train": round(float(train["target"].mean()), 4),
            "advisory_rate_test": round(float(test["target"].mean()), 4),
            "n_features": len(cols),
        },
        "models": results,
        "advisory_threshold": ADVISORY_THRESHOLD,
    }
    _p("data/processed/results.json").write_text(json.dumps(payload, indent=2, default=str))
    print("\nartefacts written. dashboard:  streamlit run src/dashboard/app.py")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["explore", "run"])
    ap.add_argument("--min-coverage", type=float, default=0.8,
                    help="minimum AQI coverage for a city to be modelled")
    a = ap.parse_args()
    stage_explore() if a.stage == "explore" else stage_run(a.min_coverage)


if __name__ == "__main__":
    main()
