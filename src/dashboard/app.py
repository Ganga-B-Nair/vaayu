"""
Vaayu — next-day air quality advisory dashboard.

Reads precomputed artefacts from data/processed/. Computes nothing live.

    streamlit run src/dashboard/app.py

If artefacts are missing:
    python -m src.pipeline explore
    python -m src.pipeline run
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

# Validated categorical slots (light surface); sequential/neutral elsewhere.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

st.set_page_config(page_title="Vaayu — air quality advisory", layout="wide",
                   initial_sidebar_state="collapsed")


def style(fig: go.Figure, h: int = 380) -> go.Figure:
    fig.update_layout(
        height=h, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=INK, size=13),
        margin=dict(l=10, r=10, t=72, b=14), hovermode="closest",
        title=dict(y=0.96, yanchor="top", font=dict(size=15)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    font=dict(size=12)))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID,
                     title_font=dict(color=MUTED, size=12))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID,
                     title_font=dict(color=MUTED, size=12))
    return fig


@st.cache_data
def load():
    out = {}
    f = PROC / "results.json"
    out["results"] = json.loads(f.read_text()) if f.exists() else None
    for k, n in [("lockdown", "lockdown.csv"), ("coverage", "coverage.csv"),
                 ("shap", "shap_importance.csv"),
                 ("gain", "feature_importance.csv")]:
        p = PROC / n
        out[k] = pd.read_csv(p) if p.exists() else None
    p = PROC / "features.parquet"
    out["features"] = pd.read_parquet(p) if p.exists() else None
    return out


D = load()
R = D["results"]

st.title("Vaayu")
st.caption("Next-day air quality advisory forecasting · CPCB station data · "
           "26 Indian cities · 2015–2020")

if R is None:
    st.error("No artefacts found. Run the pipeline first:")
    st.code("python -m src.pipeline explore\npython -m src.pipeline run")
    st.stop()

tabs = st.tabs(["Overview", "Data quality", "Forecast performance",
                "What drives it", "COVID lockdown"])

# --------------------------------------------------------------------------
with tabs[0]:
    q, sp, m = R["quality"], R["split"], R["models"]
    a, b, c, d = st.columns(4)
    a.metric("Cities", q["cities"], f"{R['features']['cities_kept']} modelled")
    b.metric("Daily records", f"{q['rows']:,}", f"{q['date_min']} → {q['date_max']}")
    if "xgboost" in m:
        gain = m["xgboost"]["f1"] - m["persistence"]["f1"]
        c.metric("Best model F1", f"{m['xgboost']['f1']:.3f}", f"{gain:+.3f} vs persistence")
    d.metric("Advisory days", f"{sp['advisory_rate_test']:.1%}", "in the test period")

    st.markdown(f"""
**The question.** CPCB public health advisories begin at AQI > {R['advisory_threshold']}
("Poor"), where sensitive groups are told to limit outdoor exposure. Advisories
are only useful issued *in advance*, so the task is: using everything known
today, will tomorrow breach {R['advisory_threshold']}?

**The honest baseline.** Air quality is strongly autocorrelated — tomorrow
usually resembles today. "Tomorrow = today" therefore scores
**{m['persistence']['accuracy']:.1%} accuracy** on its own. Any model that
cannot beat that has demonstrated nothing, so persistence is reported beside
every model here.

**The test period stops on 24 March 2020**, the day before India's COVID
lockdown. Including the lockdown would measure the model's reaction to an
unprecedented regime change rather than its forecasting skill. The lockdown is
analysed separately, on its own tab, as a natural experiment with a real and
documented date.
""")

# --------------------------------------------------------------------------
with tabs[1]:
    q = R["quality"]
    st.subheader("Physically impossible values in the published data")
    a, b = st.columns(2)
    a.metric("PM2.5 recorded above PM10", f"{q['pm25_exceeds_pm10']:,}",
             "impossible by definition")
    b.metric("AQI above the CPCB scale maximum", f"{q['aqi_above_scale_max']:,}",
             f"observed max {q['aqi_observed_max']:.0f} vs scale max 500")
    st.markdown("""
PM2.5 particles are, by definition, a **subset** of PM10 — every particle under
2.5 µm is also under 10 µm. A PM2.5 reading above its own PM10 reading is not
an extreme value, it is a wrong one, and it indicates the two instruments
disagree. The CPCB AQI scale is likewise defined only up to 500.

Both are nulled rather than having their rows dropped: a faulty PM10 sensor
should not cost that day's AQI observation.
""")
    st.dataframe(pd.DataFrame(R["cleaning"].items(), columns=["action", "rows"]),
                 use_container_width=True, hide_index=True)

    st.subheader("Station coverage decides which cities can be modelled")
    if D["coverage"] is not None:
        cov = D["coverage"].sort_values("coverage")
        kept = set(R["features"].get("dropped", []))
        fig = go.Figure(go.Bar(
            x=cov["coverage"] * 100, y=cov["City"], orientation="h",
            marker=dict(color=np.where(cov["City"].isin(kept), MUTED, C1)),
            hovertemplate="%{y}<br>%{x:.1f}% of days have an AQI reading<extra></extra>"))
        fig.add_vline(x=R["features"]["min_coverage"] * 100,
                      line=dict(color=C2, width=2, dash="dot"),
                      annotation_text=f"cut at {R['features']['min_coverage']:.0%}")
        fig.update_layout(title="AQI coverage by city")
        fig.update_xaxes(title="% of days with a reading")
        st.plotly_chart(style(fig, 620), use_container_width=True)
        st.caption(f"Grey cities fall below the cut and are excluded from "
                   f"modelling: {', '.join(R['features']['dropped'])}. "
                   "Forecasting from mostly-imputed history would produce "
                   "metrics that describe the imputation, not the air.")
        st.dataframe(D["coverage"], use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
with tabs[2]:
    m = R["models"]
    st.subheader("Every model is reported against persistence")
    rows = []
    for name, r in m.items():
        rows.append({"model": name, "accuracy": r["accuracy"], "precision": r["precision"],
                     "recall": r["recall"], "F1": r["f1"],
                     "ROC AUC": r.get("roc_auc"), "missed alerts": r["missed_alerts"],
                     "false alarms": r["false_alarms"]})
    tbl = pd.DataFrame(rows)
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.subheader("The operational trade-off")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=tbl["model"], y=tbl["missed alerts"], name="missed alerts",
                         marker_color=C2,
                         hovertemplate="%{x}<br>%{y} missed<extra></extra>"))
    fig.add_trace(go.Bar(x=tbl["model"], y=tbl["false alarms"], name="false alarms",
                         marker_color=C1,
                         hovertemplate="%{x}<br>%{y} false alarms<extra></extra>"))
    fig.update_layout(title="Missed alerts vs false alarms", barmode="group", bargap=0.35)
    fig.update_yaxes(title="days in the test period")
    st.plotly_chart(style(fig, 360), use_container_width=True)

    if "logistic" in m and "xgboost" in m:
        lg, xg = m["logistic"], m["xgboost"]
        if lg["missed_alerts"] < xg["missed_alerts"]:
            st.warning(f"""
**The two models disagree about which is better, and the metric decides.**

XGBoost wins on accuracy ({xg['accuracy']:.3f} vs {lg['accuracy']:.3f}) and F1
({xg['f1']:.3f} vs {lg['f1']:.3f}). But logistic regression misses
**{lg['missed_alerts']} advisory days against XGBoost's {xg['missed_alerts']}** —
{xg['missed_alerts'] - lg['missed_alerts']} fewer people sent outside on a day
that turned out to be hazardous.

For a public health advisory those errors are not equivalent. A false alarm
costs an unnecessary precaution; a missed alert costs exposure. Selecting on
accuracy silently assumes they cost the same. Which model is "best" depends on
a loss function that has to be stated rather than inherited from the default
metric.
""")

# --------------------------------------------------------------------------
with tabs[3]:
    st.subheader("What the model actually uses")
    src = D["shap"] if D["shap"] is not None else D["gain"]
    if src is not None:
        col = "mean_abs_shap" if "mean_abs_shap" in src.columns else "gain"
        top = src.head(15).sort_values(col)
        fig = go.Figure(go.Bar(
            x=top[col], y=top["feature"], orientation="h", marker_color=C1,
            hovertemplate="%{y}<br>%{x:.3f}<extra></extra>"))
        fig.update_layout(title=("Mean |SHAP| — average impact on a prediction"
                                 if col == "mean_abs_shap" else "XGBoost gain"))
        fig.update_xaxes(title=col.replace("_", " "))
        st.plotly_chart(style(fig, 480), use_container_width=True)
        st.dataframe(src.head(20), use_container_width=True, hide_index=True)
        st.info("""
**Read this carefully before quoting it.** The strongest features are
*today's* PM2.5 and PM10 — not the lagged AQI values. That is not leakage:
today's readings are genuinely available when a midnight advisory for tomorrow
is issued. But it is an operational assumption, and it means the system
requires same-day monitoring data to be complete before it can run. A forecast
issued at midday, on partial data, would be a different and harder problem.
""")

# --------------------------------------------------------------------------
with tabs[4]:
    st.subheader("India's COVID lockdown as a natural experiment")
    st.markdown("""
India's nationwide lockdown began **25 March 2020**. Industry, construction and
most road traffic stopped almost overnight. That is an emission reduction no
policy would ever be allowed to run deliberately — and it has a real, documented
date, so no synthetic component is needed anywhere in this analysis.

Each city's median PM2.5 for **April–May 2020** is compared against the same
two months in 2018 and 2019. Using identical calendar months removes
seasonality, which is the dominant signal in Indian air quality; comparing
April 2020 to January 2020 would mostly measure the end of winter.
""")
    if D["lockdown"] is not None:
        lk = D["lockdown"].sort_values("change_vs_baseline_pct")
        fig = go.Figure(go.Bar(
            x=lk["change_vs_baseline_pct"], y=lk["City"], orientation="h",
            marker_color=C3,
            hovertemplate="%{y}<br>%{x:.1f}% vs 2018–19 baseline<extra></extra>"))
        fig.update_layout(title="Change in median PM2.5, April–May 2020 vs 2018–19")
        fig.update_xaxes(title="% change")
        st.plotly_chart(style(fig, 560), use_container_width=True)
        st.dataframe(lk, use_container_width=True, hide_index=True)

        worst = lk.iloc[0]
        st.caption(
            f"Largest reduction: **{worst['City']}** at "
            f"{worst['change_vs_baseline_pct']:.1f}%. "
            "Cities with poor monitoring coverage — Mumbai in particular, at "
            "under 40% — produce unreliable medians here and their figures "
            "should not be read as measurements of the lockdown effect."
        )
        st.warning("""
**What this does and does not establish.** These are before-and-after
comparisons against a seasonal baseline, not a controlled estimate. Weather
in April–May 2020 was not identical to 2018–19, and no control group was
unaffected by the lockdown — it was national. The magnitudes are large enough
and consistent enough across cities to be difficult to explain by weather
alone, but the honest claim is "consistent with a large emission-driven
reduction", not a point estimate of a causal effect.
""")
