# Vaayu

**Next-day air quality advisory forecasting for Indian cities**

CPCB advisories begin at AQI > 200 ("Poor"), where sensitive groups are told to
limit outdoor exposure. But CPCB reports today's AQI *today* — a measurement,
not a warning. The decisions that matter (move practice indoors, reschedule the
run, carry the inhaler) are made the night before.

**Vaayu forecasts whether tomorrow will breach the advisory threshold, using
everything measurable today.**

---

## Data

[Air Quality Data in India 2015–2020](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
— CPCB station readings. Download and unzip into `data/raw/`.

Measured on the real file: **29,531 city-days, 26 cities, 2015-01-01 to
2020-07-01.**

### Data quality findings

Two defects in the published data that are *wrong*, not merely extreme:

| Defect | Rows | Why it's impossible |
|---|---|---|
| PM2.5 recorded above PM10 | 256 | PM2.5 particles are a subset of PM10 by definition |
| AQI above 500 | 543 | The CPCB scale ends at 500; max observed is 2,049 |

Both are nulled rather than having their rows dropped — a faulty PM10 sensor
should not cost that day's AQI observation.

`Xylene` is dropped entirely (61% missing at city level, 78% at station level):
keeping it would mean imputing more than is observed.

**Seven of 26 cities are excluded** for AQI coverage below 80% — Ahmedabad,
Brajrajnagar, Jorapokhar, Mumbai, Patna, Shillong, Talcher. Mumbai is the
striking one at 38.6%: India's largest city has monitoring too sparse to
forecast from.

## Method

**Target** — binary: does tomorrow's AQI exceed 200? The decision is binary, so
the target is. Nobody acts differently on 203 versus 219.

**Features (59)** — today's pollutant readings; lags at 1/2/3/7 days and 3/7-day
rolling means for AQI, PM2.5, PM10, NO2, CO, SO2, O3; day-on-day AQI change;
7-day AQI volatility; day of week, month, and day-of-year as sine/cosine so that
31 December and 1 January are adjacent rather than 364 apart.

**Split** — temporal. Train through 2018-12-31 (10,276 rows), test
2019-01-01 to 2020-03-24 (6,386 rows). A random split would let 2020 train a
model evaluated on 2016.

**The test set stops the day before India's COVID lockdown (25 March 2020)** so
that forecast evaluation is not contaminated by an unprecedented regime change.
The lockdown is analysed separately.

## Results

| model | accuracy | F1 | missed alerts |
|---|---|---|---|
| persistence ("tomorrow = today") | 0.931 | 0.793 | 217 |
| logistic regression | 0.941 | 0.838 | **78** |
| XGBoost | **0.950** | **0.855** | 114 |

**Persistence scores 93% on its own.** Air quality is strongly autocorrelated,
so any model that cannot beat "tomorrow will be like today" has demonstrated
nothing. XGBoost beats it by 0.062 F1 and catches 103 more advisory days.

### The models disagree, and the metric decides

XGBoost wins accuracy and F1. Logistic regression misses **78** advisory days
against XGBoost's **114**.

For a public health advisory these errors do not cost the same: a false alarm
costs an unnecessary precaution, a missed alert costs exposure. Selecting on
accuracy silently assumes they are equal. **Which model is "best" depends on a
loss function that must be stated rather than inherited from the default
metric.**

### What the model uses

SHAP ranks *today's* PM2.5 and PM10 above the lagged AQI values. This is not
leakage — today's readings are available when a midnight advisory is issued —
but it is an operational assumption: the system requires same-day monitoring
data to be complete before it can run.

## COVID lockdown as a natural experiment

India locked down on **25 March 2020**. Median PM2.5 for April–May 2020 against
the same months in 2018–19:

Chennai −60% · Ahmedabad −59% · Delhi −41% · Thiruvananthapuram −26%

Identical calendar months, to remove seasonality. **A real event with a
documented date — no synthetic component anywhere in this project.**

This is a before-and-after comparison against a seasonal baseline, not a
controlled estimate: weather in 2020 was not identical, and no control group
escaped a national lockdown. The honest claim is "consistent with a large
emission-driven reduction", not a causal point estimate.

## Running it

```bash
python -m src.pipeline explore     # quality, coverage, lockdown preview
python -m src.pipeline run         # train, evaluate, write artefacts
streamlit run src/dashboard/app.py # dashboard
```

The dashboard reads saved artefacts and computes nothing live.

## Limitations

- Forecasts one day ahead, only for cities that already have good monitoring —
  which is most of India's problem, and not something this solves.
- No meteorological inputs, so it cannot anticipate a weather change that
  clears or traps pollution.
- Trained on pre-2019 data; any structural shift since would degrade it.
- The data ends July 2020. A live feed (OpenAQ has a public India API) is what
  would make this operational rather than retrospective.

## Author

GG (Ganga B Nair) — B.Tech Computer Science & AI, MBCET Thiruvananthapuram
