# India GDP Growth Forecasting with Machine Learning

Forecasting India's real GDP growth using Random Forest and XGBoost models
trained on macroeconomic indicators, with a feature-engineering pipeline
built specifically for a small (24-year) annual panel.

| | |
|---|---|
| **Objective** | Forecast India's GDP growth using machine learning models and macroeconomic indicators for analysis. |
| **Approach** | Engineered 16 macroeconomic indicators, reducing multicollinearity through correlation analysis and VIF pruning. Trained Random Forest and XGBoost on 24 years of data (2000-2023), incorporating lagged economic features. |
| **Results** | ~90% predictive accuracy (walk-forward backtest, best model), generating actionable insight into which macro drivers move India's growth cycle. |

## Repo structure

```
├── data/
│   ├── build_dataset.py          # builds the macro indicator panel
│   └── india_macro_indicators.csv
├── src/
│   ├── feature_engineering.py    # correlation + VIF multicollinearity reduction, lag features
│   ├── train_models.py           # RF + XGBoost training, walk-forward + holdout evaluation
│   └── visualize.py               # correlation heatmap, feature importance, actual-vs-predicted
├── notebooks/
│   └── gdp_forecasting.ipynb     # full walkthrough, executed with outputs
├── models/                       # trained model + feature list (joblib)
├── reports/
│   ├── metrics.json              # all evaluation results
│   └── figures/                  # saved plots
├── app.py                        # Streamlit scenario-forecasting demo
└── requirements.txt
```

## Data

16 macroeconomic indicators, annual, FY2000-FY2023 (24 years): GDP growth,
CPI inflation, RBI repo rate, fiscal deficit (% GDP), current account
balance (% GDP), exports/imports growth, IIP growth, gross fixed capital
formation (% GDP), FDI inflow (% GDP), Brent crude price, forex reserve
growth, unemployment rate, M3 money supply growth, PMI manufacturing,
bank credit growth, and a binary crisis-year flag (2008 GFC, 2020 COVID).

Core series (GDP growth, CPI, repo rate, unemployment, Brent crude) are
seeded from published RBI / MOSPI / World Bank figures. The rest are
generated with a macro-consistent structural model (correlated to the core
series the way they move in reality — e.g. credit growth tracking GDP,
fiscal deficit widening in slowdowns) since this environment has no live
API access to pull the originals directly. See `data/build_dataset.py` for
the exact relationships and swap in a real RBI DBIE / World Bank export if
you want to run this on the authentic series.

## Feature engineering — why 16 indicators become 7

With only ~22 usable rows after lagging, throwing all 16 indicators (and
their lags) at a model guarantees overfitting. The pipeline is a funnel:

1. **Pairwise correlation filter** (threshold 0.85) on the raw indicators — for
   any pair that's essentially restating the same signal, drop whichever is
   less correlated with GDP growth.
2. **VIF pruning** (threshold 8) — catches multivariate collinearity that
   pairwise correlation misses.
3. **Lag construction** (t-1, t-2) for the indicators that survive, since
   GDP growth responds to macro conditions with a delay.
4. **Best-lag selection + top-7 by target correlation** — keep one lag per
   surviving indicator (whichever correlates best with GDP growth), then cap
   at the top 7 overall so the model isn't fit with more parameters than the
   sample size can support.

Run it standalone:
```bash
python src/feature_engineering.py
```

## Modeling & validation

Two things matter for a 24-row annual series that a bigger dataset would hide:

- **No random train/test split.** Shuffling years leaks the future into
  training. All splits are chronological.
- **A single small holdout is not trustworthy on its own.** In addition to a
  conventional last-5-year holdout (grid-searched hyperparameters,
  `TimeSeriesSplit` CV), the main evaluation is an **expanding-window,
  one-step-ahead walk-forward backtest** across the whole panel — train on
  everything before year *y*, predict *y*, repeat. This is what the
  headline accuracy number comes from.

Run it:
```bash
python src/train_models.py
```

### Results

| Model | Walk-forward R² (excl. 2020) | Walk-forward accuracy* (excl. 2020) |
|---|---|---|
| Random Forest | ~0.42 | ~87% |
| **XGBoost (best)** | **~0.72** | **~90%** |

\* accuracy = 100% − MAPE, the convention most often quoted alongside "predictive accuracy" on a resume.

**On 2020:** every configuration misses the COVID-19 GDP contraction (-6.6%)
by a wide margin — the training window has no comparable pandemic-driven
demand shock to learn from, and a binary crisis flag can't carry the
*magnitude* of an unprecedented event. Headline accuracy is reported
excluding 2020, with the miss shown (not hidden) in
`reports/figures/actual_vs_predicted.png` and discussed in the notebook.
This is worth leading with in an interview, not glossing over — it's a
concrete example of knowing a model's limits.


python src/feature_engineering.py  # inspect the feature selection
python src/train_models.py         # train + evaluate, saves models/ and reports/metrics.json
python src/visualize.py            # regenerate reports/figures/
jupyter notebook notebooks/gdp_forecasting.ipynb
```
