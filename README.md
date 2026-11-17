# Stock Analysis

Single-user stock research application for Android and a Python data backend.

## Scope

- Tushare, AKShare, and BaoStock adapters
- Stock search and daily market data
- Local watchlist on Android
- Private on-device persistence for the selected stock, search input, latest predictions, and dated prediction history
- Parquet storage with ODS/DWD/DWS/ADS layers
- Data quality checks
- Technical indicators
- Comprehensive stock analysis for returns, drawdown, volatility, volume, and trend state
- Time-series analysis with reproducible random seeds
- No news, analyst consensus, announcement events, social sentiment, macro factors, login, real-time quotes, Spark, or multi-user services

## Layout

```text
backend/
  app/
    api/
    providers/
    services/
  tests/
data/
  ods/
  dwd/
  dws/
  ads/
  quality/
```

Prediction is organized into three configured model groups in
`backend/config/model_groups.json`: Nasdaq index, a 50-stock A-share
growth-technology group, and a Nasdaq-100
tracking-product group covering
QQQ, QQQM, QQQE, QTEC, QQQI, QQMG, QLD, TQQQ, SQQQ, QID, and PSQ.
The group intentionally includes leveraged and inverse products because it
represents the full Nasdaq-100 ETF product family.

Configured horizons are independent per group: Nasdaq index `10` trading
days, growth technology `10` trading days, and Nasdaq-100 tracking products
`20` trading days.
Each group uses the same fixed GBM hyperparameters
with five-seed stock-bagging. The Nasdaq group uses time-block sampling because
it is a single index series. Each walk-forward fold purges the target horizon
between the final training label and the first test date.

Model groups can be audited before publication:

```powershell
backend\.venv\Scripts\python.exe backend\scripts\audit_model_group.py `
  --group us_ndx_etfs --horizon 20
```

The audit checks per-symbol history, factor and label coverage, cross-sectional
breadth, product-risk comparability, purged walk-forward OOS metrics, and seed
reliability. The prediction publisher rejects groups marked `require_audit`
unless the latest audit status is `pass`.

## Backend Setup

```powershell
cd D:\DevEcoStudioProjects\gupiao\backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API documentation.

## Data Quality

The first pipeline milestone checks:

- Field-level and key-field missing rates
- Duplicate stock/date/source records
- Date coverage and observed trading days
- Invalid prices and anomalous volume
- Missing financial fields
- Cross-provider close and volume differences
- Latest ingestion status

## Implemented

- FastAPI health and provider-status endpoints
- Tushare, AKShare, and BaoStock provider adapters
- Numeric stock-code search with provider fallback
- Daily OHLCV normalization
- Separate raw, forward-adjusted, and backward-adjusted daily series
- ODS/DWD Parquet persistence
- Duplicate removal and invalid-price flags
- Daily raw and cleaned quality reports
- Pipeline run history
- DWS technical indicators: MA, EMA, MACD, RSI, KDJ, Bollinger Bands, ATR, volatility, volume ratios
- Android analysis view with 5/20/60/120/240-day returns, annualized volatility, maximum drawdown, volume ratios, trend summaries, and data coverage
- Three grouped GBM predictions with five-seed bagging, continuous score rankings, directional labels, and seed dispersion
- Per-group evaluation only: cross-sectional Rank IC, Rank IC volatility, ICIR, strategy return, Sharpe, and maximum drawdown are never pooled across groups
- Point-in-time factor selection inside each walk-forward training fold, using only data available before each rebalance/test node
- Future ten-day excess-return labels versus CSI 300 for A-share groups and versus the S&P 500 for Nasdaq
- Platt probability calibration with raw probability, calibrated probability, calibration shift, and Brier score logged in every report
- Trend eligibility and a trend-filtered backtest kept separate from the model-only backtest
- Prediction and ensemble snapshots remain available after an app restart or temporary backend outage
- Reproducible time-series analysis: ADF/KPSS, ACF/PACF, additive decomposition, ARIMA backtest and forecast, GARCH volatility
- Optional GBM model with 50-stock feature engineering, four-fold walk-forward validation, regression/classification baselines, and future 5-day prediction
- Nested TPE tuning with date-level TimeSeriesSplit, target-horizon purge, embargo, and untouched outer test folds
- GBM v2 market-breadth, industry-neutral, industry-momentum, and Rank IC selection features

## API

```text
GET  /api/health
GET  /api/quality/providers
GET  /api/quality/stocks/{code}
GET  /api/quality/pipeline-runs
GET  /api/quality/sources/compare
GET  /api/stocks/search?q=000001
GET  /api/stocks/{code}/daily
GET  /api/stocks/{code}/indicators
GET  /api/stocks/{code}/financial/latest
GET  /api/stocks/{code}/analysis
GET  /api/stocks/{code}/analysis/backtest
GET  /api/stocks/{code}/model/ensemble
GET  /api/model/groups
```

The analysis endpoint supports `model=arima`, `model=gbm`, or `model=compare`.
GBM requires at least 3 stocks, 120 trading days, and 1000 valid training rows.
The current universe produces four out-of-sample folds and compares GBM with
zero-return, majority-class, and logistic-regression baselines.

## Daily Sync

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\sync_daily.py `
  --code 000001 `
  --start 20260101 `
  --end 20261231 `
  --source baostock
```

## One-click Prediction Publishing

Edit `backend/publish_config.json` once so `host` and `remote_data_root` match
the server. `remote_data_root` must be the backend `STOCK_DATA_DIR`; published
files are written under its `ads/group_bagging` and `ads/model_manifest`
directories.

Double-click `backend/publish_predictions.bat`. The tool will:

1. Optionally sync the latest daily data when `sync_before_train` is enabled.
2. Train all configured model groups.
3. Create a timestamped manifest and JSON report package.
4. Prompt for the SSH password without storing it.
5. Upload the reports atomically over SFTP/SSH.

For packaging-only tests:

```powershell
.\.venv\Scripts\python.exe scripts\publish_predictions.py `
  --package-only --skip-train
```

Provider smoke tests:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_providers.py --query 000001
.\.venv\Scripts\python.exe scripts\smoke_daily.py `
  --provider baostock --code 000001 --start 20261101 --end 20261113
.\.venv\Scripts\python.exe scripts\compute_indicators.py --code 000001
.\.venv\Scripts\python.exe scripts\compare_sources.py `
  --code 000001 --start 20261101 --end 20261113
.\.venv\Scripts\python.exe scripts\analyze_stock.py `
  --code 000001 --horizon 5 --seed 42
.\.venv\Scripts\python.exe scripts\build_gbm_features.py
.\.venv\Scripts\python.exe scripts\analyze_gbm.py --code 000001 --seed 42
.\.venv\Scripts\python.exe scripts\backtest_gbm.py --code 000001
.\.venv\Scripts\python.exe scripts\sensitivity_gbm.py --code 000001
.\.venv\Scripts\python.exe scripts\nested_tune_gbm.py `
  --code 000001 --horizon 10 --trials 20
.\.venv\Scripts\python.exe scripts\nested_tune_seeds.py `
  --code 000001 --horizon 10 --seeds 7,2024 --reuse-seed 42
.\.venv\Scripts\python.exe scripts\ensemble_seeds.py `
  --code 000001 --horizon 10 --top-fraction 0.1
.\.venv\Scripts\python.exe scripts\train_model_groups.py --all
.\.venv\Scripts\python.exe scripts\sync_universe.py
```

The first universe sync can be run in full mode:

```powershell
.\.venv\Scripts\python.exe scripts\sync_universe.py `
  --full --start 20230101 --end 20261113 --source baostock
```

## Current Provider Status

- BaoStock: verified for stock search and daily bars
- AKShare: adapter implemented; catalog source can time out and is used as fallback
- Tushare: adapter implemented; requires `TUSHARE_TOKEN`

