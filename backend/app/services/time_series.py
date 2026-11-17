from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from app.indicators import select_primary_daily
from app.symbols import normalize_stock_code


@dataclass(frozen=True)
class TimeSeriesResult:
    report: dict[str, Any]


class TimeSeriesService:
    def analyze(
        self,
        code: str,
        daily: pl.DataFrame,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        horizon: int = 5,
        seed: int = 42,
        seasonal_period: int = 5,
        max_arima_order: int = 2,
        include_volatility: bool = True,
    ) -> TimeSeriesResult:
        normalized_code = normalize_stock_code(code)
        np.random.seed(seed)
        random.seed(seed)

        selected = select_primary_daily(daily, adjust_type="qfq").sort(
            ["stock_code", "trade_date"]
        )
        if start_date:
            selected = selected.filter(
                pl.col("trade_date")
                >= pl.lit(start_date).str.strptime(pl.Date, "%Y-%m-%d")
            )
        if end_date:
            selected = selected.filter(
                pl.col("trade_date")
                <= pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d")
            )
        selected = selected.filter(
            pl.col("close").is_not_null() & (pl.col("close") > 0)
        )
        if selected.height < 40:
            raise ValueError("At least 40 valid adjusted closing prices are required")

        frame = selected.select(
            "trade_date",
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64)
            if "volume" in selected.columns
            else pl.lit(None).alias("volume"),
            pl.col("source"),
        ).to_pandas()
        trade_dates = pd.DatetimeIndex(pd.to_datetime(frame["trade_date"]))
        series = pd.Series(
            frame["close"].to_numpy(dtype=float),
            index=pd.RangeIndex(frame.shape[0]),
            name="close",
        )

        stationarity = self._stationarity(series)
        order = self._select_order(
            series,
            d=int(stationarity["suggested_differencing"]),
            max_order=max_arima_order,
        )
        report = {
            "metadata": {
                "stock_code": normalized_code,
                "source": str(selected["source"][0]),
                "adjust_type": "qfq",
                "observations": series.size,
                "start_date": trade_dates[0].date().isoformat(),
                "end_date": trade_dates[-1].date().isoformat(),
                "seed": seed,
                "generated_on": date.today().isoformat(),
            },
            "summary": self._summary(series),
            "stationarity": stationarity,
            "autocorrelation": self._autocorrelation(series),
            "decomposition": self._decomposition(
                series,
                trade_dates=trade_dates,
                seasonal_period=seasonal_period,
            ),
            "forecast": self._forecast(
                series,
                order,
                horizon,
                last_market_date=trade_dates[-1],
            ),
        }
        if include_volatility:
            report["volatility"] = self._volatility(series, horizon)
        else:
            report["volatility"] = {"available": False, "reason": "disabled"}
        return TimeSeriesResult(report=report)

    @staticmethod
    def _summary(series: pd.Series) -> dict[str, Any]:
        returns = series.pct_change().dropna()
        return {
            "latest_close": _safe_float(series.iloc[-1]),
            "mean_close": _safe_float(series.mean()),
            "std_close": _safe_float(series.std(ddof=1)),
            "min_close": _safe_float(series.min()),
            "max_close": _safe_float(series.max()),
            "total_return": _safe_float(series.iloc[-1] / series.iloc[0] - 1),
            "daily_return_mean": _safe_float(returns.mean()),
            "daily_return_std": _safe_float(returns.std(ddof=1)),
            "positive_day_ratio": _safe_float((returns > 0).mean()),
            "skewness": _safe_float(series.skew()),
            "excess_kurtosis": _safe_float(series.kurt()),
        }

    @staticmethod
    def _stationarity(series: pd.Series) -> dict[str, Any]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf = adfuller(
                series,
                autolag="AIC",
                result_object=True,
            )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_result_value = kpss(
                    series,
                    regression="c",
                    nlags="auto",
                    result_object=True,
                )
            kpss_result: dict[str, Any] = {
                "statistic": _safe_float(kpss_result_value.statistic),
                "p_value": _safe_float(kpss_result_value.pvalue),
                "lags": int(kpss_result_value.lags),
                "critical_values": {
                    key: _safe_float(value)
                    for key, value in kpss_result_value.critical_values.items()
                },
                "stationary": bool(kpss_result_value.pvalue >= 0.05),
            }
        except Exception as exc:
            kpss_result = {"error": str(exc)}

        return {
            "adf": {
                "statistic": _safe_float(adf.statistic),
                "p_value": _safe_float(adf.pvalue),
                "lags": int(adf.lags),
                "observations": int(adf.nobs),
                "critical_values": {
                    key: _safe_float(value)
                    for key, value in adf.critical_values.items()
                },
                "stationary": bool(adf.pvalue < 0.05),
            },
            "kpss": kpss_result,
            "suggested_differencing": 0 if adf.pvalue < 0.05 else 1,
        }

    @staticmethod
    def _autocorrelation(series: pd.Series) -> dict[str, Any]:
        max_lags = min(40, max(1, series.size // 2 - 1))
        acf_values = acf(series, nlags=max_lags, fft=True)
        pacf_values = pacf(series, nlags=max_lags, method="ywm")
        return {
            "max_lags": max_lags,
            "lags": list(range(max_lags + 1)),
            "acf": [_safe_float(value) for value in acf_values],
            "pacf": [_safe_float(value) for value in pacf_values],
        }

    @staticmethod
    def _decomposition(
        series: pd.Series,
        *,
        trade_dates: pd.DatetimeIndex,
        seasonal_period: int,
    ) -> dict[str, Any]:
        period = max(2, seasonal_period)
        if series.size < period * 2:
            return {
                "available": False,
                "reason": "Not enough observations for seasonal decomposition",
            }
        try:
            decomposed = seasonal_decompose(
                series,
                model="additive",
                period=period,
                extrapolate_trend="period",
            )
        except Exception as exc:
            return {"available": False, "reason": str(exc)}

        points = []
        for index, close, trend, seasonal, residual in zip(
            trade_dates,
            series.to_numpy(),
            decomposed.trend.to_numpy(),
            decomposed.seasonal.to_numpy(),
            decomposed.resid.to_numpy(),
            strict=True,
        ):
            points.append(
                {
                    "trade_date": index.date().isoformat(),
                    "close": _safe_float(close),
                    "trend": _safe_float(trend),
                    "seasonal": _safe_float(seasonal),
                    "residual": _safe_float(residual),
                }
            )
        return {
            "available": True,
            "model": "additive",
            "period": period,
            "points": points,
        }

    @staticmethod
    def _select_order(
        series: pd.Series,
        *,
        d: int,
        max_order: int,
    ) -> tuple[int, int, int]:
        best_order = (1, d, 0)
        best_aic = math.inf
        limit = max(0, min(max_order, 3))
        for p in range(limit + 1):
            for q in range(limit + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    fit = ARIMA(
                        series,
                        order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit()
                    aic = _safe_float(fit.aic)
                    if aic is not None and aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                except Exception:
                    continue
        return best_order

    @staticmethod
    def _forecast(
        series: pd.Series,
        order: tuple[int, int, int],
        horizon: int,
        *,
        last_market_date: pd.Timestamp,
    ) -> dict[str, Any]:
        test_size = min(40, max(5, math.ceil(series.size * 0.2)))
        train = series.iloc[:-test_size]
        test = series.iloc[-test_size:]
        metrics: dict[str, Any] = {"test_rows": test_size}
        try:
            backtest_fit = ARIMA(
                train,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()
            predictions = np.asarray(backtest_fit.forecast(steps=test_size), dtype=float)
            actual = test.to_numpy(dtype=float)
            errors = actual - predictions
            metrics.update(
                {
                    "rmse": _safe_float(np.sqrt(np.mean(errors**2))),
                    "mae": _safe_float(np.mean(np.abs(errors))),
                    "mape": _safe_float(
                        np.mean(np.abs(errors) / np.maximum(np.abs(actual), 1e-12))
                    ),
                }
            )
        except Exception as exc:
            metrics["error"] = str(exc)

        fit = ARIMA(
            series,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
        forecast_values = fit.forecast(steps=max(1, horizon))
        confidence = fit.get_forecast(steps=max(1, horizon)).conf_int(alpha=0.2)
        future_dates = pd.bdate_range(
            start=last_market_date + pd.offsets.BDay(1),
            periods=max(1, horizon),
        )
        points = []
        for index, (forecast_date, value, lower, upper) in enumerate(
            zip(
                future_dates,
                np.asarray(forecast_values, dtype=float),
                confidence.iloc[:, 0].to_numpy(),
                confidence.iloc[:, 1].to_numpy(),
                strict=True,
            ),
            start=1,
        ):
            points.append(
                {
                    "step": index,
                    "trade_date": forecast_date.date().isoformat(),
                    "forecast": _safe_float(value),
                    "lower_80": _safe_float(lower),
                    "upper_80": _safe_float(upper),
                }
            )
        return {
            "model": "ARIMA",
            "order": {"p": order[0], "d": order[1], "q": order[2]},
            "aic": _safe_float(fit.aic),
            "bic": _safe_float(fit.bic),
            "backtest": metrics,
            "horizon": max(1, horizon),
            "points": points,
        }

    @staticmethod
    def _volatility(series: pd.Series, horizon: int) -> dict[str, Any]:
        try:
            from arch import arch_model

            returns = series.pct_change().dropna().to_numpy(dtype=float) * 100
            model = arch_model(
                returns,
                mean="Constant",
                vol="GARCH",
                p=1,
                q=1,
                dist="t",
                rescale=False,
            )
            fit = model.fit(disp="off")
            forecast = fit.forecast(horizon=max(1, horizon), reindex=False)
            variance_value = forecast.variance
            if hasattr(variance_value, "iloc"):
                variance_value = variance_value.iloc[-1]
            else:
                variance_value = np.asarray(variance_value)[-1]
            if hasattr(variance_value, "to_numpy"):
                variance_value = variance_value.to_numpy()
            variances = np.asarray(variance_value, dtype=float)
            daily_volatility = np.sqrt(variances)
            annualized = [
                daily_percent_to_annualized(value)
                for value in daily_volatility
            ]
            conditional_volatility = fit.conditional_volatility
            if hasattr(conditional_volatility, "iloc"):
                conditional_volatility = conditional_volatility.iloc[-1]
            else:
                conditional_volatility = np.asarray(conditional_volatility)[-1]
            return {
                "available": True,
                "model": "GARCH(1,1)",
                "latest_annualized": _safe_float(
                    daily_percent_to_annualized(conditional_volatility)
                ),
                "forecast_annualized": [
                    _safe_float(value) for value in annualized
                ],
                "aic": _safe_float(fit.aic),
                "bic": _safe_float(fit.bic),
            }
        except Exception as exc:
            return {"available": False, "reason": str(exc)}


def daily_percent_to_annualized(value: float) -> float:
    return float(value) * math.sqrt(252) / 100.0


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, 8)
