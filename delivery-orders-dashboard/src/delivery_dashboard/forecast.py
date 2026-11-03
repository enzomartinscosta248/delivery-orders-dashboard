"""Previsão de demanda diária (pedidos por dia) e backtest contra os dados reais."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

Forecaster = Callable[[pd.Series, pd.DatetimeIndex], pd.Series]


def daily_series(daily: pd.DataFrame) -> pd.Series:
    """DataFrame (day, orders) -> Series diária contínua (dias sem pedido viram 0)."""
    series = pd.Series(daily["orders"].to_numpy(dtype=float), index=pd.to_datetime(daily["day"]))
    return series.asfreq("D", fill_value=0.0).rename("pedidos reais")


def seasonal_naive(train: pd.Series, future_index: pd.DatetimeIndex) -> pd.Series:
    """Repete o valor do mesmo dia da semana mais recente."""
    last = train.index[-1]
    values = []
    for day in future_index:
        reference = last - pd.Timedelta(days=(last.weekday() - day.weekday()) % 7)
        values.append(float(train.loc[reference]))
    return pd.Series(values, index=future_index)


def weekday_average(train: pd.Series, future_index: pd.DatetimeIndex, weeks: int = 4) -> pd.Series:
    """Média do mesmo dia da semana nas últimas `weeks` semanas."""
    recent = train.iloc[-7 * weeks:]
    by_weekday = recent.groupby(recent.index.weekday).mean()
    values = by_weekday.reindex(future_index.weekday).to_numpy(dtype=float)
    return pd.Series(values, index=future_index)


def ridge_calendar(train: pd.Series, future_index: pd.DatetimeIndex) -> pd.Series:
    """Regressão com tendência linear + dia da semana (aprende o nível e o ritmo semanal)."""
    origin = train.index[0]

    def features(index: pd.DatetimeIndex) -> np.ndarray:
        trend = np.asarray((index - origin).days, dtype=float)[:, None] / 30.0
        weekday = np.eye(7)[np.asarray(index.weekday)]
        return np.hstack([trend, weekday])

    model = Ridge(alpha=1.0).fit(features(train.index), train.to_numpy(dtype=float))
    return pd.Series(np.clip(model.predict(features(future_index)), 0, None), index=future_index)


FORECASTERS: dict[str, Forecaster] = {
    "Mesmo dia da semana passada": seasonal_naive,
    "Média do dia da semana (4 semanas)": weekday_average,
    "Regressão (tendência + dia da semana)": ridge_calendar,
}


def rolling_backtest(
    series: pd.Series,
    forecasters: dict[str, Forecaster] | None = None,
    horizon: int = 7,
    n_folds: int = 4,
    min_train: int = 28,
) -> pd.DataFrame:
    """Simula previsões semanais: a cada fold o método só vê o passado e prevê `horizon` dias."""
    forecasters = forecasters or FORECASTERS
    n = len(series)
    if n - horizon * n_folds < min_train:
        raise ValueError(
            f"Histórico curto: são necessários pelo menos {min_train + horizon * n_folds} dias "
            f"para testar {n_folds} semanas (há {n})."
        )
    frames = []
    for fold in range(n_folds):
        cut = n - (n_folds - fold) * horizon
        train, future = series.iloc[:cut], series.index[cut : cut + horizon]
        frame = pd.DataFrame({"real": series.iloc[cut : cut + horizon].to_numpy()}, index=future)
        for name, forecaster in forecasters.items():
            frame[name] = forecaster(train, future).to_numpy()
        frame["fold"] = fold
        frames.append(frame)
    return pd.concat(frames)


def score_forecasts(backtest: pd.DataFrame) -> pd.DataFrame:
    """MAE, RMSE e MAPE de cada método, do melhor para o pior (menor MAE)."""
    real = backtest["real"]
    rows = []
    for name in [c for c in backtest.columns if c not in ("real", "fold")]:
        error = backtest[name] - real
        mape = (error.abs() / real.where(real > 0)).mean() * 100
        rows.append(
            {
                "modelo": name,
                "MAE": error.abs().mean(),
                "RMSE": float(np.sqrt((error**2).mean())),
                "MAPE (%)": mape,
            }
        )
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def forecast_future(
    series: pd.Series, forecaster: Forecaster, horizon: int = 14, sigma: float = 0.0, z: float = 1.28
) -> pd.DataFrame:
    """Previsão para os próximos dias, com faixa aproximada (±z desvios do erro do backtest)."""
    future = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    point = forecaster(series, future)
    return pd.DataFrame(
        {
            "previsão": point,
            "mínimo provável": (point - z * sigma).clip(lower=0),
            "máximo provável": point + z * sigma,
        }
    )
