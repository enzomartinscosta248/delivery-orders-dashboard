import numpy as np
import pandas as pd

from delivery_dashboard.forecast import (
    FORECASTERS,
    daily_series,
    forecast_future,
    rolling_backtest,
    score_forecasts,
)


def make_series(n: int = 140, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-05", periods=n, freq="D")
    weekly = np.array([0.85, 0.90, 0.95, 1.00, 1.20, 1.30, 1.15])[index.weekday]
    values = 150 * weekly * (1 + 0.2 * np.arange(n) / n) * rng.lognormal(0, 0.06, n)
    return pd.Series(values.round(), index=index, name="pedidos reais")


def test_backtest_shape_alignment_and_scores():
    series = make_series()
    backtest = rolling_backtest(series, horizon=7, n_folds=4)
    assert len(backtest) == 28 and set(backtest["fold"]) == {0, 1, 2, 3}
    assert list(backtest.columns) == ["real", *FORECASTERS, "fold"]
    assert backtest.index[-1] == series.index[-1]
    assert (backtest["real"].to_numpy() == series.iloc[-28:].to_numpy()).all()

    scores = score_forecasts(backtest)
    assert len(scores) == 3 and scores["MAE"].is_monotonic_increasing
    assert {"modelo", "MAE", "RMSE", "MAPE (%)"} <= set(scores.columns)


def test_calendar_regression_beats_naive_on_seasonal_series_with_trend():
    scores = score_forecasts(rolling_backtest(make_series(), horizon=7, n_folds=6)).set_index("modelo")
    regression = scores.loc["Regressão (tendência + dia da semana)", "MAE"]
    naive = scores.loc["Mesmo dia da semana passada", "MAE"]
    assert regression < naive


def test_forecast_future_shape_bounds_and_non_negative():
    series = make_series()
    future = forecast_future(
        series, FORECASTERS["Regressão (tendência + dia da semana)"], horizon=14, sigma=10.0
    )
    assert len(future) == 14
    assert future.index[0] == series.index[-1] + pd.Timedelta(days=1)
    assert (future["mínimo provável"] <= future["previsão"]).all()
    assert (future["previsão"] <= future["máximo provável"]).all()
    assert (future["mínimo provável"] >= 0).all()


def test_short_history_raises_and_missing_days_become_zero():
    try:
        rolling_backtest(make_series(n=40), n_folds=4)
    except ValueError as error:
        assert "Histórico curto" in str(error)
    else:
        raise AssertionError("era esperado ValueError")

    daily = pd.DataFrame({"day": ["2026-01-01", "2026-01-03"], "orders": [5, 7]})
    series = daily_series(daily)
    assert series.tolist() == [5.0, 0.0, 7.0] and series.index.freq is not None
