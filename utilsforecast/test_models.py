"""Reference models used to exercise :mod:`utilsforecast.clustering`.

These deliberately tiny models implement the two-method protocol expected by
:class:`~utilsforecast.clustering.SimilarSeriesForecaster` (``fit(history)``
and ``predict(history, future)``) and live inside the package so that
declarative :class:`~utilsforecast.clustering.Factory` specs can resolve them
in a freshly started, isolated interpreter.
"""

__all__ = [
    "BadFitModel",
    "BadPredictModel",
    "ConstantMeanModel",
    "WrongKeyModel",
]

from typing import Any, List

import pandas as pd

from .compat import pl


class ConstantMeanModel:
    """Predict every future row with the mean of the cluster's real history."""

    def __init__(self, target_col: str = "y", output_col: str = "y_hat") -> None:
        self.target_col = target_col
        self.output_col = output_col
        self.mean_: float = 0.0
        self.fit_ids: List[Any] = []
        self.predict_histories: List[Any] = []
        self.predict_futures: List[Any] = []

    def fit(self, history_df: Any) -> "ConstantMeanModel":
        if isinstance(history_df, pd.DataFrame):
            ids = history_df.iloc[:, 0].unique().tolist()
            self.mean_ = float(history_df[self.target_col].mean())
        else:
            ids = history_df[history_df.columns[0]].unique().to_list()
            self.mean_ = float(
                history_df.select(pl.col(self.target_col).mean()).item()
            )
        self.fit_ids = ids
        return self

    def predict(self, history_df: Any, future_df: Any) -> Any:
        # record what the forecaster actually handed over; tests inspect this
        self.predict_histories.append(history_df)
        self.predict_futures.append(future_df)
        if isinstance(future_df, pd.DataFrame):
            out = future_df.iloc[:, :2].copy()
            out[self.output_col] = self.mean_
            return out
        return future_df.select(future_df.columns[:2]).with_columns(
            pl.lit(self.mean_).alias(self.output_col)
        )


class BadFitModel:
    """Model whose ``fit`` always fails, used to test atomic fit behavior."""

    def fit(self, _history_df: Any) -> "BadFitModel":
        raise RuntimeError("boom during fit")

    def predict(self, _history_df: Any, _future_df: Any) -> Any:
        raise AssertionError("predict must never be called on a failed fit")


class BadPredictModel:
    """Model whose ``predict`` always fails, used to test predict errors."""

    def fit(self, _history_df: Any) -> "BadPredictModel":
        return self

    def predict(self, _history_df: Any, _future_df: Any) -> Any:
        raise RuntimeError("boom during predict")


class WrongKeyModel:
    """Model that drops one future key, used to test partial-result failure."""

    def __init__(self, output_col: str = "y_hat") -> None:
        self.output_col = output_col

    def fit(self, _history_df: Any) -> "WrongKeyModel":
        return self

    def predict(self, history_df: Any, future_df: Any) -> Any:
        del history_df
        if isinstance(future_df, pd.DataFrame):
            out = future_df.iloc[:-1, :2].copy()
            out[self.output_col] = 1.0
            return out
        out = future_df.select(future_df.columns[:2]).head(future_df.height - 1)
        return out.with_columns(pl.lit(1.0).alias(self.output_col))
