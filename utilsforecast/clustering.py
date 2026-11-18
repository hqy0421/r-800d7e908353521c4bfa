"""Share one fitted model across similar time series.

This module provides :class:`SimilarSeriesForecaster`, a predictor that learns a
standardization and a clustering from one static numeric feature per series
during :meth:`~SimilarSeriesForecaster.fit`, fits one independent model on each
cluster's real history and routes every series to the model of its cluster at
prediction time.

The clustering and the standardization are fully learned on ``fit`` and frozen
afterwards: at prediction time (including for unseen ids) the saved feature
order, center and scale are reused and the already fitted clusterer assigns the
cluster, never re-fitting and never re-clustering existing series.
"""

__all__ = [
    "Factory",
    "KMeans",
    "SimilarSeriesForecaster",
    "StandardScaler",
]

import pickle
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .compat import POLARS_INSTALLED, DataFrame, pl, pl_DataFrame

# ---------------------------------------------------------------------------
# Declarative callable specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Factory:
    """Declarative, serializable specification of a zero-argument callable.

    ``Factory("package.module", "ClassName", kwarg=value)`` stores the import
    path and the keyword arguments instead of the callable itself, so any
    object built through a factory can be reconstructed in a fresh process
    without relying on cloudpickle.

    Args:
        module (str): Importable module, e.g. ``"utilsforecast.clustering"``.
        attr (str): Name of the callable inside ``module``.
        kwargs (dict): Keyword arguments forwarded to the callable.
    """

    module: str
    attr: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_spec(cls, spec: "FactorySpec") -> "Factory":
        """Coerce a callable, dict, tuple or :class:`Factory` into a factory."""
        if isinstance(spec, Factory):
            return spec
        if isinstance(spec, dict):
            try:
                module = spec["module"]
                attr = spec["attr"]
            except KeyError as exc:
                raise ValueError(
                    "Factory dicts must contain 'module' and 'attr' keys."
                ) from exc
            return cls(
                module=module, attr=attr, kwargs=dict(spec.get("kwargs", {}))
            )
        if isinstance(spec, tuple):
            if len(spec) == 2:
                module, attr = spec
                return cls(module=module, attr=attr)
            if len(spec) == 3:
                module, attr, kwargs = spec
                return cls(module=module, attr=attr, kwargs=dict(kwargs))
            raise ValueError(
                "Factory tuples must have the form (module, attr) or "
                "(module, attr, kwargs)."
            )
        if callable(spec):
            resolved = _resolve_callable(spec)
            if resolved is None:
                raise ValueError(
                    f"Callable {spec!r} cannot be serialized declaratively. "
                    "Pass a Factory, a (module, attr[, kwargs]) tuple or a "
                    "{'module', 'attr', 'kwargs'} dict instead, e.g. "
                    "Factory('your_pkg.your_module', 'YourModel')."
                )
            module, attr = resolved
            return cls(module=module, attr=attr)
        raise TypeError(
            "Expected a Factory, callable, dict or tuple, got "
            f"{type(spec).__name__}."
        )

    def resolve(self) -> Callable[..., Any]:
        """Import and return the described callable."""
        try:
            module = import_module(self.module)
        except ImportError as exc:
            raise ImportError(
                f"Could not import factory module {self.module!r}: {exc}"
            ) from exc
        try:
            obj = getattr(module, self.attr)
        except AttributeError as exc:
            raise AttributeError(
                f"Module {self.module!r} has no attribute {self.attr!r}."
            ) from exc
        if not callable(obj):
            raise TypeError(
                f"{self.module}.{self.attr} resolved to a non-callable object."
            )
        return obj

    def create(self) -> Any:
        """Instantiate the described object with the stored keyword args."""
        return self.resolve()(**self.kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Return the plain dict representation used when serializing."""
        return {
            "module": self.module,
            "attr": self.attr,
            "kwargs": dict(self.kwargs),
        }


FactorySpec = Union[
    Factory, Callable[..., Any], Dict[str, Any], Tuple[str, ...]
]


def _resolve_callable(obj: Callable[..., Any]) -> Optional[Tuple[str, str]]:
    """Return (module, attr) if ``obj`` is reachable by name, else ``None``."""
    module_name = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if module_name is None or qualname is None or "<" in qualname:
        return None
    try:
        module = import_module(module_name)
    except ImportError:
        return None
    target: Any = module
    for part in qualname.split("."):
        target = getattr(target, part, None)
        if target is None:
            return None
    if target is not obj:
        return None
    return module_name, qualname


# ---------------------------------------------------------------------------
# Standardization and clustering (numpy-only)
# ---------------------------------------------------------------------------


class StandardScaler:
    """Per-column standardization learned once and frozen after ``fit``."""

    def __init__(self) -> None:
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0

    def fit(self, x: np.ndarray) -> "StandardScaler":
        x = _as_float_matrix(x)
        self.center_ = x.mean(axis=0)
        scale = x.std(axis=0)
        # constant columns would otherwise produce division by zero
        self.scale_ = np.where(scale > 0.0, scale, 1.0)
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise ValueError("StandardScaler must be fitted before transform.")
        x = _as_float_matrix(x, n_features=self.n_features_in_)
        return (x - self.center_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


class KMeans:
    """Minimal deterministic k-means implemented with numpy.

    Uses k-means++-style initialization and Lloyd's algorithm. The fitted
    centroids are the only state kept, so :meth:`predict` assigns every series
    (old or new) identically after serialization.

    Args:
        n_clusters (int): Number of clusters, must be positive.
        max_iter (int): Maximum Lloyd iterations.
        n_init (int): Number of independent initializations; the one with the
            lowest inertia is kept.
        random_state (int): Seed making every result reproducible.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 100,
        n_init: int = 10,
        random_state: int = 0,
    ) -> None:
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("`n_clusters` must be a positive integer.")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("`max_iter` must be a positive integer.")
        if not isinstance(n_init, int) or n_init <= 0:
            raise ValueError("`n_init` must be a positive integer.")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_centers_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_features_in_: int = 0

    @staticmethod
    def _sq_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
        # (n, k) squared euclidean distances; clamp tiny negatives produced
        # by floating-point cancellation
        distances = (
            (x * x).sum(axis=1, keepdims=True)
            - 2.0 * x @ centers.T
            + (centers * centers).sum(axis=1)
        )
        return np.maximum(distances, 0.0)

    def _init_centers(
        self, x: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        n = x.shape[0]
        k = min(self.n_clusters, n)
        first = int(rng.integers(n))
        centers = [x[first].copy()]
        closest = self._sq_distances(x, centers[0][None, :]).ravel()
        for _ in range(1, k):
            total = closest.sum()
            if total <= 0.0:
                idx = int(rng.integers(n))
            else:
                probs = closest / total
                idx = int(rng.choice(n, p=probs))
            centers.append(x[idx].copy())
            new_dist = self._sq_distances(x, centers[-1][None, :]).ravel()
            closest = np.minimum(closest, new_dist)
        return np.vstack(centers)

    def _lloyd(
        self, x: np.ndarray, centers: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        n = x.shape[0]
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(self.max_iter):
            dists = self._sq_distances(x, centers)
            new_labels = dists.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            new_centers = centers.copy()
            for j in range(centers.shape[0]):
                members = x[labels == j]
                if members.shape[0] == 0:
                    # reseed the empty cluster on the farthest current point
                    farthest = int(dists.min(axis=1).argmax())
                    new_centers[j] = x[farthest]
                else:
                    new_centers[j] = members.mean(axis=0)
            if np.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers
        dists = self._sq_distances(x, centers)
        labels = dists.argmin(axis=1)
        inertia = float(dists[np.arange(n), labels].sum())
        return labels, centers, inertia

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        x = _as_float_matrix(x)
        n = x.shape[0]
        if self.n_clusters > n:
            raise ValueError(
                f"`n_clusters` ({self.n_clusters}) cannot be larger than the "
                f"number of series ({n})."
            )
        rng = np.random.default_rng(self.random_state)
        best_labels: Optional[np.ndarray] = None
        best_centers: Optional[np.ndarray] = None
        best_inertia = np.inf
        for _ in range(self.n_init):
            init_centers = self._init_centers(x, rng)
            labels, centers, inertia = self._lloyd(x, init_centers)
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = centers
        assert best_labels is not None and best_centers is not None
        self.cluster_centers_ = best_centers
        self.inertia_ = best_inertia
        self.n_features_in_ = x.shape[1]
        return best_labels

    def fit(self, x: np.ndarray) -> "KMeans":
        self.fit_predict(x)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise ValueError("KMeans must be fitted before predict.")
        x = _as_float_matrix(x, n_features=self.n_features_in_)
        return self._sq_distances(x, self.cluster_centers_).argmin(axis=1)


def _as_float_matrix(
    x: np.ndarray, n_features: Optional[int] = None
) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got {arr.ndim}D.")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"Expected a numeric array, got dtype {arr.dtype!r}.")
    if n_features is not None and arr.shape[1] != n_features:
        raise ValueError(
            f"Expected {n_features} features, got {arr.shape[1]}."
        )
    return arr.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Shared-model forecaster
# ---------------------------------------------------------------------------

_SAVE_MAGIC = b"UFSIMILAR"
_SAVE_VERSION = 1


class SimilarSeriesForecaster:
    """Fit one model per cluster of similar series and route by cluster.

    Args:
        model_factory: Zero-argument callable (or declarative
            :class:`Factory` / dict / tuple) that builds a fresh model for each
            cluster. Each model must implement
            ``fit(history_df)`` and
            ``predict(history_df, future_df) -> forecast_df`` where every
            frame uses the same long-format column names as the forecaster.
        n_clusters (int): Number of clusters for the default :class:`KMeans`.
        feature_cols (list of str, optional): Numeric per-id static feature
            columns. When omitted every non-id column of the fit feature frame
            is used, in that frame's order.
        id_col (str): Name of the series identifier column.
        time_col (str): Name of the time column.
        target_col (str): Name of the target column.
        output_col (str): Name of the prediction column returned by the models.
        scaler_factory: Declarative factory for the feature standardizer.
        clusterer_factory: Declarative factory for the clusterer, which must
            expose ``fit_predict(x)`` and ``predict(x)``.
        random_state (int): Seed forwarded to the default clusterer factory.
    """

    def __init__(
        self,
        model_factory: FactorySpec,
        n_clusters: int = 8,
        feature_cols: Optional[Sequence[str]] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        output_col: str = "y_hat",
        scaler_factory: Optional[FactorySpec] = None,
        clusterer_factory: Optional[FactorySpec] = None,
        random_state: int = 0,
    ) -> None:
        self.model_factory = Factory.from_spec(model_factory)
        self.n_clusters = n_clusters
        self.feature_cols = (
            list(feature_cols) if feature_cols is not None else None
        )
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.output_col = output_col
        self.scaler_factory = (
            Factory.from_spec(scaler_factory)
            if scaler_factory is not None
            else Factory("utilsforecast.clustering", "StandardScaler")
        )
        if clusterer_factory is None:
            clusterer_factory = (
                "utilsforecast.clustering",
                "KMeans",
                {
                    "n_clusters": n_clusters,
                    "random_state": random_state,
                },
            )
        self.clusterer_factory = Factory.from_spec(clusterer_factory)
        self.random_state = random_state
        # fitted state, populated only after a fully successful fit
        self.feature_names_: Optional[List[str]] = None
        self.scaler_: Optional[StandardScaler] = None
        self.clusterer_: Optional[Any] = None
        self.cluster_assignment_: Optional[pd.DataFrame] = None
        self.models_: Optional[Dict[int, Any]] = None

    # -- internal helpers ---------------------------------------------------

    def _check_fitted(self) -> None:
        if self.models_ is None or self.cluster_assignment_ is None:
            raise ValueError(
                "SimilarSeriesForecaster is not fitted. Call `fit` first."
            )

    def _check_backend(self, *frames: DataFrame) -> None:
        if all(isinstance(frame, pd.DataFrame) for frame in frames):
            return
        if POLARS_INSTALLED and all(
            isinstance(frame, pl_DataFrame) for frame in frames
        ):
            return
        backends = sorted(type(frame).__name__ for frame in frames)
        raise ValueError(
            "history_df, future_df and features_df must all be pandas or all "
            f"be polars dataframes, got {backends}."
        )

    def _check_required_cols(
        self, df: DataFrame, cols: Sequence[str], where: str
    ) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required column(s) {missing} in {where}; "
                f"available columns: {list(df.columns)}."
            )

    def _has_duplicate_keys(
        self, df: DataFrame, keys: Sequence[str]
    ) -> bool:
        if isinstance(df, pd.DataFrame):
            return bool(df.duplicated(subset=list(keys)).any())
        # group_by-based count works on every supported polars release
        return bool(df.group_by(list(keys)).len().height < df.height)

    def _has_nulls(self, df: DataFrame, col: str) -> bool:
        if isinstance(df, pd.DataFrame):
            return bool(df[col].isna().any())
        return bool(df[col].null_count() > 0)

    def _check_numeric_feature(self, df: DataFrame, col: str) -> None:
        if isinstance(df, pd.DataFrame):
            is_numeric = pd.api.types.is_numeric_dtype(df[col].dtype)
        else:
            is_numeric = df[col].dtype.is_numeric()
        if not is_numeric:
            raise ValueError(
                f"Feature column {col!r} must have a numeric dtype, got "
                f"{df[col].dtype!r}."
            )

    def _feature_matrix(
        self, features_df: DataFrame, names: Sequence[str]
    ) -> np.ndarray:
        for col in names:
            self._check_numeric_feature(features_df, col)
        if isinstance(features_df, pd.DataFrame):
            x = features_df.loc[:, list(names)].to_numpy(dtype=np.float64)
            bad_id = features_df[self.id_col]
        else:
            x = features_df.select(list(names)).to_numpy().astype(np.float64)
            bad_id = features_df[self.id_col]
        if not np.isfinite(x).all():
            bad = np.argwhere(~np.isfinite(x))[0]
            row, feat = int(bad[0]), int(bad[1])
            raise ValueError(
                "Feature values must be finite; found null/NaN/inf in column "
                f"{names[feat]!r} for {self.id_col}={bad_id[row]}."
            )
        return x

    def _unique_ids(self, df: DataFrame) -> np.ndarray:
        if isinstance(df, pd.DataFrame):
            return df[self.id_col].drop_duplicates().to_numpy()
        return df[self.id_col].unique().to_numpy()

    def _ordered_ids(self, df: DataFrame) -> np.ndarray:
        """First occurrence order, stable across pandas and polars."""
        if isinstance(df, pd.DataFrame):
            return df[self.id_col].drop_duplicates(keep="first").to_numpy()
        return df[self.id_col].unique(maintain_order=True).to_numpy()

    def _filter_ids(
        self, df: DataFrame, ids: np.ndarray
    ) -> DataFrame:
        id_list = ids.tolist()
        if isinstance(df, pd.DataFrame):
            return df[df[self.id_col].isin(id_list)]
        return df.filter(pl.col(self.id_col).is_in(id_list))

    # -- fit ----------------------------------------------------------------

    def fit(
        self,
        history_df: DataFrame,
        features_df: DataFrame,
    ) -> "SimilarSeriesForecaster":
        """Learn standardization and clustering, then fit one model per cluster.

        Args:
            history_df: Long-format history with id, time and target columns.
                Only real observations of each cluster are handed to the
                cluster's model.
            features_df: One row per id containing the static numeric features.

        Returns:
            SimilarSeriesForecaster: the fitted forecaster. The previous fitted
            state is only replaced once every step succeeds.
        """
        self._check_backend(history_df, features_df)
        self._check_required_cols(
            history_df,
            [self.id_col, self.time_col, self.target_col],
            "history_df",
        )
        self._check_required_cols(features_df, [self.id_col], "features_df")
        if self._has_duplicate_keys(
            history_df, [self.id_col, self.time_col]
        ):
            raise ValueError(
                "history_df contains duplicate (id, time) pairs; each series "
                "must have at most one observation per timestamp."
            )
        if self._has_duplicate_keys(features_df, [self.id_col]):
            raise ValueError(
                f"features_df contains duplicate {self.id_col!r} values; each "
                "series must have exactly one feature row."
            )
        if self._has_nulls(features_df, self.id_col):
            raise ValueError(f"The id column {self.id_col!r} contains nulls.")
        if self._has_nulls(history_df, self.id_col):
            raise ValueError(f"The id column {self.id_col!r} contains nulls.")

        feature_names = (
            list(self.feature_cols)
            if self.feature_cols is not None
            else [c for c in features_df.columns if c != self.id_col]
        )
        if not feature_names:
            raise ValueError(
                "At least one numeric feature column is required; got none."
            )
        if len(set(feature_names)) != len(feature_names):
            raise ValueError(f"Duplicate feature column names: {feature_names}.")
        self._check_required_cols(features_df, feature_names, "features_df")

        hist_ids = set(self._unique_ids(history_df).tolist())
        # keep first-occurrence order so the feature matrix rows are defined
        ordered_ids = self._ordered_ids(features_df)
        feat_ids = set(ordered_ids.tolist())
        missing_history = sorted(feat_ids - hist_ids)
        if missing_history:
            raise ValueError(
                "Every id in features_df must have history; missing history "
                f"for {missing_history[:5]}"
                + (" ..." if len(missing_history) > 5 else "")
                + "."
            )
        extra_history = sorted(hist_ids - feat_ids)
        if extra_history:
            raise ValueError(
                "Every id in history_df must have a feature row; missing "
                f"features for {extra_history[:5]}"
                + (" ..." if len(extra_history) > 5 else "")
                + "."
            )

        x = self._feature_matrix(features_df, feature_names)

        scaler = self.scaler_factory.create()
        x_scaled = scaler.fit_transform(x)
        clusterer = self.clusterer_factory.create()
        labels = np.asarray(clusterer.fit_predict(x_scaled))
        if labels.shape != (ordered_ids.shape[0],):
            raise ValueError(
                "clusterer.fit_predict must return one label per series, got "
                f"shape {labels.shape} for {ordered_ids.shape[0]} series."
            )
        unique_labels = np.unique(labels)
        n_labels = unique_labels.shape[0]
        if not np.array_equal(unique_labels, np.arange(n_labels)):
            raise ValueError(
                "Cluster labels must be contiguous integers starting at 0, "
                f"got {unique_labels.tolist()}."
            )

        assignment = pd.DataFrame(
            {
                self.id_col: ordered_ids,
                "cluster": labels.astype(np.int64),
            }
        )

        models: Dict[int, Any] = {}
        for cluster_id in unique_labels.tolist():
            member_ids = ordered_ids[labels == cluster_id]
            cluster_history = self._filter_ids(history_df, member_ids)
            try:
                model = self.model_factory.create()
                model.fit(cluster_history)
            except Exception as exc:
                raise RuntimeError(
                    f"Model fitting failed for cluster {cluster_id} "
                    f"({member_ids.tolist()}): {exc}"
                ) from exc
            models[cluster_id] = model

        # commit only after every cluster succeeded; a failed fit leaves the
        # previous valid fit untouched
        self.feature_names_ = feature_names
        self.scaler_ = scaler
        self.clusterer_ = clusterer
        self.cluster_assignment_ = assignment
        self.models_ = models
        return self

    # -- predict ------------------------------------------------------------

    def predict(
        self,
        history_df: DataFrame,
        future_df: DataFrame,
        features_df: DataFrame,
    ) -> DataFrame:
        """Predict every future row using the fitted model of its cluster.

        The clustering learned during :meth:`fit` is reused as-is: ids present
        at fit time keep their saved cluster, unseen ids are assigned with the
        fitted scaler and clusterer. Nothing is re-fitted or re-clustered.

        Args:
            history_df: History available at prediction time.
            future_df: Rows to predict, containing id and time columns.
            features_df: One row per predicted id with the same feature
                columns used at fit time (column order may differ).

        Returns:
            pandas or polars DataFrame matching the backend of the inputs with
            the id, time and prediction columns, aligned one-to-one with
            ``future_df``'s (id, time) keys.
        """
        self._check_fitted()
        assert self.feature_names_ is not None and self.scaler_ is not None
        assert self.clusterer_ is not None and self.models_ is not None
        self._check_backend(history_df, future_df, features_df)
        self._check_required_cols(
            history_df,
            [self.id_col, self.time_col, self.target_col],
            "history_df",
        )
        self._check_required_cols(
            future_df, [self.id_col, self.time_col], "future_df"
        )
        self._check_required_cols(
            features_df, [self.id_col] + self.feature_names_, "features_df"
        )
        if self._has_duplicate_keys(
            history_df, [self.id_col, self.time_col]
        ):
            raise ValueError(
                "history_df contains duplicate (id, time) pairs."
            )
        if self._has_duplicate_keys(
            future_df, [self.id_col, self.time_col]
        ):
            raise ValueError(
                "future_df contains duplicate (id, time) pairs; each future "
                "key must be unique."
            )
        if self._has_duplicate_keys(features_df, [self.id_col]):
            raise ValueError(
                f"features_df contains duplicate {self.id_col!r} values."
            )

        future_ids = set(self._unique_ids(future_df).tolist())
        feature_ids_list = self._ordered_ids(features_df).tolist()
        feature_ids = set(feature_ids_list)
        history_ids = set(self._unique_ids(history_df).tolist())
        if feature_ids != future_ids:
            missing = sorted(future_ids - feature_ids)
            extra = sorted(feature_ids - future_ids)
            parts = []
            if missing:
                parts.append(f"missing feature rows for {missing[:5]}")
            if extra:
                parts.append(f"unexpected feature rows for {extra[:5]}")
            raise ValueError(
                "features_df ids must match future_df ids exactly; "
                + "; ".join(parts)
                + "."
            )
        if not history_ids.issubset(future_ids):
            raise ValueError(
                "history_df contains ids without future rows: "
                f"{sorted(history_ids - future_ids)[:5]}."
            )

        # align by name, ignoring the column order of the input frame
        x = self._feature_matrix(features_df, self.feature_names_)
        x_scaled = self.scaler_.transform(x)

        assert self.cluster_assignment_ is not None
        saved_map = dict(
            zip(
                self.cluster_assignment_[self.id_col].tolist(),
                self.cluster_assignment_["cluster"].tolist(),
            )
        )
        new_mask = np.array(
            [uid not in saved_map for uid in feature_ids_list]
        )
        labels = np.empty(len(feature_ids_list), dtype=np.int64)
        for i, uid in enumerate(feature_ids_list):
            if uid in saved_map:
                # fitted ids keep their frozen assignment, no re-clustering
                labels[i] = saved_map[uid]
        if new_mask.any():
            # unseen ids are scored with the frozen scaler and clusterer only
            labels[new_mask] = np.asarray(
                self.clusterer_.predict(x_scaled[new_mask])
            )
        id_to_cluster = dict(zip(feature_ids_list, labels.tolist()))

        predictions: List[DataFrame] = []
        for cluster_id in sorted(set(labels.tolist())):
            try:
                model = self.models_[cluster_id]
            except KeyError as exc:
                raise ValueError(
                    f"Cluster {cluster_id} has no fitted model."
                ) from exc
            member_ids = np.asarray(
                [
                    uid
                    for uid, c in id_to_cluster.items()
                    if c == cluster_id
                ]
            )
            cluster_history = self._filter_ids(history_df, member_ids)
            cluster_future = self._filter_ids(future_df, member_ids)
            try:
                pred = model.predict(cluster_history, cluster_future)
            except Exception as exc:
                raise RuntimeError(
                    f"Model prediction failed for cluster {cluster_id}: {exc}"
                ) from exc
            self._validate_prediction(pred, cluster_future, cluster_id)
            predictions.append(pred)

        return self._assemble(predictions, future_df)

    def _validate_prediction(
        self,
        pred: DataFrame,
        cluster_future: DataFrame,
        cluster_id: int,
    ) -> None:
        if not isinstance(pred, type(cluster_future)):
            raise ValueError(
                f"Model of cluster {cluster_id} returned "
                f"{type(pred).__name__} but a different backend was provided."
            )
        required = [self.id_col, self.time_col, self.output_col]
        missing = [c for c in required if c not in pred.columns]
        if missing:
            raise ValueError(
                f"Model of cluster {cluster_id} is missing output column(s) "
                f"{missing}; expected {required}."
            )
        if self._has_duplicate_keys(pred, [self.id_col, self.time_col]):
            raise ValueError(
                f"Model of cluster {cluster_id} returned duplicate (id, time) "
                "keys."
            )
        if self._has_nulls(pred, self.output_col):
            raise ValueError(
                f"Model of cluster {cluster_id} returned null predictions."
            )
        key_cols = [self.id_col, self.time_col]
        if isinstance(pred, pd.DataFrame):
            pred_keys = set(
                map(
                    tuple,
                    pred[key_cols].itertuples(index=False, name=None),
                )
            )
            future_keys = set(
                map(
                    tuple,
                    cluster_future[key_cols].itertuples(
                        index=False, name=None
                    ),
                )
            )
        else:
            pred_keys = set(
                map(tuple, pred.select(key_cols).iter_rows())
            )
            future_keys = set(
                map(
                    tuple,
                    cluster_future.select(key_cols).iter_rows(),
                )
            )
        if pred_keys != future_keys:
            missing_keys = sorted(future_keys - pred_keys)[:3]
            extra_keys = sorted(pred_keys - future_keys)[:3]
            raise ValueError(
                f"Model of cluster {cluster_id} did not return exactly the "
                f"future keys; missing={missing_keys}, unexpected={extra_keys}."
            )

    def _assemble(
        self,
        predictions: Sequence[DataFrame],
        future_df: DataFrame,
    ) -> DataFrame:
        assert predictions, "at least one cluster must produce predictions"
        if isinstance(future_df, pd.DataFrame):
            out = pd.concat(predictions, ignore_index=True)
            key_cols = [self.id_col, self.time_col]
            aligned = future_df[key_cols].copy().merge(
                out,
                on=key_cols,
                how="left",
                validate="one_to_one",
            )
            if aligned[self.output_col].isna().any():
                raise ValueError(
                    "Could not align every future (id, time) key to a "
                    "prediction; refusing to deliver partial results."
                )
            return aligned
        out = pl.concat(predictions, how="vertical_relaxed")
        key_cols = [self.id_col, self.time_col]
        aligned = future_df.select(key_cols).join(
            out,
            on=key_cols,
            how="left",
        )
        if aligned.height != future_df.height:
            raise ValueError(
                "Could not align every future (id, time) key to a unique "
                "prediction; refusing to deliver partial results."
            )
        if aligned[self.output_col].null_count() > 0:
            raise ValueError(
                "Could not align every future (id, time) key to a prediction; "
                "refusing to deliver partial results."
            )
        # restore the exact row order of future_df
        order_name = "__uf_row_order__"
        indexed = future_df.select(key_cols).with_row_index(order_name)
        aligned = indexed.join(
            aligned, on=key_cols, how="inner"
        ).sort(order_name)
        return aligned.drop(order_name)

    # -- cluster table ------------------------------------------------------

    def cluster_assignment(self) -> pd.DataFrame:
        """Return a copy of the id-to-cluster table learned during fit."""
        self._check_fitted()
        assert self.cluster_assignment_ is not None
        return self.cluster_assignment_.copy()

    # -- (de)serialization --------------------------------------------------

    def _serializable_state(self) -> Dict[str, Any]:
        self._check_fitted()
        return {
            "magic": _SAVE_MAGIC,
            "version": _SAVE_VERSION,
            "params": {
                "model_factory": self.model_factory.to_dict(),
                "n_clusters": self.n_clusters,
                "feature_cols": self.feature_cols,
                "id_col": self.id_col,
                "time_col": self.time_col,
                "target_col": self.target_col,
                "output_col": self.output_col,
                "scaler_factory": self.scaler_factory.to_dict(),
                "clusterer_factory": self.clusterer_factory.to_dict(),
                "random_state": self.random_state,
            },
            "fit": {
                "feature_names_": self.feature_names_,
                "scaler_": self.scaler_,
                "clusterer_": self.clusterer_,
                "cluster_assignment_": self.cluster_assignment_,
                "models_": self.models_,
            },
        }

    def save(self, path: str) -> None:
        """Persist the fitted forecaster to ``path``.

        Factories are stored as declarative import specs while fitted objects
        are pickled, so a reloaded forecaster keeps the same clusters and
        predictions without re-fitting.
        """
        state = self._serializable_state()
        with open(path, "wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "SimilarSeriesForecaster":
        """Load a forecaster previously written with :meth:`save`."""
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        if not isinstance(state, dict) or state.get("magic") != _SAVE_MAGIC:
            raise ValueError(
                f"File {path!r} is not a SimilarSeriesForecaster save file."
            )
        if state.get("version") != _SAVE_VERSION:
            raise ValueError(
                f"Unsupported save version {state.get('version')!r}; expected "
                f"{_SAVE_VERSION}."
            )
        params = dict(state["params"])
        params["model_factory"] = Factory.from_spec(params["model_factory"])
        params["scaler_factory"] = Factory.from_spec(params["scaler_factory"])
        params["clusterer_factory"] = Factory.from_spec(
            params["clusterer_factory"]
        )
        forecaster = cls(**params)
        fit_state = state["fit"]
        forecaster.feature_names_ = fit_state["feature_names_"]
        forecaster.scaler_ = fit_state["scaler_"]
        forecaster.clusterer_ = fit_state["clusterer_"]
        forecaster.cluster_assignment_ = fit_state["cluster_assignment_"]
        forecaster.models_ = fit_state["models_"]
        return forecaster
