import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import polars as pl
import pytest

from utilsforecast.clustering import (
    Factory,
    KMeans,
    SimilarSeriesForecaster,
    StandardScaler,
)
from utilsforecast.compat import POLARS_INSTALLED
from utilsforecast.test_models import (
    BadFitModel,
    BadPredictModel,
    ConstantMeanModel,
    WrongKeyModel,
)

N_IDS = 6
H = 4
HIST_LEN = 10
F1 = np.array([0.0, 0.1, -0.1, 10.0, 9.9, 10.1])
F2 = np.array([5.0, 5.1, 4.9, -5.0, -4.9, -5.1])
GROUP_A = [0, 1, 2]
GROUP_B = [3, 4, 5]


def _build_history(engine="pandas"):
    rows = []
    for uid in range(N_IDS):
        for t in range(HIST_LEN):
            # distinct levels per cluster so means are clearly cluster-specific
            level = 100.0 if uid in GROUP_A else -100.0
            rows.append((uid, t, level + uid * 0.01 + t * 0.001))
    pdf = pd.DataFrame(rows, columns=["unique_id", "ds", "y"])
    if engine == "polars":
        return pl.from_pandas(pdf)
    return pdf


def _build_future(ids=None, engine="pandas"):
    if ids is None:
        ids = list(range(N_IDS))
    rows = [(uid, HIST_LEN + h) for uid in ids for h in range(H)]
    pdf = pd.DataFrame(rows, columns=["unique_id", "ds"])
    if engine == "polars":
        return pl.from_pandas(pdf)
    return pdf


def _build_features(ids=None, shuffled=False, engine="pandas"):
    if ids is None:
        ids = list(range(N_IDS))
    data = {
        "unique_id": ids,
        "f1": F1[ids].tolist(),
        "f2": F2[ids].tolist(),
    }
    pdf = pd.DataFrame(data)
    if shuffled:
        pdf = pdf[["unique_id", "f2", "f1"]]
        pdf = pdf.sample(frac=1.0, random_state=0).reset_index(drop=True)
    if engine == "polars":
        return pl.from_pandas(pdf)
    return pdf


def _forecaster(n_clusters=2, **kwargs):
    return SimilarSeriesForecaster(
        model_factory=Factory(
            "utilsforecast.test_models",
            "ConstantMeanModel",
        ),
        n_clusters=n_clusters,
        **kwargs,
    )


@pytest.fixture
def fitted():
    hist = _build_history()
    feats = _build_features()
    fc = _forecaster()
    fc.fit(hist, feats)
    return fc, hist, feats


backends = ["pandas"] + (["polars"] if POLARS_INSTALLED else [])


# ---------------------------------------------------------------------------
# fit / predict basics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", backends)
def test_fit_predict_shapes_and_alignment(engine):
    hist = _build_history(engine)
    feats = _build_features(engine=engine)
    future = _build_future(engine=engine)
    fc = _forecaster()
    fc.fit(hist, feats)
    pred = fc.predict(hist, future, feats)

    if engine == "pandas":
        assert isinstance(pred, pd.DataFrame)
        pd.testing.assert_frame_equal(
            pred[["unique_id", "ds"]],
            future[["unique_id", "ds"]],
            check_like=False,
        )
    else:
        assert isinstance(pred, pl.DataFrame)
        assert pred.select(["unique_id", "ds"]).equals(
            future.select(["unique_id", "ds"])
        )
    assert list(pred.columns) == ["unique_id", "ds", "y_hat"]
    assert pred.shape[0] == future.shape[0]
    n_unique = (
        pred["unique_id"].n_unique()
        if engine == "polars"
        else pred["unique_id"].nunique()
    )
    assert n_unique == N_IDS


def test_clusters_match_feature_similarity(fitted):
    fc, _, _ = fitted
    assignment = fc.cluster_assignment()
    assert list(assignment.columns) == ["unique_id", "cluster"]
    assert assignment.shape[0] == N_IDS
    assert set(assignment["cluster"].unique()) == {0, 1}
    cluster_a = set(
        assignment.loc[assignment["unique_id"].isin(GROUP_A), "cluster"]
    )
    cluster_b = set(
        assignment.loc[assignment["unique_id"].isin(GROUP_B), "cluster"]
    )
    assert len(cluster_a) == 1 and len(cluster_b) == 1
    assert cluster_a != cluster_b


def test_one_independent_model_per_cluster_real_history_only(fitted):
    fc, hist, _ = fitted
    assert set(fc.models_.keys()) == {0, 1}
    for cluster_id, model in fc.models_.items():
        members = set(
            fc.cluster_assignment_.loc[
                fc.cluster_assignment_["cluster"] == cluster_id, "unique_id"
            ]
        )
        assert set(model.fit_ids) == members
        seen = set(hist["unique_id"].unique())
        assert members.issubset(seen)
        # the constant equals the mean of that cluster's history only
        expected = hist.loc[
            hist["unique_id"].isin(members), "y"
        ].mean()
        assert model.mean_ == pytest.approx(expected)
    means = [m.mean_ for m in fc.models_.values()]
    assert means[0] != pytest.approx(means[1])


def test_predict_routes_each_id_to_its_own_cluster_model(fitted):
    fc, hist, feats = fitted
    future = _build_future()
    for model in fc.models_.values():
        model.predict_histories.clear()
        model.predict_futures.clear()
    fc.predict(hist, future, feats)
    for cluster_id, model in fc.models_.items():
        assert len(model.predict_futures) == 1
        routed = set(model.predict_futures[0]["unique_id"].unique())
        members = set(
            fc.cluster_assignment_.loc[
                fc.cluster_assignment_["cluster"] == cluster_id, "unique_id"
            ]
        )
        assert routed == members
        hist_routed = set(model.predict_histories[0]["unique_id"].unique())
        assert hist_routed == members


def test_predict_values_are_cluster_means(fitted):
    fc, hist, feats = fitted
    future = _build_future()
    pred = fc.predict(hist, future, feats)
    for uid in range(N_IDS):
        cluster_id = int(
            fc.cluster_assignment_.loc[
                fc.cluster_assignment_["unique_id"] == uid, "cluster"
            ].iloc[0]
        )
        expected = fc.models_[cluster_id].mean_
        got = pred.loc[pred["unique_id"] == uid, "y_hat"].to_numpy()
        assert np.allclose(got, expected)


def test_shuffled_feature_columns_align_by_name(fitted):
    fc, hist, _ = fitted
    feats_shuffled = _build_features(shuffled=True)
    future = _build_future()
    pred = fc.predict(hist, future, feats_shuffled)
    feats_clean = _build_features()
    pred_clean = fc.predict(hist, future, feats_clean)
    pd.testing.assert_frame_equal(
        pred.sort_values(["unique_id", "ds"]).reset_index(drop=True),
        pred_clean.sort_values(["unique_id", "ds"]).reset_index(drop=True),
    )
    # fit with a different column order must produce the same clusters
    fc2 = _forecaster()
    fc2.fit(hist, feats_shuffled)
    assert fc2.feature_names_ == ["f2", "f1"]
    a1 = fc.cluster_assignment().sort_values("unique_id").reset_index(drop=True)
    a2 = fc2.cluster_assignment().sort_values("unique_id").reset_index(drop=True)
    pd.testing.assert_series_equal(a1["cluster"], a2["cluster"], check_names=False)


def test_unseen_id_uses_frozen_scaler_and_clusterer_without_refit(fitted):
    fc, hist, feats = fitted
    new_id = N_IDS
    # halfway between the group A feature vertices -> same cluster as group A
    new_feats = pd.DataFrame(
        {
            "unique_id": list(range(N_IDS)) + [new_id],
            "f1": F1.tolist() + [0.05],
            "f2": F2.tolist() + [5.0],
        }
    )
    future = _build_future(ids=list(range(N_IDS)) + [new_id])
    fit_calls = []
    for model in fc.models_.values():
        original_fit = model.fit

        def counting_fit(*args, _orig=original_fit, _calls=fit_calls, **kwargs):
            _calls.append(1)
            return _orig(*args, **kwargs)

        model.fit = counting_fit
    saved_assignment = fc.cluster_assignment().copy()
    pred = fc.predict(hist, future, new_feats)
    assert fit_calls == []  # no re-fitting, no re-clustering of old ids
    pd.testing.assert_frame_equal(fc.cluster_assignment(), saved_assignment)
    cluster_a = int(
        fc.cluster_assignment_.loc[
            fc.cluster_assignment_["unique_id"].isin(GROUP_A), "cluster"
        ].iloc[0]
    )
    new_pred = pred.loc[pred["unique_id"] == new_id, "y_hat"].to_numpy()
    assert np.allclose(new_pred, fc.models_[cluster_a].mean_)


def test_predict_does_not_mutate_fitted_state(fitted):
    fc, hist, feats = fitted
    before = fc.cluster_assignment().copy()
    future = _build_future()
    fc.predict(hist, future, feats)
    fc.predict(hist, future, feats)
    pd.testing.assert_frame_equal(fc.cluster_assignment(), before)


def test_cluster_assignment_returns_copy(fitted):
    fc, _, _ = fitted
    table = fc.cluster_assignment()
    table["cluster"] = -999
    assert (fc.cluster_assignment_["cluster"] != -999).all()


# ---------------------------------------------------------------------------
# validation failures
# ---------------------------------------------------------------------------


def test_predict_before_fit_fails():
    fc = _forecaster()
    with pytest.raises(ValueError, match="not fitted"):
        fc.predict(_build_history(), _build_future(), _build_features())
    with pytest.raises(ValueError, match="not fitted"):
        fc.cluster_assignment()


def test_missing_feature_column_at_fit_fails():
    hist = _build_history()
    feats = _build_features().drop(columns=["f2"])
    fc = _forecaster(feature_cols=["f1", "f2"])
    with pytest.raises(ValueError, match="Missing required column"):
        fc.fit(hist, feats)


def test_missing_feature_column_at_predict_fails(fitted):
    fc, hist, _ = fitted
    feats = _build_features().drop(columns=["f2"])
    with pytest.raises(ValueError, match="Missing required column"):
        fc.predict(hist, _build_future(), feats)


def test_extra_feature_column_at_predict_is_ignored(fitted):
    fc, hist, feats = fitted
    extra = feats.copy()
    extra["f3"] = 1.0
    pred = fc.predict(hist, _build_future(), extra)
    assert list(pred.columns) == ["unique_id", "ds", "y_hat"]


def test_duplicate_id_feature_rows_fail():
    hist = _build_history()
    feats = _build_features()
    feats = pd.concat([feats, feats.iloc[[0]]], ignore_index=True)
    fc = _forecaster()
    with pytest.raises(ValueError, match="duplicate"):
        fc.fit(hist, feats)


def test_duplicate_history_keys_fail():
    hist = _build_history()
    dup = pd.concat([hist, hist.iloc[[0]]], ignore_index=True)
    fc = _forecaster()
    with pytest.raises(ValueError, match="duplicate"):
        fc.fit(dup, _build_features())


def test_duplicate_future_keys_fail(fitted):
    fc, hist, feats = fitted
    future = pd.concat([_build_future(), _build_future().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        fc.predict(hist, future, feats)


def test_feature_id_mismatch_at_fit_fails():
    hist = _build_history()
    feats = _build_features()
    fc = _forecaster()
    with pytest.raises(ValueError, match="Every id in history_df must have"):
        fc.fit(hist, feats[feats["unique_id"] != 0])
    with pytest.raises(ValueError, match="Every id in features_df must have"):
        fc.fit(hist[hist["unique_id"] != 0], feats)


def test_feature_id_mismatch_at_predict_fails(fitted):
    fc, hist, feats = fitted
    with pytest.raises(ValueError, match="must match future_df ids"):
        fc.predict(hist, _build_future(ids=list(range(N_IDS - 1))), feats)
    with pytest.raises(ValueError, match="must match future_df ids"):
        fc.predict(
            hist,
            _build_future(),
            feats[feats["unique_id"] != 0],
        )


def test_history_id_without_future_fails(fitted):
    fc, hist, feats = fitted
    future = _build_future(ids=[1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="ids without future"):
        fc.predict(hist, future, feats[feats["unique_id"] != 0])


def test_non_numeric_and_non_finite_features_fail():
    hist = _build_history()
    feats = _build_features()
    feats["f1"] = feats["f1"].astype(str)
    fc = _forecaster()
    with pytest.raises(ValueError, match="numeric dtype"):
        fc.fit(hist, feats)
    feats2 = _build_features()
    feats2.loc[0, "f1"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fc.fit(hist, feats2)


def test_mixed_backends_fail(fitted):
    fc, hist_pd, feats_pd = fitted
    future_pl = _build_future(engine="polars")
    with pytest.raises(ValueError, match="all be pandas or all be polars"):
        fc.predict(hist_pd, future_pl, feats_pd)


def test_fit_failure_does_not_pollute_previous_fit():
    hist = _build_history()
    feats = _build_features()
    good = _forecaster()
    good.fit(hist, feats)
    good_models = dict(good.models_)
    good_assignment = good.cluster_assignment().copy()

    # swap the declarative factory, then trigger a failing re-fit
    good.model_factory = Factory(
        "utilsforecast.test_models", "BadFitModel"
    )
    with pytest.raises(RuntimeError, match="boom during fit"):
        good.fit(hist, feats)
    assert good.models_ == good_models
    assert set(good.models_.keys()) == {0, 1}
    pd.testing.assert_frame_equal(good.cluster_assignment(), good_assignment)
    # old fit still serves correct predictions after the failed attempt
    pred = good.predict(hist, _build_future(), feats)
    assert pred.shape[0] == N_IDS * H


def test_unserializable_lambda_factory_is_rejected():
    with pytest.raises(ValueError, match="cannot be serialized declaratively"):
        Factory.from_spec(lambda: ConstantMeanModel())
    with pytest.raises(TypeError, match="Expected a Factory"):
        Factory.from_spec(42)


def test_predict_callback_error_does_not_return_partial_results():
    hist = _build_history()
    feats = _build_features()
    fc = SimilarSeriesForecaster(
        model_factory=Factory(
            "utilsforecast.test_models", "BadPredictModel"
        ),
        n_clusters=2,
    )
    fc.fit(hist, feats)
    with pytest.raises(RuntimeError, match="boom during predict"):
        fc.predict(hist, _build_future(), feats)


def test_wrong_prediction_keys_fail():
    hist = _build_history()
    feats = _build_features()
    fc = SimilarSeriesForecaster(
        model_factory=Factory(
            "utilsforecast.test_models", "WrongKeyModel"
        ),
        n_clusters=2,
    )
    fc.fit(hist, feats)
    with pytest.raises(ValueError, match="did not return exactly the future keys"):
        fc.predict(hist, _build_future(), feats)


def test_kmeans_more_clusters_than_series_fails():
    fc = _forecaster(n_clusters=99)
    with pytest.raises(ValueError, match="cannot be larger than"):
        fc.fit(_build_history(), _build_features())


# ---------------------------------------------------------------------------
# polars path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not POLARS_INSTALLED, reason="polars not installed")
def test_polars_predict_and_alignment():
    hist = _build_history(engine="polars")
    feats = _build_features(engine="polars")
    future = _build_future(engine="polars")
    fc = _forecaster()
    fc.fit(hist, feats)
    pred = fc.predict(hist, future, feats)
    assert isinstance(pred, pl.DataFrame)
    assert list(pred.columns) == ["unique_id", "ds", "y_hat"]
    assert pred.shape[0] == future.shape[0]
    assert pred.select(["unique_id", "ds"]).equals(
        future.select(["unique_id", "ds"])
    )
    # shuffled columns and rows still align by name
    shuffled_feats = _build_features(shuffled=True, engine="polars")
    pred2 = fc.predict(hist, future, shuffled_feats)
    assert pred.sort("unique_id", "ds").equals(pred2.sort("unique_id", "ds"))


@pytest.mark.skipif(not POLARS_INSTALLED, reason="polars not installed")
def test_polars_unseen_id_and_frozen_clusters():
    hist = _build_history(engine="polars")
    feats = _build_features(engine="polars")
    fc = _forecaster()
    fc.fit(hist, feats)
    new_id = N_IDS
    new_feats = pl.DataFrame(
        {
            "unique_id": list(range(N_IDS)) + [new_id],
            "f1": F1.tolist() + [0.05],
            "f2": F2.tolist() + [5.0],
        }
    )
    new_future = _build_future(
        ids=list(range(N_IDS)) + [new_id], engine="polars"
    )
    before = fc.cluster_assignment()
    pred = fc.predict(hist, new_future, new_feats)
    # fitted state (including the assignment table) is untouched
    pd.testing.assert_frame_equal(fc.cluster_assignment(), before)
    cluster_a = int(
        before.loc[before["unique_id"].isin(GROUP_A), "cluster"].iloc[0]
    )
    unseen = pred.filter(pl.col("unique_id") == new_id)["y_hat"].to_numpy()
    assert np.allclose(unseen, fc.models_[cluster_a].mean_)
    # moving a fitted id's features cannot re-cluster it
    moved = new_feats.with_columns(
        pl.when(pl.col("unique_id") == 0)
        .then(pl.col("f1") + 1000.0)
        .otherwise(pl.col("f1"))
        .alias("f1")
    )
    moved_pred = fc.predict(hist, new_future, moved)
    assert moved_pred.sort("unique_id", "ds").equals(
        pred.sort("unique_id", "ds")
    )


@pytest.mark.skipif(not POLARS_INSTALLED, reason="polars not installed")
def test_polars_validation_errors():
    hist = _build_history(engine="polars")
    feats = _build_features(engine="polars")
    fc = _forecaster()
    fc.fit(hist, feats)
    with pytest.raises(ValueError, match="duplicate"):
        fc.predict(
            hist,
            pl.concat([_build_future(engine="polars"),
                       _build_future(engine="polars").head(1)]),
            feats,
        )
    with pytest.raises(ValueError, match="Missing required column"):
        fc.predict(hist, _build_future(engine="polars"), feats.drop("f1"))


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_keeps_clusters_and_predictions(tmp_path, fitted):
    fc, hist, feats = fitted
    future = _build_future()
    pred_before = fc.predict(hist, future, feats)
    path = tmp_path / "model.pkl"
    fc.save(str(path))
    reloaded = SimilarSeriesForecaster.load(str(path))

    pd.testing.assert_frame_equal(
        reloaded.cluster_assignment().sort_values("unique_id").reset_index(drop=True),
        fc.cluster_assignment().sort_values("unique_id").reset_index(drop=True),
    )
    pred_after = reloaded.predict(hist, future, feats)
    pd.testing.assert_frame_equal(pred_before, pred_after)
    # the new-id path also matches after reload (frozen scaler + clusterer)
    new_feats = pd.DataFrame(
        {
            "unique_id": list(range(N_IDS)) + [N_IDS],
            "f1": F1.tolist() + [0.05],
            "f2": F2.tolist() + [5.0],
        }
    )
    new_future = _build_future(ids=list(range(N_IDS)) + [N_IDS])
    a = fc.predict(hist, new_future, new_feats)
    b = reloaded.predict(hist, new_future, new_feats)
    pd.testing.assert_frame_equal(a, b)


def test_plain_pickle_roundtrip(tmp_path, fitted):
    fc, hist, feats = fitted
    data = pickle.dumps(fc)
    reloaded = pickle.loads(data)
    future = _build_future()
    pd.testing.assert_frame_equal(
        fc.predict(hist, future, feats),
        reloaded.predict(hist, future, feats),
    )


def test_save_uses_declarative_factory_specs(tmp_path, fitted):
    fc, _, _ = fitted
    path = tmp_path / "model.pkl"
    fc.save(str(path))
    state = pickle.loads(path.read_bytes())
    spec = state["params"]["model_factory"]
    assert spec == {
        "module": "utilsforecast.test_models",
        "attr": "ConstantMeanModel",
        "kwargs": {},
    }
    assert state["params"]["clusterer_factory"]["attr"] == "KMeans"


def test_load_rejects_foreign_file(tmp_path):
    path = tmp_path / "foreign.pkl"
    path.write_bytes(pickle.dumps({"hello": "world"}))
    with pytest.raises(ValueError, match="not a SimilarSeriesForecaster"):
        SimilarSeriesForecaster.load(str(path))


def test_reload_in_fresh_isolated_interpreter(tmp_path, fitted):
    """Real save/reload: factories must resolve without the parent process."""
    fc, hist, feats = fitted
    path = tmp_path / "model.pkl"
    fc.save(str(path))
    pred_path = tmp_path / "pred.pkl"
    script = f"""
import pickle
import pandas as pd
from utilsforecast.clustering import SimilarSeriesForecaster

fc = SimilarSeriesForecaster.load({str(path)!r})
hist = pickle.loads({pickle.dumps(hist)!r})
feats = pickle.loads({pickle.dumps(feats)!r})
future = pd.DataFrame(
    [(uid, t) for uid in range({N_IDS}) for t in range({HIST_LEN}, {HIST_LEN + H})],
    columns=["unique_id", "ds"],
)
pred = fc.predict(hist, future, feats)
with open({str(pred_path)!r}, "wb") as fh:
    pickle.dump((fc.cluster_assignment(), pred), fh)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        cwd="/workspace",
    )
    assert result.returncode == 0, result.stderr
    assignment, pred = pickle.loads(pred_path.read_bytes())
    pd.testing.assert_frame_equal(
        assignment.sort_values("unique_id").reset_index(drop=True),
        fc.cluster_assignment().sort_values("unique_id").reset_index(drop=True),
    )
    assert pred.shape[0] == N_IDS * H


def test_factory_spec_variants():
    f1 = Factory.from_spec(("mymod", "MyClass"))
    f2 = Factory.from_spec(
        {"module": "mymod", "attr": "MyClass", "kwargs": {"k": 1}}
    )
    f3 = Factory.from_spec(StandardScaler)
    assert f1.module == "mymod" and f1.attr == "MyClass" and f1.kwargs == {}
    assert f2.kwargs == {"k": 1}
    assert (
        f3.module == "utilsforecast.clustering"
        and f3.attr == "StandardScaler"
    )
    assert f3.create().__class__ is StandardScaler
    with pytest.raises(ValueError, match="'module' and 'attr'"):
        Factory.from_spec({"attr": "X"})
    with pytest.raises(ValueError, match="tuples must have"):
        Factory.from_spec(("a", "b", {}, 1))


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------


def test_standard_scaler_centers_and_zero_variance():
    x = np.array([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]])
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    assert np.allclose(z.mean(axis=0), 0.0)
    assert np.isclose(z[:, 0].std(), 1.0)
    assert np.allclose(z[:, 1], 0.0)  # constant column -> unit scale
    with pytest.raises(ValueError, match="Expected 2 features"):
        scaler.transform(np.array([[1.0]]))
    with pytest.raises(ValueError, match="fitted before"):
        StandardScaler().transform(x)


def test_kmeans_deterministic_and_predictable():
    rng = np.random.default_rng(7)
    x = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0], size=(20, 2)),
            rng.normal(loc=[10.0, 10.0], size=(20, 2)),
        ]
    )
    km1 = KMeans(n_clusters=2, random_state=0, n_init=5)
    km2 = KMeans(n_clusters=2, random_state=0, n_init=5)
    labels1 = km1.fit_predict(x)
    labels2 = km2.fit_predict(x)
    assert np.array_equal(labels1, labels2)
    assert np.array_equal(km1.predict(x), labels1)
    # a point next to a centroid joins that cluster
    assert km1.predict(km1.cluster_centers_).tolist() == [0, 1]
    with pytest.raises(ValueError, match="fitted before"):
        KMeans(n_clusters=2).predict(x)


def test_kmeans_validation():
    with pytest.raises(ValueError, match="positive integer"):
        KMeans(n_clusters=0)
    with pytest.raises(ValueError, match="cannot be larger than"):
        KMeans(n_clusters=5).fit_predict(np.ones((2, 2)))


def test_custom_output_column_and_feature_cols():
    hist = _build_history()
    feats = _build_features()
    fc = SimilarSeriesForecaster(
        model_factory=Factory(
            "utilsforecast.test_models",
            "ConstantMeanModel",
            {"output_col": "pred"},
        ),
        n_clusters=2,
        feature_cols=["f2", "f1"],
        output_col="pred",
    )
    fc.fit(hist, feats)
    pred = fc.predict(hist, _build_future(), feats)
    assert list(pred.columns) == ["unique_id", "ds", "pred"]


def test_no_features_fails():
    fc = SimilarSeriesForecaster(
        model_factory=Factory(
            "utilsforecast.test_models", "ConstantMeanModel"
        ),
        feature_cols=[],
    )
    with pytest.raises(ValueError, match="At least one numeric feature"):
        fc.fit(_build_history(), _build_features())
