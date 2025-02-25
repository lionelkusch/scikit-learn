"""Accumulated Local Effect plots for regression and classification models."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import pandas as pd
from scipy import sparse

from ..base import is_classifier, is_regressor
from ..utils import Bunch, _safe_indexing, check_array
from ..utils._indexing import _determine_key_type, _get_column_indices
from ..utils._param_validation import (
    HasMethods,
    Integral,
    Interval,
    StrOptions,
    validate_params,
)
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._partial_dependence import _grid_from_X, _partial_dependence_brute
from ._pd_utils import _check_feature_names, _get_feature_index

__all__ = [
    "accumulated_local_effect",
]

@validate_params(
    {
        "estimator": [
            HasMethods(["fit", "predict"]),
            HasMethods(["fit", "predict_proba"]),
            HasMethods(["fit", "decision_function"]),
        ],
        "X": ["array-like", "sparse matrix"],
        "features": ["array-like", Integral, str],
        "sample_weight": ["array-like", None],
        "categorical_features": ["array-like", None],
        "feature_names": ["array-like", None],
        "response_method": [StrOptions({"auto", "predict_proba", "decision_function"})],
        "grid_resolution": [Interval(Integral, 1, None, closed="left")],
        "custom_values": [dict, None],
    },
    prefer_skip_nested_validation=True,
)
def accumulated_local_effect(
    estimator,
    X,
    features,
    *,
    sample_weight=None,
    categorical_features=None,
    feature_names=None,
    response_method="auto",
    grid_resolution=100,
    custom_values=None,
):
    """Accumulated Local Effect of ``features``.

    Accumulated Local Effect of a feature (or a set of features)
    corresponds to the average response of an estimator for each
    possible value of the feature considerating local effect.

    Read more in
    :ref:`sphx_glr_auto_examples_inspection_accumulated_local_effect.py`
    and the :ref:`User Guide <accumulated_local_effect>`.

    Parameters
    ----------
    estimator : BaseEstimator
        A fitted estimator object implementing :term:`predict`,
        :term:`predict_proba`, or :term:`decision_function`.
        Multioutput-multiclass classifiers are not supported.

    X : {array-like, sparse matrix or dataframe} of shape (n_samples, n_features)
        ``X`` is used to generate a grid of values for the target
        ``features`` (where the accumulated local effect will be evaluated), and
        also to generate values for the complement features

    features : array-like of {int, str, bool} or int or str
        The feature (e.g. `[0]`) or pair of interacting features
        (e.g. `[(0, 1)]`) for which the accumulated local effect should be computed.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights are used to calculate weighted means when averaging the
        model output. If `None`, then samples are equally weighted.

    categorical_features : array-like of shape (n_features,) or shape \
            (n_categorical_features,), dtype={bool, int, str}, default=None
        Indicates the categorical features.

        - `None`: no feature will be considered categorical;
        - boolean array-like: boolean mask of shape `(n_features,)`
            indicating which features are categorical. Thus, this array has
            the same shape has `X.shape[1]`;
        - integer or string array-like: integer indices or strings
            indicating categorical features.

    feature_names : array-like of shape (n_features,), dtype=str, default=None
        Name of each feature; `feature_names[i]` holds the name of the feature
        with index `i`.
        By default, the name of the feature corresponds to their numerical
        index for NumPy array and their column name for pandas dataframe.

    response_method : {'auto', 'predict_proba', 'decision_function'}, \
            default='auto'
        Specifies whether to use :term:`predict_proba` or
        :term:`decision_function` as the target response. For regressors
        this parameter is ignored and the response is always the output of
        :term:`predict`. By default, :term:`predict_proba` is tried first
        and we revert to :term:`decision_function` if it doesn't exist. If
        ``method`` is 'recursion', the response is always the output of
        :term:`decision_function`.

    grid_resolution : int, default=100
        The number of equally spaced points on the grid, for each target
        feature.
        This parameter is overridden by `custom_values` if that parameter is set.

    custom_values : dict
        A dictionary mapping the index of an element of `features` to an array
        of values where the accumulated local effect should be calculated
        for that feature. Setting a range of values for a feature overrides
        `grid_resolution` and `percentiles`.

        See :ref:`how to use accumulated_local_effect
        <plt_accumulated_local_effect_custom_values>` for an example of how this
        parameter can be used.

    Returns
    -------
    predictions : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        individual : ndarray of shape (n_outputs, n_instances, \
                len(values[0]), len(values[1]), ...)
            The predictions for all the points in the grid for all
            samples in X. This is also known as Individual
            Conditional Expectation (ICE).
            Only available when `kind='individual'` or `kind='both'`.

        local_effect: ndarray of shape (n_outputs, len(values[0]), \
                len(values[1]), ...)
            The local effect for each quantile of the points.

        average : ndarray of shape (n_outputs, len(values[0]), \
                len(values[1]), ...)
            The predictions for all the points in the grid, averaged
            over all samples in X corrected with local effetc.

        grid_values : seq of 1d ndarrays
            The values with which the grid has been created. The generated
            grid is a cartesian product of the arrays in `grid_values` where
            `len(grid_values) == len(features)`. The size of each array
            `grid_values[j]` is either `grid_resolution`, or the number of
            unique values in `X[:, j]`, whichever is smaller.

        `n_outputs` corresponds to the number of classes in a multi-class
        setting, or to the number of tasks for multi-output regression.
        For classical regression and binary classification `n_outputs==1`.
        `n_values_feature_j` corresponds to the size `grid_values[j]`.

    See Also
    --------
    AccumulatedLocalEffectDisplay.from_estimator : Plot Accumulated Local Effect.
    AccumulatedLocalEffectDisplay : Accumulated Local Effect visualization.

    Examples
    --------
    >>> X = [[0, 0, 2], [1, 0, 0]]
    >>> y = [0, 1]
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> gb = GradientBoostingClassifier(random_state=0).fit(X, y)
    >>> accumulated_local_effect(gb, features=[0], X=X, percentiles=(0, 1),
    ...                    grid_resolution=2) # doctest: +SKIP
    (array([[-4.52...,  4.52...]]), [array([ 0.,  1.])])
    """
    check_is_fitted(estimator)

    if not (is_classifier(estimator) or is_regressor(estimator)):
        raise ValueError("'estimator' must be a fitted regressor or classifier.")

    if is_classifier(estimator) and isinstance(estimator.classes_[0], np.ndarray):
        raise ValueError("Multiclass-multioutput estimators are not supported")

    # Use check_array only on lists and other non-array-likes / sparse. Do not
    # convert DataFrame into a NumPy array.
    if not (hasattr(X, "__array__") or sparse.issparse(X)):
        X = check_array(X, ensure_all_finite="allow-nan", dtype=object)

    if is_regressor(estimator) and response_method != "auto":
        raise ValueError(
            "The response_method parameter is ignored for regressors and "
            "must be 'auto'."
        )

    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X)

    if _determine_key_type(features, accept_slice=False) == "int":
        # _get_column_indices() supports negative indexing. Here, we limit
        # the indexing to be positive. The upper bound will be checked
        # by _get_column_indices()
        if np.any(np.less(features, 0)):
            raise ValueError("all features must be in [0, {}]".format(X.shape[1] - 1))

    features_indices = np.asarray(
        _get_column_indices(X, features), dtype=np.intp, order="C"
    ).ravel()

    feature_names = _check_feature_names(X, feature_names)

    n_features = X.shape[1]
    if categorical_features is None:
        is_categorical = [False] * len(features_indices)
    else:
        categorical_features = np.asarray(categorical_features)
        if categorical_features.dtype.kind == "b":
            # categorical features provided as a list of boolean
            if categorical_features.size != n_features:
                raise ValueError(
                    "When `categorical_features` is a boolean array-like, "
                    "the array should be of shape (n_features,). Got "
                    f"{categorical_features.size} elements while `X` contains "
                    f"{n_features} features."
                )
            is_categorical = [categorical_features[idx] for idx in features_indices]
        elif categorical_features.dtype.kind in ("i", "O", "U"):
            # categorical features provided as a list of indices or feature names
            categorical_features_idx = [
                _get_feature_index(cat, feature_names=feature_names)
                for cat in categorical_features
            ]
            is_categorical = [
                idx in categorical_features_idx for idx in features_indices
            ]
        else:
            raise ValueError(
                "Expected `categorical_features` to be an array-like of boolean,"
                f" integer, or string. Got {categorical_features.dtype} instead."
            )

    custom_values = custom_values or {}
    if isinstance(features, (str, int)):
        features = [features]

    warning_integer = False
    ale_results = Bunch(ale=[], quantile=[], center_quantile=[], mean_effect=[])
    for feature_idx, feature, is_cat in zip(features_indices, features, is_categorical):
        if warning_integer and _safe_indexing(X, feature_idx, axis=1).dtype.kind in "iu":
            # TODO(1.9): raise a ValueError instead.
            warnings.warn(
                f"The column {feature!r} contains integer data. Partial "
                "dependence plots are not supported for integer data: this "
                "can lead to implicit rounding with NumPy arrays or even errors "
                "with newer pandas versions. Please convert numerical features"
                "to floating point dtypes ahead of time to avoid problems. "
                "This will raise ValueError in scikit-learn 1.9.",
                FutureWarning,
            )
            # Do not warn again for other features to avoid spamming the caller.
            break
        if not is_cat:
            quantile = np.unique(
                np.quantile(
                    X[feature], np.linspace(0, 1, grid_resolution + 1), interpolation="lower"
                )
            )
            indices = np.clip(
                np.digitize(X[feature], quantile, right=True) - 1, 0, None
            )

        averaged_predictions, predictions = _partial_dependence_brute(
            estimator, [[quantile[indices]],[quantile[indices + 1]]], [feature_idx], X, response_method, sample_weight
        )
        effects = np.diff(predictions)
        # Average these differences within each bin.
        mean_effect = [np.mean(effects[np.where(indices == i)]) for i in np.unique(indices)]
        size_index = [len(np.where(indices == i)[0]) for i in np.unique(indices)]
        ale = np.array([0, *np.cumsum(mean_effect)])
        ale = (ale[1:] + ale[:-1]) / 2
        centers_quantile = (quantile[1:] + quantile[:-1]) / 2
        ale -= np.sum(ale * size_index / X.shape[0])

        ale_results["ale"].append(ale)
        ale_results["quantile"].append(quantile)
        ale_results["center_quantile"].append(centers_quantile)
        ale_results['mean_effect'].append(mean_effect)

    return ale_results
