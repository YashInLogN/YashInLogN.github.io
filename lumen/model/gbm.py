"""LightGBM regressor with sklearn fallback when native lib is unavailable."""

from __future__ import annotations

import warnings
from typing import Any, Literal

Backend = Literal["lightgbm", "sklearn_hist_gb"]


def create_fit_regressor(random_state: int = 42) -> tuple[Any, Backend]:
    try:
        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            verbose=-1,
        )
        return model, "lightgbm"
    except (OSError, ImportError) as exc:
        warnings.warn(
            "LightGBM could not load (on macOS, run: brew install libomp). "
            f"Using scikit-learn HistGradientBoostingRegressor for this session. ({exc})",
            stacklevel=2,
        )
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=200,
            random_state=random_state,
        )
        return model, "sklearn_hist_gb"
