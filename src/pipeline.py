from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

RANDOM_SEED = 42
VAL_START = pd.Timestamp("2024-01-01 01:00:00")
VAL_END = pd.Timestamp("2025-01-01 01:00:00")
ACTIVE_THRESHOLD_CF = 0.10
HUB_HEIGHT_M = 117.0
PREDICTION_MAX_CF = 1.05

# 장기정지 시간 처리 방식 (실험 9)
#   delete : 학습에서 제외 (기존 방식, 재현용)
#   keep   : 그대로 학습에 포함 (9-A)
#   weight : 학습에 포함하되 가중치를 낮춤 (9-B)
# 검증은 어떤 모드에서도 항상 전체 행으로 수행한다.
OUTAGE_MODE = "weight"
OUTAGE_WEIGHT = 0.5
OUTAGE_FLAG_COL = "is_long_outage_hour"

# ===== 실험 7: Quantile alpha 확장 =====
# 기존 후보(LightGBM 0.58 / CatBoost 0.55)에 더해 그룹별로 상위 alpha를 추가한다.
# 근거: Group1 최종 앙상블이 조건을 바꿔도 반복적으로 quantile 후보로만 채워졌고
# (삭제 모드 q58+q55, 가중0.5 모드 q58 80%+q55 20%), 이는 상위 quantile 쪽이
# 유리하다는 신호인데 0.58이 상한이라 더 위를 탐색하지 못하고 있었다.
EXTRA_LGB_QUANTILES = {
    1: (0.60, 0.65),
    2: (),                    # Group2는 가장 안정적이므로 대조군으로 유지
    3: (0.60, 0.65, 0.70),
}

# ===== 실험 6: 고출력 가중 학습 =====
# FICR은 실제 발전량으로 가중되므로 고출력 시간의 중요도가 훨씬 크다.
# 실제 CF 0.85 이상 구간의 2024 검증 진단(장기정지 가중0.5 기준):
#   Group1: 편향 -0.135, FICR 0.315, 발전량의 65.9%가 오차 8% 밖
#   Group2: 편향 -0.087, FICR 0.502, 46.2%   <- 세 그룹 중 가장 양호
#   Group3: 편향 -0.224, FICR 0.022, 97.4%   <- 사실상 전멸
# 다만 Group2는 전체 signed bias가 +2.11%로 이미 과대예측 경향이라
# 상향 압력을 더 주면 역효과가 우려된다. 따라서 Group2는 대조군으로 두고
# Group1·3에만 고출력 가중 후보를 추가한다.
# 기존 MAE 모델을 대체하지 않고 별도 후보로만 추가한다.
HIGH_OUTPUT_LAMBDA = 2.0
HIGH_OUTPUT_GROUPS = (1, 3)

# 파이프라인 단계 calibration 사용 여부.
#
# False: postprocess.py가 Group1 affine 보정을 고정 상수로 담당한다.
#        파이프라인은 보정하지 않은 base 예측을 그대로 내보낸다.
# True : 예전처럼 파이프라인이 H1에서 scale/offset을 탐색해 적용한다.
#        이 경우 postprocess.py의 Group1 affine과 중복되므로
#        둘 중 하나만 켜야 한다.
APPLY_PIPELINE_CALIBRATION = False


@dataclass(frozen=True)
class GroupConfig:
    group_id: int
    target_col: str
    capacity_kwh: float
    ldaps_grids: tuple[int, ...]
    turbine_weights: tuple[float, ...]
    is_vestas: int
    n_turbines: int
    rotor_diameter_m: float
    turbine_rated_mw: float


GROUPS: dict[int, GroupConfig] = {
    1: GroupConfig(1, "kpx_group_1", 21600.0, (5, 6, 10), (3.0, 2.0, 1.0), 1, 6, 126.0, 3.6),
    2: GroupConfig(2, "kpx_group_2", 21600.0, (6, 11), (3.0, 3.0), 1, 6, 126.0, 3.6),
    3: GroupConfig(3, "kpx_group_3", 21000.0, (6, 12), (1.0, 4.0), 0, 5, 136.0, 4.2),
}


@dataclass
class CandidateRecord:
    name: str
    best_iteration: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupRecipe:
    group_id: int
    component_weights: dict[str, float]
    candidate_records: dict[str, CandidateRecord]
    guided_floor: float
    guided_gamma: float
    calibration_scale: float
    calibration_offset_cf: float
    validation_metrics: dict[str, float]
    base_columns: list[str]
    advanced_columns: list[str]
    # 후처리 검증용 (직렬화 대상 아님)
    validation_base_cf: Any = None
    validation_expert_cf: Any = None
    validation_high85_probability: Any = None
    group3_gate_params: Any = None


def official_mask_kwh(y_true_kwh, capacity_kwh):
    y_true = np.asarray(y_true_kwh, dtype=float)
    return y_true >= capacity_kwh * ACTIVE_THRESHOLD_CF


def official_nmae(y_true_kwh, y_pred_kwh, capacity_kwh):
    y_true = np.asarray(y_true_kwh, dtype=float)
    y_pred = np.asarray(y_pred_kwh, dtype=float)
    mask = official_mask_kwh(y_true, capacity_kwh)
    if not mask.any():
        raise ValueError("The official evaluation mask is empty.")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])) / capacity_kwh)


def official_ficr(y_true_kwh, y_pred_kwh, capacity_kwh):
    y_true = np.asarray(y_true_kwh, dtype=float)
    y_pred = np.asarray(y_pred_kwh, dtype=float)
    mask = official_mask_kwh(y_true, capacity_kwh)
    if not mask.any():
        raise ValueError("The official evaluation mask is empty.")
    actual = y_true[mask]
    error_ratio = np.abs(actual - y_pred[mask]) / capacity_kwh
    settlement_rate = np.where(error_ratio <= 0.06, 4.0, np.where(error_ratio <= 0.08, 3.0, 0.0))
    maximum = float((4.0 * actual).sum())
    if maximum <= 0:
        return float("nan")
    return float((settlement_rate * actual).sum() / maximum)


def official_metrics(y_true_kwh, y_pred_kwh, capacity_kwh):
    nmae = official_nmae(y_true_kwh, y_pred_kwh, capacity_kwh)
    ficr = official_ficr(y_true_kwh, y_pred_kwh, capacity_kwh)
    return {
        "nmae": nmae,
        "one_minus_nmae": 1.0 - nmae,
        "ficr": ficr,
        "official_score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
    }


def active_nmae_eval_factory(capacity_kwh):
    def _metric(y_true, y_pred):
        y_true_kwh = np.asarray(y_true, dtype=float) * capacity_kwh
        y_pred_kwh = np.asarray(y_pred, dtype=float) * capacity_kwh
        return "official_nmae", official_nmae(y_true_kwh, y_pred_kwh, capacity_kwh), False
    return _metric


def read_group_csv(data_dir: Path, split: str, config: GroupConfig) -> pd.DataFrame:
    path = data_dir / f"{split}_group{config.group_id}_preprocessed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "forecast_kst_dtm" not in frame.columns:
        raise KeyError(f"{path.name}: forecast_kst_dtm column is missing.")
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"], errors="raise")
    frame = frame.sort_values("forecast_kst_dtm").reset_index(drop=True)
    if frame["forecast_kst_dtm"].duplicated().any():
        raise ValueError(f"{path.name}: duplicated timestamps detected")
    if split == "train":
        if config.target_col not in frame.columns:
            raise KeyError(f"{path.name}: target column missing.")
        if frame[config.target_col].isna().any():
            raise ValueError(f"{path.name}: target contains missing values.")
    elif config.target_col in frame.columns:
        frame = frame.drop(columns=config.target_col)
    return frame


def split_masks(timestamps: pd.Series):
    dt = pd.to_datetime(timestamps, errors="raise")
    train_mask = (dt < VAL_START).to_numpy()
    val_mask = ((dt >= VAL_START) & (dt < VAL_END)).to_numpy()
    if not train_mask.any() or not val_mask.any():
        raise ValueError("Train/validation split produced an empty partition.")
    return train_mask, val_mask


def numeric_feature_frame(frame: pd.DataFrame, target_col: str | None) -> pd.DataFrame:
    excluded = {"forecast_kst_dtm", OUTAGE_FLAG_COL}
    if target_col is not None:
        excluded.add(target_col)
    result = frame[[c for c in frame.columns if c not in excluded]].copy()
    for col in result.columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)


def _group_cycle_key(timestamps: pd.Series) -> pd.Series:
    return (pd.to_datetime(timestamps, errors="raise") - pd.Timedelta(hours=1)).dt.normalize()


def _add_weighted_grid_features(result: pd.DataFrame, config: GroupConfig) -> pd.DataFrame:
    additions: dict[str, np.ndarray] = {}
    grid_regex = re.compile(r"(.+)_grid(\d+)$")
    grouped: dict[str, dict[int, str]] = {}
    for col in result.columns:
        match = grid_regex.fullmatch(col)
        if match:
            grouped.setdefault(match.group(1), {})[int(match.group(2))] = col
    weights = np.asarray(config.turbine_weights, dtype=float)
    weights = weights / weights.sum()
    for base_name, grid_map in grouped.items():
        if not all(grid in grid_map for grid in config.ldaps_grids):
            continue
        cols = [grid_map[grid] for grid in config.ldaps_grids]
        values = result[cols].to_numpy(dtype=float)
        weighted_mean = values @ weights
        additions[f"{base_name}_group_wmean"] = weighted_mean
        additions[f"{base_name}_group_min"] = np.nanmin(values, axis=1)
        additions[f"{base_name}_group_max"] = np.nanmax(values, axis=1)
        additions[f"{base_name}_group_range"] = np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
        additions[f"{base_name}_group_wstd"] = np.sqrt(
            np.nansum(weights * (values - weighted_mean[:, None]) ** 2, axis=1))
    if additions:
        result = pd.concat([result, pd.DataFrame(additions, index=result.index)], axis=1)
    return result


def build_advanced_features(frame: pd.DataFrame, config: GroupConfig, target_col: str | None) -> pd.DataFrame:
    excluded = {"forecast_kst_dtm", OUTAGE_FLAG_COL}
    if target_col is not None:
        excluded.add(target_col)
    source_cols = [c for c in frame.columns if c not in excluded and not c.endswith(("_roll3h", "_roll6h"))]
    result = frame[source_cols].copy()
    for col in result.columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    dt = pd.to_datetime(frame["forecast_kst_dtm"], errors="raise")
    cycle_key = _group_cycle_key(dt)
    lead = np.where(dt.dt.hour.eq(0), 24, dt.dt.hour).astype(np.int8)

    time_features = pd.DataFrame({
        "lead_hour": lead,
        "lead_sin": np.sin(2.0 * np.pi * lead / 24.0),
        "lead_cos": np.cos(2.0 * np.pi * lead / 24.0),
        "hour_sin_v2": np.sin(2.0 * np.pi * dt.dt.hour.to_numpy() / 24.0),
        "hour_cos_v2": np.cos(2.0 * np.pi * dt.dt.hour.to_numpy() / 24.0),
        "doy_sin_v2": np.sin(2.0 * np.pi * dt.dt.dayofyear.to_numpy() / 365.25),
        "doy_cos_v2": np.cos(2.0 * np.pi * dt.dt.dayofyear.to_numpy() / 365.25),
        "month_sin": np.sin(2.0 * np.pi * (dt.dt.month.to_numpy() - 1) / 12.0),
        "month_cos": np.cos(2.0 * np.pi * (dt.dt.month.to_numpy() - 1) / 12.0),
    }, index=result.index)
    result = pd.concat([result, time_features], axis=1)

    hub_additions: dict[str, np.ndarray] = {}
    for grid in config.ldaps_grids:
        speed_col = f"ldaps_50m_mean_wind_speed_grid{grid}"
        shear_col = f"ldaps_shear_alpha_10m_50m_grid{grid}"
        rho_col = f"ldaps_rho_grid{grid}"
        if speed_col in result.columns and shear_col in result.columns:
            speed = pd.to_numeric(result[speed_col], errors="coerce")
            alpha = pd.to_numeric(result[shear_col], errors="coerce").clip(-0.5, 0.7)
            hub = (speed * (HUB_HEIGHT_M / 50.0) ** alpha).clip(0.0, 60.0)
            hub_additions[f"ldaps_hub117_grid{grid}"] = hub.to_numpy()
            hub_additions[f"ldaps_hub117_cubed_grid{grid}"] = hub.pow(3).to_numpy()
            if rho_col in result.columns:
                rho = pd.to_numeric(result[rho_col], errors="coerce")
                hub_additions[f"ldaps_wpd117_grid{grid}"] = (0.5 * rho * hub.pow(3)).to_numpy()

    if "gfs_100m_wind_speed" in result.columns:
        gfs_speed = pd.to_numeric(result["gfs_100m_wind_speed"], errors="coerce")
        if "gfs_shear_alpha_10m_100m" in result.columns:
            gfs_alpha = pd.to_numeric(result["gfs_shear_alpha_10m_100m"], errors="coerce").clip(-0.5, 0.7)
        else:
            gfs_alpha = pd.Series(0.14, index=result.index)
        gfs_hub = (gfs_speed * (HUB_HEIGHT_M / 100.0) ** gfs_alpha).clip(0.0, 60.0)
        hub_additions["gfs_hub117"] = gfs_hub.to_numpy()
        hub_additions["gfs_hub117_cubed"] = gfs_hub.pow(3).to_numpy()
        if "gfs_rho" in result.columns:
            rho = pd.to_numeric(result["gfs_rho"], errors="coerce")
            hub_additions["gfs_wpd117"] = (0.5 * rho * gfs_hub.pow(3)).to_numpy()

    if hub_additions:
        result = pd.concat([result, pd.DataFrame(hub_additions, index=result.index)], axis=1)

    result = _add_weighted_grid_features(result, config)

    ldaps_hub = "ldaps_hub117_group_wmean"
    if ldaps_hub in result.columns and "gfs_hub117" in result.columns:
        result["hub117_mean"] = 0.5 * (result[ldaps_hub] + result["gfs_hub117"])
        result["hub117_diff"] = result[ldaps_hub] - result["gfs_hub117"]
        result["hub117_abs_diff"] = result["hub117_diff"].abs()
        result["hub117_ratio"] = result[ldaps_hub] / (result["gfs_hub117"] + 0.3)
        result["hub117_mean_squared"] = result["hub117_mean"].pow(2)
        result["hub117_mean_cubed"] = result["hub117_mean"].pow(3)

    if "gfs_surface_0_gust" in result.columns and "gfs_10m_wind_speed" in result.columns:
        result["gfs_gust_factor"] = result["gfs_surface_0_gust"] / (result["gfs_10m_wind_speed"] + 0.3)

    temp_col = "heightAboveGround_2_t_group_wmean"
    humidity_col = "heightAboveGround_2_r_group_wmean"
    if temp_col in result.columns and humidity_col in result.columns:
        cold = np.maximum(273.15 - result[temp_col], 0.0)
        humid = np.maximum(result[humidity_col] - 85.0, 0.0)
        result["icing_risk_soft"] = cold * humid / 15.0
        result["icing_risk_flag"] = ((result[temp_col] < 273.15) & (result[humidity_col] >= 90.0)).astype(np.int8)

    sin_candidates = ["ldaps_50m_mean_dir_sin_group_wmean", "ldaps_10m_dir_sin_group_wmean"]
    cos_candidates = ["ldaps_50m_mean_dir_cos_group_wmean", "ldaps_10m_dir_cos_group_wmean"]
    for sin_col, cos_col in zip(sin_candidates, cos_candidates):
        if sin_col in result.columns and cos_col in result.columns:
            prefix = sin_col.replace("_dir_sin_group_wmean", "")
            result[f"{prefix}_dir_coherence"] = np.sqrt(result[sin_col].pow(2) + result[cos_col].pow(2))

    context_cols = [c for c in ("ldaps_hub117_group_wmean", "ldaps_hub117_group_range", "gfs_hub117",
                                "hub117_mean", "hub117_diff", "gfs_surface_0_sp") if c in result.columns]
    for col in context_cols:
        grouped = result[col].groupby(cycle_key, sort=False)
        result[f"{col}_diff_m1"] = grouped.diff(1).fillna(0.0)
        result[f"{col}_diff_m3"] = grouped.diff(3).fillna(0.0)
        result[f"{col}_diff_p1"] = grouped.shift(-1).sub(result[col]).fillna(0.0)
        result[f"{col}_cycle_roll3"] = grouped.transform(lambda s: s.rolling(7, center=True, min_periods=1).mean())
        result[f"{col}_cycle_roll6"] = grouped.transform(lambda s: s.rolling(13, center=True, min_periods=1).mean())
        result[f"{col}_cycle_std3"] = grouped.transform(lambda s: s.rolling(7, center=True, min_periods=2).std()).fillna(0.0)

    # ---- 추가 피처: 예보 회차 내부 lag/lead, 구간 변동성, LDAPS-GFS 소스 불일치 ----
    # (검증 실험 결과 팀원 기본 피처 대비 3개 그룹 모두 Score 개선 확인됨)
    key_wind_cols_by_group = {
        1: ["ldaps_10m_wind_speed_grid6", "ldaps_50m_mean_wind_speed_grid6", "gfs_850hpa_wind_speed"],
        2: ["ldaps_10m_wind_speed_grid6", "ldaps_50m_mean_wind_speed_grid6",
            "ldaps_10m_wind_speed_grid11", "gfs_850hpa_wind_speed"],
        3: ["ldaps_10m_wind_speed_grid12", "ldaps_50m_mean_wind_speed_grid12", "gfs_850hpa_wind_speed"],
    }
    extra_additions: dict[str, Any] = {}
    for col in key_wind_cols_by_group.get(config.group_id, []):
        if col not in result.columns:
            continue
        grouped = result[col].groupby(cycle_key, sort=False)
        for k in (1, 2, 3):
            extra_additions[f"{col}_cyc_lag{k}"] = grouped.shift(k)
            extra_additions[f"{col}_cyc_lead{k}"] = grouped.shift(-k)
        extra_additions[f"{col}_cyc_rollstd3"] = grouped.transform(
            lambda s: s.rolling(7, center=True, min_periods=2).std())
        roll_mean = grouped.transform(lambda s: s.rolling(13, center=True, min_periods=1).mean())
        extra_additions[f"{col}_cyc_dev"] = result[col] - roll_mean

    main_grid = config.ldaps_grids[0]
    disagreement_pairs = [
        (f"ldaps_10m_wind_speed_grid{main_grid}", "gfs_10m_wind_speed", "src_disagree_10m"),
        (f"ldaps_50m_mean_wind_speed_grid{main_grid}", "gfs_80m_wind_speed", "src_disagree_mid"),
        (f"ldaps_10m_wind_speed_grid{main_grid}", "gfs_850hpa_wind_speed", "src_disagree_850"),
    ]
    for col_a, col_b, name in disagreement_pairs:
        if col_a in result.columns and col_b in result.columns:
            diff = result[col_a] - result[col_b]
            extra_additions[name] = diff
            extra_additions[f"{name}_abs"] = diff.abs()

    if extra_additions:
        result = pd.concat([result, pd.DataFrame(extra_additions, index=result.index).fillna(0.0)], axis=1)

    return result.replace([np.inf, -np.inf], np.nan)


def align_features(train_features, other_features, medians=None):
    train_cols = list(train_features.columns)
    missing = sorted(set(train_cols) - set(other_features.columns))
    extra = sorted(set(other_features.columns) - set(train_cols))
    if missing or extra:
        raise ValueError(f"Feature schema mismatch. missing={missing[:10]}, extra={extra[:10]}")
    other = other_features[train_cols].copy()
    train = train_features[train_cols].copy()
    if medians is None:
        medians = train.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train = train.fillna(medians).fillna(0.0).astype(np.float32)
    other = other.fillna(medians).fillna(0.0).astype(np.float32)
    return train, other, medians


def build_pooled_features(frame: pd.DataFrame, config: GroupConfig) -> pd.DataFrame:
    advanced = build_advanced_features(frame, config, config.target_col if config.target_col in frame else None)
    result: dict[str, np.ndarray] = {}
    keep_named = {"lead_hour", "lead_sin", "lead_cos", "hour_sin_v2", "hour_cos_v2", "doy_sin_v2",
                  "doy_cos_v2", "month_sin", "month_cos", "hub117_mean", "hub117_diff",
                  "hub117_abs_diff", "hub117_ratio", "hub117_mean_squared", "hub117_mean_cubed",
                  "icing_risk_soft", "icing_risk_flag"}
    for col in advanced.columns:
        if col.startswith("gfs_") or "_group_" in col or col in keep_named:
            result[col] = pd.to_numeric(advanced[col], errors="coerce").to_numpy()
    n = len(frame)
    result["group_id"] = np.full(n, config.group_id, dtype=np.int8)
    result["is_vestas"] = np.full(n, config.is_vestas, dtype=np.int8)
    result["n_turbines"] = np.full(n, config.n_turbines, dtype=np.int8)
    result["rotor_diameter_m"] = np.full(n, config.rotor_diameter_m, dtype=np.float32)
    result["turbine_rated_mw"] = np.full(n, config.turbine_rated_mw, dtype=np.float32)
    result["group_capacity_kwh"] = np.full(n, config.capacity_kwh, dtype=np.float32)
    return pd.DataFrame(result, index=frame.index).replace([np.inf, -np.inf], np.nan)


def lgb_base_params(group_id: int) -> dict[str, Any]:
    return {
        "n_estimators": 2600,
        "learning_rate": 0.018,
        "num_leaves": {1: 31, 2: 31, 3: 23}[group_id],
        "max_depth": -1,
        "min_child_samples": {1: 35, 2: 40, 3: 55}[group_id],
        "colsample_bytree": 0.82,
        "subsample": 0.90,
        "subsample_freq": 1,
        "reg_alpha": {1: 0.15, 2: 0.20, 3: 0.35}[group_id],
        "reg_lambda": {1: 2.5, 2: 3.0, 3: 4.0}[group_id],
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": -1,
    }


def fit_lgb_regressor(group_id, x_train, y_train_cf, x_val, y_val_cf, capacity_kwh, *,
                      objective="regression_l1", alpha=None, sample_weight=None,
                      active_validation_only=False):
    params = lgb_base_params(group_id)
    params["objective"] = objective
    if alpha is not None:
        params["alpha"] = alpha
    model = lgb.LGBMRegressor(**params)
    eval_metric = "l1" if active_validation_only else active_nmae_eval_factory(capacity_kwh)
    model.fit(x_train, y_train_cf, sample_weight=sample_weight, eval_set=[(x_val, y_val_cf)],
              eval_metric=eval_metric,
              callbacks=[lgb.early_stopping(140, verbose=False, first_metric_only=True), lgb.log_evaluation(0)])
    return model


def fit_lgb_classifier(group_id, x_train, y_train, x_val, y_val):
    params = lgb_base_params(group_id)
    params.update({"objective": "binary", "n_estimators": 1800, "learning_rate": 0.025,
                   "metric": "binary_logloss"})
    model = lgb.LGBMClassifier(**params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    return model


def catboost_params(group_id, loss_function):
    return {
        "loss_function": loss_function,
        "eval_metric": "MAE",
        "iterations": 2200,
        "depth": {1: 7, 2: 7, 3: 6}[group_id],
        "learning_rate": 0.025,
        "l2_leaf_reg": {1: 6.0, 2: 7.0, 3: 9.0}[group_id],
        "random_seed": RANDOM_SEED,
        "thread_count": -1,
        "verbose": False,
        "od_type": "Iter",
        "od_wait": 140,
        "allow_writing_files": False,
    }


def fit_catboost(group_id, x_train, y_train_cf, x_val, y_val_cf, *, loss_function, sample_weight=None):
    model = CatBoostRegressor(**catboost_params(group_id, loss_function))
    model.fit(x_train, y_train_cf, sample_weight=sample_weight, eval_set=(x_val, y_val_cf),
              use_best_model=True, verbose=False)
    return model


def safe_best_iteration(model, default=1000):
    if hasattr(model, "best_iteration_") and model.best_iteration_ is not None:
        return max(1, int(model.best_iteration_))
    if isinstance(model, CatBoostRegressor):
        value = model.get_best_iteration()
        return max(1, int(value + 1)) if value is not None and value >= 0 else default
    return default


def clipped_cf_prediction(model, features):
    return np.clip(np.asarray(model.predict(features), dtype=float), 0.0, PREDICTION_MAX_CF)


def fit_pooled_validation_models(train_frames):
    feature_parts, targets, train_masks, val_masks, group_ids = [], [], [], [], []
    for group_id, config in GROUPS.items():
        frame = train_frames[group_id]
        feature_parts.append(build_pooled_features(frame, config))
        targets.append(frame[config.target_col].to_numpy(dtype=float) / config.capacity_kwh)
        train_mask, val_mask = split_masks(frame["forecast_kst_dtm"])
        train_masks.append(train_mask)
        val_masks.append(val_mask)
        group_ids.append(np.full(len(frame), group_id, dtype=np.int8))

    features = pd.concat(feature_parts, ignore_index=True)
    y = np.concatenate(targets)
    train_mask = np.concatenate(train_masks)
    val_mask = np.concatenate(val_masks)
    groups = np.concatenate(group_ids)

    medians = features.loc[train_mask].median(numeric_only=True).fillna(0.0)
    features = features.fillna(medians).fillna(0.0).astype(np.float32)
    active = y >= ACTIVE_THRESHOLD_CF

    models, predictions, iterations = {}, {}, {}
    for name, group3_multiplier in (("pooled_cat_balanced", 1.0), ("pooled_cat_g3x2", 2.0)):
        fit_mask = train_mask & active
        val_active = val_mask & active
        weights = np.ones(fit_mask.sum(), dtype=float)
        fit_groups = groups[fit_mask]
        weights[fit_groups == 3] *= group3_multiplier
        model = CatBoostRegressor(loss_function="MAE", eval_metric="MAE", iterations=2400, depth=7,
                                   learning_rate=0.025, l2_leaf_reg=8.0, random_seed=RANDOM_SEED,
                                   thread_count=-1, verbose=False, od_type="Iter", od_wait=150,
                                   allow_writing_files=False)
        model.fit(features.loc[fit_mask], y[fit_mask], sample_weight=weights,
                  eval_set=(features.loc[val_active], y[val_active]), use_best_model=True, verbose=False)
        models[name] = model
        iterations[name] = safe_best_iteration(model, 1400)
        predictions[name] = {}
        for group_id in GROUPS:
            mask = val_mask & (groups == group_id)
            predictions[name][group_id] = clipped_cf_prediction(model, features.loc[mask])
    return models, predictions, list(features.columns), medians, iterations


def fit_pooled_full_models(train_frames, test_frames, columns, medians, iterations):
    train_parts, test_parts, y_parts, train_group_parts, test_group_parts = [], [], [], [], []
    for group_id, config in GROUPS.items():
        train_parts.append(build_pooled_features(train_frames[group_id], config))
        test_parts.append(build_pooled_features(test_frames[group_id], config))
        y_parts.append(train_frames[group_id][config.target_col].to_numpy(dtype=float) / config.capacity_kwh)
        train_group_parts.append(np.full(len(train_frames[group_id]), group_id, dtype=np.int8))
        test_group_parts.append(np.full(len(test_frames[group_id]), group_id, dtype=np.int8))

    x_train_raw = pd.concat(train_parts, ignore_index=True)[columns]
    x_test_raw = pd.concat(test_parts, ignore_index=True)[columns]
    full_medians = x_train_raw.median(numeric_only=True).fillna(medians).fillna(0.0)
    x_train = x_train_raw.fillna(full_medians).fillna(0.0).astype(np.float32)
    x_test = x_test_raw.fillna(full_medians).fillna(0.0).astype(np.float32)
    y_train = np.concatenate(y_parts)
    train_groups = np.concatenate(train_group_parts)
    test_groups = np.concatenate(test_group_parts)
    active = y_train >= ACTIVE_THRESHOLD_CF

    models, predictions = {}, {}
    for name, group3_multiplier in (("pooled_cat_balanced", 1.0), ("pooled_cat_g3x2", 2.0)):
        weights = np.ones(active.sum(), dtype=float)
        weights[train_groups[active] == 3] *= group3_multiplier
        model = CatBoostRegressor(loss_function="MAE", iterations=max(1, int(iterations[name])), depth=7,
                                   learning_rate=0.025, l2_leaf_reg=8.0, random_seed=RANDOM_SEED,
                                   thread_count=-1, verbose=False, allow_writing_files=False)
        model.fit(x_train.loc[active], y_train[active], sample_weight=weights, verbose=False)
        models[name] = model
        predictions[name] = {}
        for group_id in GROUPS:
            mask = test_groups == group_id
            predictions[name][group_id] = clipped_cf_prediction(model, x_test.loc[mask])
    return models, predictions


def select_guided_hurdle(y_val_kwh, capacity_kwh, direct_cf, conditional_cf, active_probability):
    best_pred = conditional_cf.copy()
    best_floor, best_gamma = 1.0, 1.0
    best_metrics = official_metrics(y_val_kwh, best_pred * capacity_kwh, capacity_kwh)
    for floor in np.arange(0.50, 1.001, 0.05):
        for gamma in (0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            trust = floor + (1.0 - floor) * np.power(active_probability, gamma)
            prediction = np.clip(trust * conditional_cf + (1.0 - trust) * direct_cf, 0.0, PREDICTION_MAX_CF)
            metrics = official_metrics(y_val_kwh, prediction * capacity_kwh, capacity_kwh)
            if (metrics["nmae"] < best_metrics["nmae"] - 1e-12 or
                    (abs(metrics["nmae"] - best_metrics["nmae"]) <= 1e-12 and metrics["ficr"] > best_metrics["ficr"])):
                best_pred, best_floor, best_gamma, best_metrics = prediction, float(floor), float(gamma), metrics
    return best_pred, best_floor, best_gamma, best_metrics


def _candidate_weight_vectors(n_components, random_trials):
    for i in range(n_components):
        v = np.zeros(n_components); v[i] = 1.0; yield v
    for i in range(n_components):
        for j in range(i + 1, n_components):
            for first in np.arange(0.1, 1.0, 0.1):
                v = np.zeros(n_components); v[i] = first; v[j] = 1.0 - first; yield v
    yield np.full(n_components, 1.0 / n_components)
    rng = np.random.default_rng(RANDOM_SEED)
    for v in rng.dirichlet(np.full(n_components, 0.45), size=random_trials):
        yield v


def optimize_constrained_blend(y_val_kwh, capacity_kwh, components_cf, *, random_trials=35000, nmae_tolerance=0.0015):
    names = list(components_cf)
    matrix = np.column_stack([np.asarray(components_cf[n], dtype=float) for n in names])
    component_metrics = {n: official_metrics(y_val_kwh, matrix[:, i] * capacity_kwh, capacity_kwh)
                         for i, n in enumerate(names)}
    best_component_nmae = min(m["nmae"] for m in component_metrics.values())
    maximum_allowed_nmae = best_component_nmae + nmae_tolerance
    best_weights = best_prediction = best_metrics = None
    for weights in _candidate_weight_vectors(len(names), random_trials):
        prediction = np.clip(matrix @ weights, 0.0, PREDICTION_MAX_CF)
        metrics = official_metrics(y_val_kwh, prediction * capacity_kwh, capacity_kwh)
        if metrics["nmae"] > maximum_allowed_nmae:
            continue
        if best_metrics is None or (
            metrics["official_score"] > best_metrics["official_score"] + 1e-12 or
            (abs(metrics["official_score"] - best_metrics["official_score"]) <= 1e-12 and metrics["nmae"] < best_metrics["nmae"])):
            best_weights, best_prediction, best_metrics = weights.copy(), prediction, metrics
    if best_weights is None:
        best_name = min(component_metrics, key=lambda n: component_metrics[n]["nmae"])
        best_weights = np.zeros(len(names)); best_weights[names.index(best_name)] = 1.0
        best_prediction = matrix[:, names.index(best_name)]
        best_metrics = component_metrics[best_name]
    return ({n: float(w) for n, w in zip(names, best_weights)}, best_prediction, best_metrics, component_metrics)


def conservative_calibration(timestamps, y_val_kwh, prediction_cf, capacity_kwh):
    # 파이프라인 단계의 calibration은 비활성화한다.
    #
    # 이유: Group1 affine 보정(scale 1.08, offset -0.025)은 postprocess.py가
    # 고정 상수로 담당한다. 여기서 또 scale/offset을 탐색해 적용하면
    # 같은 예측에 보정이 두 번 걸린다.
    #
    # 또한 이 값들은 H1 순방향 검증으로 선택한 고정 규칙이므로
    # 매 실행마다 재탐색하면 재현성이 깨진다.
    if not APPLY_PIPELINE_CALIBRATION:
        return 1.0, 0.0, prediction_cf, {
            "accepted": False,
            "reason": "disabled - postprocess.py가 calibration을 담당함",
        }

    dt = pd.to_datetime(timestamps, errors="raise")
    eval_mask = official_mask_kwh(y_val_kwh, capacity_kwh)
    h1 = (dt < pd.Timestamp("2024-07-01 01:00:00")).to_numpy() & eval_mask
    h2 = (dt >= pd.Timestamp("2024-07-01 01:00:00")).to_numpy() & eval_mask
    base_full = official_metrics(y_val_kwh, prediction_cf * capacity_kwh, capacity_kwh)
    if not h1.any() or not h2.any():
        return 1.0, 0.0, prediction_cf, {"accepted": False, "reason": "empty half-year mask"}
    best_h1_score, best_scale, best_offset = -math.inf, 1.0, 0.0
    for scale in np.arange(0.94, 1.061, 0.005):
        for offset in np.arange(-0.025, 0.0251, 0.005):
            pred = np.clip(prediction_cf * scale + offset, 0.0, PREDICTION_MAX_CF)
            metrics = official_metrics(y_val_kwh[h1], pred[h1] * capacity_kwh, capacity_kwh)
            if metrics["official_score"] > best_h1_score:
                best_h1_score, best_scale, best_offset = metrics["official_score"], float(scale), float(offset)
    candidate = np.clip(prediction_cf * best_scale + best_offset, 0.0, PREDICTION_MAX_CF)
    h2_before = official_metrics(y_val_kwh[h2], prediction_cf[h2] * capacity_kwh, capacity_kwh)
    h2_after = official_metrics(y_val_kwh[h2], candidate[h2] * capacity_kwh, capacity_kwh)
    full_after = official_metrics(y_val_kwh, candidate * capacity_kwh, capacity_kwh)
    accepted = (h2_after["official_score"] > h2_before["official_score"] and
                full_after["official_score"] > base_full["official_score"] and
                full_after["nmae"] <= base_full["nmae"] + 0.0005)
    if not accepted:
        return 1.0, 0.0, prediction_cf, {"accepted": False, "base_full": base_full,
                                          "candidate_full": full_after, "h2_before": h2_before, "h2_after": h2_after}
    return best_scale, best_offset, candidate, {"accepted": True, "base_full": base_full,
                                                 "candidate_full": full_after, "h2_before": h2_before, "h2_after": h2_after}


def outage_training_adjustment(frame, train_mask):
    """
    실험 9용. 학습 행에 대해서만 장기정지 처리를 적용한다.

    반환
      keep_mask : 학습에 사용할 행 (train_mask 내부 기준, bool 배열)
      multiplier: 학습 행별 가중치 배수 (bool 배열과 동일 길이)

    검증 행에는 어떤 처리도 적용하지 않는다.
    """
    n_train = int(train_mask.sum())

    if OUTAGE_FLAG_COL not in frame.columns:
        return np.ones(n_train, dtype=bool), np.ones(n_train, dtype=float)

    flag = pd.to_numeric(frame.loc[train_mask, OUTAGE_FLAG_COL], errors="coerce")
    flag = flag.fillna(0).to_numpy().astype(bool)

    if OUTAGE_MODE == "delete":
        return ~flag, np.ones(n_train, dtype=float)
    if OUTAGE_MODE == "keep":
        return np.ones(n_train, dtype=bool), np.ones(n_train, dtype=float)
    if OUTAGE_MODE == "weight":
        mult = np.where(flag, OUTAGE_WEIGHT, 1.0)
        return np.ones(n_train, dtype=bool), mult
    raise ValueError(f"알 수 없는 OUTAGE_MODE: {OUTAGE_MODE}")


def train_group_candidates(frame, config, pooled_predictions):
    train_mask, val_mask = split_masks(frame["forecast_kst_dtm"])
    timestamps = frame["forecast_kst_dtm"]
    y_cf = frame[config.target_col].to_numpy(dtype=float) / config.capacity_kwh
    y_train, y_val = y_cf[train_mask], y_cf[val_mask]
    y_val_kwh = y_val * config.capacity_kwh

    base_all = numeric_feature_frame(frame, config.target_col)
    advanced_all = build_advanced_features(frame, config, config.target_col)
    x_base_train, x_base_val, base_medians = align_features(base_all.loc[train_mask], base_all.loc[val_mask])
    x_adv_train, x_adv_val, adv_medians = align_features(advanced_all.loc[train_mask], advanced_all.loc[val_mask])

    # ---- 실험 9: 장기정지 시간 처리 (학습 행에만 적용) ----
    keep_mask, outage_mult = outage_training_adjustment(frame, train_mask)
    train_times_full = timestamps.loc[train_mask].reset_index(drop=True)

    if not keep_mask.all():
        x_base_train = x_base_train.loc[keep_mask]
        x_adv_train = x_adv_train.loc[keep_mask]
        y_train = y_train[keep_mask]
        outage_mult = outage_mult[keep_mask]
        train_times_full = train_times_full.loc[keep_mask].reset_index(drop=True)

    active_train = y_train >= ACTIVE_THRESHOLD_CF
    active_val = y_val >= ACTIVE_THRESHOLD_CF
    if active_train.sum() == 0 or active_val.sum() == 0:
        raise ValueError(f"Group {config.group_id}: active-region subset is empty.")

    # keep 모드에서는 전부 1.0이라 가중치를 넘기지 않는 것과 동일하다.
    use_outage_weight = not np.allclose(outage_mult, 1.0)
    w_all = outage_mult if use_outage_weight else None
    w_active = outage_mult[active_train] if use_outage_weight else None

    candidate_predictions, records, models = {}, {}, {}

    direct_model = fit_lgb_regressor(config.group_id, x_adv_train, y_train, x_adv_val, y_val,
                                      config.capacity_kwh, sample_weight=w_all)
    models["direct_adv_lgb"] = direct_model
    candidate_predictions["direct_adv_lgb"] = clipped_cf_prediction(direct_model, x_adv_val)
    records["direct_adv_lgb"] = CandidateRecord("direct_adv_lgb", safe_best_iteration(direct_model))

    train_times = train_times_full
    recent_weight = np.where(train_times >= pd.Timestamp("2023-01-01 01:00:00"), 1.35, 1.0)
    active_weight = np.where(active_train, 2.2, 0.45)
    sample_weight = recent_weight * active_weight * outage_mult
    weighted_model = fit_lgb_regressor(config.group_id, x_adv_train, y_train, x_adv_val, y_val,
                                        config.capacity_kwh, sample_weight=sample_weight)
    models["weighted_adv_lgb"] = weighted_model
    candidate_predictions["weighted_adv_lgb"] = clipped_cf_prediction(weighted_model, x_adv_val)
    records["weighted_adv_lgb"] = CandidateRecord("weighted_adv_lgb", safe_best_iteration(weighted_model),
                                                   {"recent_multiplier": 1.35, "active_multiplier": 2.2,
                                                    "inactive_multiplier": 0.45})

    active_base_model = fit_lgb_regressor(config.group_id, x_base_train.loc[active_train], y_train[active_train],
                                           x_base_val.loc[active_val], y_val[active_val], config.capacity_kwh,
                                           active_validation_only=True, sample_weight=w_active)
    models["active_base_lgb"] = active_base_model
    candidate_predictions["active_base_lgb"] = clipped_cf_prediction(active_base_model, x_base_val)
    records["active_base_lgb"] = CandidateRecord("active_base_lgb", safe_best_iteration(active_base_model))

    active_adv_model = fit_lgb_regressor(config.group_id, x_adv_train.loc[active_train], y_train[active_train],
                                          x_adv_val.loc[active_val], y_val[active_val], config.capacity_kwh,
                                          active_validation_only=True, sample_weight=w_active)
    models["active_adv_lgb"] = active_adv_model
    candidate_predictions["active_adv_lgb"] = clipped_cf_prediction(active_adv_model, x_adv_val)
    records["active_adv_lgb"] = CandidateRecord("active_adv_lgb", safe_best_iteration(active_adv_model))

    quantile_lgb = fit_lgb_regressor(config.group_id, x_adv_train.loc[active_train], y_train[active_train],
                                      x_adv_val.loc[active_val], y_val[active_val], config.capacity_kwh,
                                      objective="quantile", alpha=0.58, active_validation_only=True,
                                      sample_weight=w_active)
    models["active_lgb_q58"] = quantile_lgb
    candidate_predictions["active_lgb_q58"] = clipped_cf_prediction(quantile_lgb, x_adv_val)
    records["active_lgb_q58"] = CandidateRecord("active_lgb_q58", safe_best_iteration(quantile_lgb), {"alpha": 0.58})

    cat_mae = fit_catboost(config.group_id, x_base_train.loc[active_train], y_train[active_train],
                            x_base_val.loc[active_val], y_val[active_val], loss_function="MAE",
                            sample_weight=w_active)
    models["active_cat_mae"] = cat_mae
    candidate_predictions["active_cat_mae"] = clipped_cf_prediction(cat_mae, x_base_val)
    records["active_cat_mae"] = CandidateRecord("active_cat_mae", safe_best_iteration(cat_mae))

    cat_q55 = fit_catboost(config.group_id, x_adv_train.loc[active_train], y_train[active_train],
                            x_adv_val.loc[active_val], y_val[active_val], loss_function="Quantile:alpha=0.55",
                            sample_weight=w_active)
    models["active_cat_q55"] = cat_q55
    candidate_predictions["active_cat_q55"] = clipped_cf_prediction(cat_q55, x_adv_val)
    records["active_cat_q55"] = CandidateRecord("active_cat_q55", safe_best_iteration(cat_q55), {"alpha": 0.55})

    # ---- 실험 6: 고출력 가중 MAE 모델 (기존 모델 대체가 아니라 후보 추가) ----
    if config.group_id in HIGH_OUTPUT_GROUPS:
        cf_train_active = y_train[active_train]
        high_output_w = 1.0 + HIGH_OUTPUT_LAMBDA * np.square(cf_train_active)
        if w_active is not None:
            high_output_w = high_output_w * w_active
        hi_model = fit_lgb_regressor(config.group_id, x_adv_train.loc[active_train], y_train[active_train],
                                      x_adv_val.loc[active_val], y_val[active_val], config.capacity_kwh,
                                      active_validation_only=True, sample_weight=high_output_w)
        models["active_hiout_lgb"] = hi_model
        candidate_predictions["active_hiout_lgb"] = clipped_cf_prediction(hi_model, x_adv_val)
        records["active_hiout_lgb"] = CandidateRecord("active_hiout_lgb", safe_best_iteration(hi_model),
                                                       {"lambda": HIGH_OUTPUT_LAMBDA, "form": "1+lambda*cf^2"})

    # ---- 실험 7: 그룹별 상위 Quantile 후보 추가 ----
    for alpha in EXTRA_LGB_QUANTILES.get(config.group_id, ()):
        name = f"active_lgb_q{int(round(alpha * 100))}"
        qmodel = fit_lgb_regressor(config.group_id, x_adv_train.loc[active_train], y_train[active_train],
                                    x_adv_val.loc[active_val], y_val[active_val], config.capacity_kwh,
                                    objective="quantile", alpha=alpha, active_validation_only=True,
                                    sample_weight=w_active)
        models[name] = qmodel
        candidate_predictions[name] = clipped_cf_prediction(qmodel, x_adv_val)
        records[name] = CandidateRecord(name, safe_best_iteration(qmodel), {"alpha": alpha})

    active_classifier = fit_lgb_classifier(config.group_id, x_adv_train, active_train.astype(np.int8),
                                            x_adv_val, active_val.astype(np.int8))
    models["active_classifier"] = active_classifier
    active_probability = active_classifier.predict_proba(x_adv_val)[:, 1]
    guided, floor, gamma, guided_metrics = select_guided_hurdle(
        y_val_kwh, config.capacity_kwh, candidate_predictions["direct_adv_lgb"],
        candidate_predictions["active_adv_lgb"], active_probability)
    candidate_predictions["guided_hurdle"] = guided
    records["guided_hurdle"] = CandidateRecord("guided_hurdle", None,
                                                {"floor": floor, "gamma": gamma, "metrics": guided_metrics})
    records["active_classifier"] = CandidateRecord("active_classifier", safe_best_iteration(active_classifier))

    for name, prediction in pooled_predictions.items():
        candidate_predictions[name] = np.asarray(prediction, dtype=float)
        records[name] = CandidateRecord(name)

    if config.group_id == 3 and USE_GROUP3_GATE:
        expert_name = GROUP3_GATE_EXPERT
        if expert_name in candidate_predictions:
            val_expert_cf = candidate_predictions[expert_name]
        else:
            val_expert_cf = candidate_predictions["active_cat_q55"]
        val_clf, val_scaler = fit_high85_classifier(x_adv_train, y_train)
        val_high85 = predict_high85_probability(val_clf, val_scaler, x_adv_val)
    else:
        val_expert_cf, val_high85 = None, None

    auxiliary = {
        "val_expert_cf": val_expert_cf,
        "val_high85_probability": val_high85,
        "models": models, "base_medians": base_medians, "advanced_medians": adv_medians,
        "base_columns": list(x_base_train.columns), "advanced_columns": list(x_adv_train.columns),
        "active_probability": active_probability, "guided_floor": floor, "guided_gamma": gamma,
        "y_val_cf": y_val, "y_val_kwh": y_val_kwh,
        "val_timestamps": timestamps.loc[val_mask].reset_index(drop=True),
        "train_mask": train_mask, "val_mask": val_mask,
    }
    return candidate_predictions, records, auxiliary, {
        "generation_active_auc": float(roc_auc_score(active_val, active_probability))}


def _full_lgb_params(group_id, record, objective, alpha=None):
    params = lgb_base_params(group_id)
    params.update({"objective": objective, "n_estimators": max(1, int(record.best_iteration or 1000))})
    if alpha is not None:
        params["alpha"] = alpha
    return params


def refit_group_and_predict(train_frame, test_frame, config, recipe, pooled_test_predictions):
    y_full = train_frame[config.target_col].to_numpy(dtype=float) / config.capacity_kwh
    active_full = y_full >= ACTIVE_THRESHOLD_CF

    base_train_raw = numeric_feature_frame(train_frame, config.target_col)
    base_test_raw = numeric_feature_frame(test_frame, None)
    x_base, x_base_test, _ = align_features(base_train_raw, base_test_raw)

    adv_train_raw = build_advanced_features(train_frame, config, config.target_col)
    adv_test_raw = build_advanced_features(test_frame, config, None)
    x_adv, x_adv_test, _ = align_features(adv_train_raw, adv_test_raw)

    # ---- 실험 9: 검증 단계와 동일한 장기정지 처리를 재학습에도 적용 ----
    all_train = np.ones(len(train_frame), dtype=bool)
    keep_mask, outage_mult = outage_training_adjustment(train_frame, all_train)
    full_times_all = pd.to_datetime(train_frame["forecast_kst_dtm"], errors="raise").reset_index(drop=True)

    if not keep_mask.all():
        x_base = x_base.loc[keep_mask]
        x_adv = x_adv.loc[keep_mask]
        y_full = y_full[keep_mask]
        active_full = active_full[keep_mask]
        outage_mult = outage_mult[keep_mask]
        full_times_all = full_times_all.loc[keep_mask].reset_index(drop=True)

    use_outage_weight = not np.allclose(outage_mult, 1.0)
    w_all_full = outage_mult if use_outage_weight else None
    w_active_full = outage_mult[active_full] if use_outage_weight else None

    records = recipe.candidate_records
    predictions, fitted_models = {}, {}
    n_test = len(test_frame)

    # 가중치가 정확히 0인 후보는 재학습을 생략한다 (결과는 동일, 시간만 절약).
    # guided_hurdle은 내부적으로 direct_adv_lgb/active_adv_lgb/active_classifier가 필요하다.
    needed = {name for name, weight in recipe.component_weights.items() if weight != 0.0}
    if "guided_hurdle" in needed:
        needed.update({"direct_adv_lgb", "active_adv_lgb", "active_classifier"})

    if "direct_adv_lgb" in needed:
        direct = lgb.LGBMRegressor(**_full_lgb_params(config.group_id, records["direct_adv_lgb"], "regression_l1"))
        direct.fit(x_adv, y_full, sample_weight=w_all_full)
        fitted_models["direct_adv_lgb"] = direct
        predictions["direct_adv_lgb"] = clipped_cf_prediction(direct, x_adv_test)
    else:
        predictions["direct_adv_lgb"] = np.zeros(n_test, dtype=float)

    if "weighted_adv_lgb" in needed:
        recent_weight = np.where(full_times_all >= pd.Timestamp("2023-01-01 01:00:00"), 1.35, 1.0)
        sample_weight = recent_weight * np.where(active_full, 2.2, 0.45) * outage_mult
        weighted = lgb.LGBMRegressor(**_full_lgb_params(config.group_id, records["weighted_adv_lgb"], "regression_l1"))
        weighted.fit(x_adv, y_full, sample_weight=sample_weight)
        fitted_models["weighted_adv_lgb"] = weighted
        predictions["weighted_adv_lgb"] = clipped_cf_prediction(weighted, x_adv_test)
    else:
        predictions["weighted_adv_lgb"] = np.zeros(n_test, dtype=float)

    if "active_base_lgb" in needed:
        active_base = lgb.LGBMRegressor(**_full_lgb_params(config.group_id, records["active_base_lgb"], "regression_l1"))
        active_base.fit(x_base.loc[active_full], y_full[active_full], sample_weight=w_active_full)
        fitted_models["active_base_lgb"] = active_base
        predictions["active_base_lgb"] = clipped_cf_prediction(active_base, x_base_test)
    else:
        predictions["active_base_lgb"] = np.zeros(n_test, dtype=float)

    if "active_adv_lgb" in needed:
        active_adv = lgb.LGBMRegressor(**_full_lgb_params(config.group_id, records["active_adv_lgb"], "regression_l1"))
        active_adv.fit(x_adv.loc[active_full], y_full[active_full], sample_weight=w_active_full)
        fitted_models["active_adv_lgb"] = active_adv
        predictions["active_adv_lgb"] = clipped_cf_prediction(active_adv, x_adv_test)
    else:
        predictions["active_adv_lgb"] = np.zeros(n_test, dtype=float)

    if "active_lgb_q58" in needed:
        q58 = lgb.LGBMRegressor(**_full_lgb_params(config.group_id, records["active_lgb_q58"], "quantile", alpha=0.58))
        q58.fit(x_adv.loc[active_full], y_full[active_full], sample_weight=w_active_full)
        fitted_models["active_lgb_q58"] = q58
        predictions["active_lgb_q58"] = clipped_cf_prediction(q58, x_adv_test)
    else:
        predictions["active_lgb_q58"] = np.zeros(n_test, dtype=float)

    if "active_cat_mae" in needed:
        cat_mae_record = records["active_cat_mae"]
        cat_mae_params = catboost_params(config.group_id, "MAE")
        cat_mae_params.update({"iterations": max(1, int(cat_mae_record.best_iteration or 1200)),
                                "od_type": None, "od_wait": None})
        cat_mae_params = {k: v for k, v in cat_mae_params.items() if v is not None}
        cat_mae = CatBoostRegressor(**cat_mae_params)
        cat_mae.fit(x_base.loc[active_full], y_full[active_full], sample_weight=w_active_full, verbose=False)
        fitted_models["active_cat_mae"] = cat_mae
        predictions["active_cat_mae"] = clipped_cf_prediction(cat_mae, x_base_test)
    else:
        predictions["active_cat_mae"] = np.zeros(n_test, dtype=float)

    if "active_cat_q55" in needed:
        cat_q_record = records["active_cat_q55"]
        cat_q_params = catboost_params(config.group_id, "Quantile:alpha=0.55")
        cat_q_params.update({"iterations": max(1, int(cat_q_record.best_iteration or 1200)),
                              "od_type": None, "od_wait": None})
        cat_q_params = {k: v for k, v in cat_q_params.items() if v is not None}
        cat_q = CatBoostRegressor(**cat_q_params)
        cat_q.fit(x_adv.loc[active_full], y_full[active_full], sample_weight=w_active_full, verbose=False)
        fitted_models["active_cat_q55"] = cat_q
        predictions["active_cat_q55"] = clipped_cf_prediction(cat_q, x_adv_test)
    else:
        predictions["active_cat_q55"] = np.zeros(n_test, dtype=float)

    # ---- 실험 6: 고출력 가중 모델 재학습 ----
    if "active_hiout_lgb" in needed and config.group_id in HIGH_OUTPUT_GROUPS:
        hi_w = 1.0 + HIGH_OUTPUT_LAMBDA * np.square(y_full[active_full])
        if w_active_full is not None:
            hi_w = hi_w * w_active_full
        hi = lgb.LGBMRegressor(**_full_lgb_params(config.group_id, records["active_hiout_lgb"], "regression_l1"))
        hi.fit(x_adv.loc[active_full], y_full[active_full], sample_weight=hi_w)
        fitted_models["active_hiout_lgb"] = hi
        predictions["active_hiout_lgb"] = clipped_cf_prediction(hi, x_adv_test)
    elif config.group_id in HIGH_OUTPUT_GROUPS:
        predictions["active_hiout_lgb"] = np.zeros(n_test, dtype=float)

    # ---- 실험 7: 추가 Quantile 후보 재학습 ----
    for alpha in EXTRA_LGB_QUANTILES.get(config.group_id, ()):
        name = f"active_lgb_q{int(round(alpha * 100))}"
        if name in needed:
            qm = lgb.LGBMRegressor(**_full_lgb_params(config.group_id, records[name], "quantile", alpha=alpha))
            qm.fit(x_adv.loc[active_full], y_full[active_full], sample_weight=w_active_full)
            fitted_models[name] = qm
            predictions[name] = clipped_cf_prediction(qm, x_adv_test)
        else:
            predictions[name] = np.zeros(n_test, dtype=float)

    if "active_classifier" in needed or "guided_hurdle" in needed:
        clf_record = records["active_classifier"]
        clf_params = lgb_base_params(config.group_id)
        clf_params.update({"objective": "binary", "n_estimators": max(1, int(clf_record.best_iteration or 900)),
                            "learning_rate": 0.025})
        classifier = lgb.LGBMClassifier(**clf_params)
        classifier.fit(x_adv, active_full.astype(np.int8), sample_weight=w_all_full)
        fitted_models["active_classifier"] = classifier
        probability = classifier.predict_proba(x_adv_test)[:, 1]
        trust = recipe.guided_floor + (1.0 - recipe.guided_floor) * np.power(probability, recipe.guided_gamma)
        predictions["guided_hurdle"] = np.clip(
            trust * predictions["active_adv_lgb"] + (1.0 - trust) * predictions["direct_adv_lgb"],
            0.0, PREDICTION_MAX_CF)
    else:
        predictions["guided_hurdle"] = np.zeros(n_test, dtype=float)

    for name, prediction in pooled_test_predictions.items():
        predictions[name] = np.asarray(prediction, dtype=float)

    missing_components = set(recipe.component_weights) - set(predictions)
    if missing_components:
        raise KeyError(f"Missing test predictions for components: {sorted(missing_components)}")

    final_cf = np.zeros(len(test_frame), dtype=float)
    for name, weight in recipe.component_weights.items():
        final_cf += weight * predictions[name]
    final_cf = np.clip(final_cf * recipe.calibration_scale + recipe.calibration_offset_cf, 0.0, PREDICTION_MAX_CF)
    final_kwh = np.clip(final_cf * config.capacity_kwh, 0.0, config.capacity_kwh * PREDICTION_MAX_CF)

    # ---- 후처리 입력 준비 ----
    # base_cf: 후처리 이전의 앙상블 예측 (Group1 affine 등은 postprocess에서 적용)
    extras = {"base_cf": final_cf}

    if config.group_id == 3 and USE_GROUP3_GATE:
        # Gate 전문가(q70)는 앙상블 가중치가 0이어도 반드시 필요하므로
        # 선택 여부와 무관하게 전체 데이터로 학습해 예측을 만든다.
        expert_alpha = 0.70
        expert_name = GROUP3_GATE_EXPERT
        if expert_name in predictions and np.any(predictions[expert_name] > 0):
            expert_cf = predictions[expert_name]
        else:
            expert_params = lgb_base_params(config.group_id)
            expert_params.update({
                "objective": "quantile",
                "alpha": expert_alpha,
                "n_estimators": max(1, int(records[expert_name].best_iteration or 1000))
                if expert_name in records else 1000,
            })
            expert_model = lgb.LGBMRegressor(**expert_params)
            expert_model.fit(x_adv.loc[active_full], y_full[active_full], sample_weight=w_active_full)
            fitted_models["group3_gate_expert_q70"] = expert_model
            expert_cf = clipped_cf_prediction(expert_model, x_adv_test)

        # High-85 분류기는 기상 피처만 사용하므로 OOF가 필요 없다.
        high85_clf, high85_scaler = fit_high85_classifier(x_adv, y_full)
        fitted_models["group3_high85_classifier"] = high85_clf
        fitted_models["group3_high85_scaler"] = high85_scaler

        extras["expert_cf"] = expert_cf
        extras["high85_probability"] = predict_high85_probability(
            high85_clf, high85_scaler, x_adv_test
        )

    return final_kwh, fitted_models, extras


def save_group_models(models, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        if isinstance(model, CatBoostRegressor):
            model.save_model(model_dir / f"{name}.cbm")
        else:
            joblib.dump(model, model_dir / f"{name}.joblib")


def build_submission(sample_submission_path, test_frames, group_predictions_kwh):
    sample = pd.read_csv(sample_submission_path, encoding="utf-8-sig")
    required = {"forecast_id", "forecast_kst_dtm"}
    if not required.issubset(sample.columns):
        raise KeyError("sample_submission is missing columns")
    sample["forecast_kst_dtm"] = pd.to_datetime(sample["forecast_kst_dtm"], errors="raise")
    result = sample[["forecast_id", "forecast_kst_dtm"]].copy()
    for group_id, config in GROUPS.items():
        frame = test_frames[group_id]
        prediction = group_predictions_kwh[group_id]
        if len(frame) != len(prediction):
            raise ValueError(f"Group {group_id}: row count mismatch.")
        pred_frame = pd.DataFrame({"forecast_kst_dtm": frame["forecast_kst_dtm"].to_numpy(),
                                    config.target_col: prediction})
        result = result.merge(pred_frame, on="forecast_kst_dtm", how="left", validate="one_to_one")
    target_cols = [GROUPS[g].target_col for g in GROUPS]
    if result[target_cols].isna().any().any():
        raise ValueError("Submission contains missing predictions")
    return result[["forecast_id", "forecast_kst_dtm", *target_cols]]


def run_pipeline(preprocessed_dir, sample_submission_path, output_dir, *, make_submission, save_models):
    output_dir.mkdir(parents=True, exist_ok=True)

    if make_submission and not Path(sample_submission_path).is_file():
        raise FileNotFoundError(
            "sample_submission 파일을 찾을 수 없습니다: "
            f"{Path(sample_submission_path).resolve()}\n"
            "학습을 시작하기 전에 경로를 확인하세요."
        )

    train_frames = {g: read_group_csv(preprocessed_dir, "train", c) for g, c in GROUPS.items()}
    test_frames = {}
    if make_submission:
        test_frames = {g: read_group_csv(preprocessed_dir, "test", c) for g, c in GROUPS.items()}

    print("[1/4] Training pooled validation models...", flush=True)
    (pooled_validation_models, pooled_val_predictions, pooled_columns,
     pooled_medians, pooled_iterations) = fit_pooled_validation_models(train_frames)

    group_recipes, validation_rows, validation_prediction_frames = {}, [], []

    print("[2/4] Training group candidates and selecting constrained ensembles...", flush=True)
    for group_id, config in GROUPS.items():
        print(f"  Group {group_id}", flush=True)
        pooled_for_group = {n: p[group_id] for n, p in pooled_val_predictions.items()}
        candidates, records, auxiliary, diagnostics = train_group_candidates(
            train_frames[group_id], config, pooled_for_group)
        weights, ensemble_cf, blend_metrics, component_metrics = optimize_constrained_blend(
            auxiliary["y_val_kwh"], config.capacity_kwh, candidates)
        scale, offset, calibrated_cf, calibration_details = conservative_calibration(
            auxiliary["val_timestamps"], auxiliary["y_val_kwh"], ensemble_cf, config.capacity_kwh)
        final_metrics = official_metrics(auxiliary["y_val_kwh"], calibrated_cf * config.capacity_kwh,
                                          config.capacity_kwh)
        recipe = GroupRecipe(group_id, weights, records, float(auxiliary["guided_floor"]),
                              float(auxiliary["guided_gamma"]), scale, offset, final_metrics,
                              auxiliary["base_columns"], auxiliary["advanced_columns"])
        recipe.validation_base_cf = ensemble_cf
        if group_id == 3 and USE_GROUP3_GATE:
            recipe.validation_expert_cf = auxiliary.get("val_expert_cf")
            recipe.validation_high85_probability = auxiliary.get("val_high85_probability")

            if recipe.validation_expert_cf is not None and recipe.validation_high85_probability is not None:
                gate_params, gate_diag = tune_group3_gate(
                    timestamps=auxiliary["val_timestamps"],
                    actual_cf=auxiliary["y_val_cf"],
                    base_cf=ensemble_cf,
                    expert_cf=recipe.validation_expert_cf,
                    probability=recipe.validation_high85_probability,
                    capacity_kwh=config.capacity_kwh,
                )
                recipe.group3_gate_params = gate_params
                print(f"    [Group3 Gate] {'채택' if gate_diag.get('accepted') else '미채택'} "
                      f"- {gate_diag.get('reason', '')}", flush=True)
                if gate_diag.get("accepted"):
                    print(f"      threshold={gate_diag['threshold']:.2f} gamma={gate_diag['gamma']:.2f} "
                          f"eta={gate_diag['eta']:.2f}", flush=True)
                    print(f"      H1 개선 {gate_diag['h1_gain']:+.6f} / "
                          f"H2 독립검증 {gate_diag['h2_before']:.6f} -> {gate_diag['h2_after']:.6f} "
                          f"({gate_diag['h2_gain']:+.6f})", flush=True)
                    print(f"      발동 {gate_diag['fired_rows']}행 "
                          f"({gate_diag['fired_ratio']*100:.2f}%)", flush=True)
                (output_dir / "group3_gate_diagnostics.json").write_text(
                    json.dumps(gate_diag, ensure_ascii=False, indent=2), encoding="utf-8")
        group_recipes[group_id] = recipe
        validation_rows.append({"group": group_id, **final_metrics,
                                 "eval_rows": int(official_mask_kwh(auxiliary["y_val_kwh"], config.capacity_kwh).sum()),
                                 "active_classifier_auc": diagnostics["generation_active_auc"]})
        validation_prediction_frames.append(pd.DataFrame({
            "group": group_id, "forecast_kst_dtm": auxiliary["val_timestamps"],
            "actual_kwh": auxiliary["y_val_kwh"], "prediction_kwh": calibrated_cf * config.capacity_kwh,
            "official_eval_row": official_mask_kwh(auxiliary["y_val_kwh"], config.capacity_kwh)}))
        print(f"    NMAE={final_metrics['nmae']:.6f}  1-NMAE={final_metrics['one_minus_nmae']:.6f}  "
              f"FICR={final_metrics['ficr']:.6f}  Score={final_metrics['official_score']:.6f}", flush=True)
        details = {"component_metrics": component_metrics, "selected_weights": weights,
                   "blend_metrics_before_calibration": blend_metrics, "calibration": calibration_details}
        (output_dir / f"group{group_id}_selection_details.json").write_text(
            json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

    metric_frame = pd.DataFrame(validation_rows)
    mean_row = {"group": "mean", "nmae": float(metric_frame["nmae"].mean()),
                "one_minus_nmae": float(metric_frame["one_minus_nmae"].mean()),
                "ficr": float(metric_frame["ficr"].mean()),
                "official_score": float(metric_frame["official_score"].mean()),
                "eval_rows": int(metric_frame["eval_rows"].sum()),
                "active_classifier_auc": float(metric_frame["active_classifier_auc"].mean())}
    metric_frame = pd.concat([metric_frame, pd.DataFrame([mean_row])], ignore_index=True)
    metric_frame.to_csv(output_dir / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(validation_prediction_frames, ignore_index=True).to_csv(
        output_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S")

    evaluate_postprocess_on_validation(train_frames, group_recipes, output_dir)

    result = {"validation_metrics": metric_frame.to_dict(orient="records"), "recipes": group_recipes}
    if not make_submission:
        return result

    print("[3/4] Refitting pooled and group models on all available labels...", flush=True)
    pooled_full_models, pooled_test_predictions = fit_pooled_full_models(
        train_frames, test_frames, pooled_columns, pooled_medians, pooled_iterations)

    group_predictions, fitted_group_models = {}, {}
    postprocess_inputs = {}
    for group_id, config in GROUPS.items():
        pooled_for_group = {n: p[group_id] for n, p in pooled_test_predictions.items()}
        prediction, fitted, extras = refit_group_and_predict(
            train_frames[group_id], test_frames[group_id],
            config, group_recipes[group_id], pooled_for_group)
        group_predictions[group_id] = prediction
        fitted_group_models[group_id] = fitted
        postprocess_inputs[group_id] = extras

    # ---- 후처리 적용 ----
    # 모든 그룹의 base 예측이 준비된 뒤에 한 번에 적용한다.
    # Group1 그룹간 보정이 Group2 base를 참조하므로 순서가 중요하다.
    reference_times = test_frames[1]["forecast_kst_dtm"]
    base_cf_by_group = {g: postprocess_inputs[g]["base_cf"] for g in GROUPS}

    g3_extras = postprocess_inputs.get(3, {})
    postprocessed_kwh = apply_postprocess_to_groups(
        timestamps=reference_times,
        base_cf_by_group=base_cf_by_group,
        group3_expert_cf=g3_extras.get("expert_cf"),
        group3_high85_probability=g3_extras.get("high85_probability"),
        use_group3_gate=USE_GROUP3_GATE,
        group3_gate_params=group_recipes[3].group3_gate_params,
    )

    print("  후처리 적용 완료 "
          f"(Group3 Gate: {'ON' if group_recipes[3].group3_gate_params is not None else 'OFF'})",
          flush=True)

    for group_id in GROUPS:
        group_predictions[group_id] = postprocessed_kwh[group_id]

    # 제출 파일 생성 전에 예측값을 체크포인트로 저장한다.
    # sample_submission 경로 문제 등으로 build_submission이 실패해도
    # 여기까지의 재학습 결과(가장 오래 걸리는 부분)는 보존된다.
    checkpoint_path = output_dir / "test_predictions.joblib"
    joblib.dump(
        {f"group{group_id}": prediction for group_id, prediction in group_predictions.items()},
        checkpoint_path,
    )
    print(f"Test predictions saved to: {checkpoint_path}", flush=True)

    print("[4/4] Building submission...", flush=True)
    submission = build_submission(sample_submission_path, test_frames, group_predictions)
    submission_path = output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
    if save_models:
        for group_id, models in fitted_group_models.items():
            save_group_models(models, output_dir / "models" / f"group{group_id}")
        save_group_models(pooled_full_models, output_dir / "models" / "pooled")
    result["submission_path"] = str(submission_path)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-dir", type=Path, default=Path("/content/preprocessed"))
    parser.add_argument("--sample-submission", type=Path, default=Path("/content/sample_submission.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("/content/best_model_outputs"))
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--outage-mode", choices=["delete", "keep", "weight"], default=OUTAGE_MODE,
                        help="장기정지 시간 처리: delete(기존) / keep(9-A) / weight(9-B)")
    parser.add_argument("--outage-weight", type=float, default=OUTAGE_WEIGHT,
                        help="--outage-mode weight일 때 장기정지 행에 적용할 가중치")
    return parser.parse_args()


def main():
    global OUTAGE_MODE, OUTAGE_WEIGHT
    args = parse_args()
    OUTAGE_MODE = args.outage_mode
    OUTAGE_WEIGHT = args.outage_weight
    print(f"[설정] 장기정지 처리 모드: {OUTAGE_MODE}"
          + (f" (weight={OUTAGE_WEIGHT})" if OUTAGE_MODE == "weight" else ""), flush=True)
    result = run_pipeline(preprocessed_dir=args.preprocessed_dir,
                          sample_submission_path=args.sample_submission,
                          output_dir=args.output_dir,
                          make_submission=not args.validation_only,
                          save_models=args.save_models)
    print("\n=== Official 2024 validation ===")
    for row in result["validation_metrics"]:
        print(f"Group {row['group']}: NMAE={row['nmae']:.6f}, 1-NMAE={row['one_minus_nmae']:.6f}, "
              f"FICR={row['ficr']:.6f}, Score={row['official_score']:.6f}")
    if "submission_path" in result:
        print(f"\nSubmission saved to: {result['submission_path']}")



# =============================================================================
# 후처리 (postprocess)
#
# 모든 계산은 Capacity Factor 단위에서 수행하고 마지막에만 kWh로 변환한다.
#
# 2024 검증에서 직접 재현한 결과 (동일 검증셋 기준)
#   기준(보정 없음)                    Score 0.644922
#   + G1 affine                       Score 0.645587
#   + G1 구간 offset                   Score 0.646954
#   + G1 그룹간보정 + 평활화            Score 0.648197   <- 채택 구성
#
# 예보 회차 경계는 01:00 ~ 익일 00:00 이다.
# =============================================================================

# Group1 분산 확대 (평균 보존 방식).
#
# 이전에는 고정 affine(1.08 * cf - 0.025)을 썼으나 2025 제출에서 실패했다.
# 원인: 그 상수는 2024 예측 분포에 맞춰진 값인데, 2025는 풍속이 6~8% 강해
# 예측 분포 자체가 위로 이동해 있었다. 그 결과 같은 규칙이 2025에서는
# 훨씬 많은 행을 위로 밀어 과보정이 됐다 (평균 이동 2024 +0.64%p vs 2025 +1.03%p).
# 실제로 nMAE와 FICR이 동시에 악화됐는데, 이는 과보정의 전형적 증상이다.
#
# 따라서 '자기 예측 평균'을 축으로 분산만 확대하는 방식으로 바꾼다.
# 평균이 보존되므로 예측 분포가 어디로 이동하든 상대적으로 동일하게 작동한다.
# 2024 검증에서도 고정 affine보다 오히려 좋았다 (0.660414 vs 0.659645).
GROUP1_EXPANSION_K = 1.13

# Group1 예측 CF 구간별 offset.
# 구간은 반드시 '보정 후 예측값'으로 결정한다 (실제값 사용은 누수).
# 적용된 offset의 평균을 빼서 평균 중립으로 만든다. 원본 offset은
# 전체적으로 위로 미는 효과가 있어 분포 이동에 취약하기 때문이다.
GROUP1_BIN_EDGES = [0.25, 0.50, 0.75]
GROUP1_BIN_OFFSETS = np.array([0.000, -0.015, 0.015, 0.015])
GROUP1_BIN_MEAN_NEUTRAL = True

# 후처리 전체의 평균 이동 허용치 (CF).
# 어떤 이유로든 후처리가 예측 평균을 이보다 크게 움직이면 되돌린다.
# 2025 실패가 평균이 +1.03%p 밀린 것이었으므로 마지막 안전장치로 둔다.
POSTPROCESS_MEAN_SHIFT_TOLERANCE_CF = 0.005

# Group1 그룹간 보정. G1·G2는 같은 단지 VESTAS 그룹이고 실제 CF 상관이 0.962다.
CROSS_DIFF_SCALE_CF = 0.20
CROSS_MOVE_RATIO = 0.25

# Group1 예보 회차 내부 평활화
SMOOTH_WINDOW = 5
SMOOTH_BLEND = 0.30

# Group3 High-85 Settlement Gate
USE_GROUP3_GATE = True
HIGH85_TARGET_CF = 0.85
HIGH85_PROB_THRESHOLD = 0.50
HIGH85_GAMMA = 0.50
HIGH85_ETA = 1.50
HIGH85_CORRECTION_CAP_CF = 0.15
GROUP3_GATE_EXPERT = "active_lgb_q70"


def forecast_cycle_key(timestamps):
    """예보 회차 식별자. 한 회차는 01:00 ~ 익일 00:00 이다."""
    ts = pd.to_datetime(timestamps, errors="raise")
    return (ts - pd.Timedelta(hours=1)).dt.normalize()


def cycle_centered_rolling_mean(values, cycle_key, window):
    """예보 회차 경계를 넘지 않는 중심 이동평균."""
    series = pd.Series(np.asarray(values, dtype=float))
    return (
        series.groupby(np.asarray(cycle_key))
        .transform(lambda s: s.rolling(window, center=True, min_periods=1).mean())
        .to_numpy()
    )


def apply_group1_postprocess(base_cf, group2_reference_cf, timestamps):
    """
    Group1 후처리.
        affine -> CF 구간 offset -> Group2 방향 그룹간 보정 -> 회차내 평활화

    group2_reference_cf에는 반드시 '보정 전' Group2 예측을 넣는다.
    서로의 보정 결과를 참조하면 순환 구조가 되어 검증과 달라진다.
    """
    base_cf = np.asarray(base_cf, dtype=float)
    base_mean = float(base_cf.mean())

    # 평균 보존 분산 확대
    cf = np.clip(base_mean + GROUP1_EXPANSION_K * (base_cf - base_mean),
                 0.0, PREDICTION_MAX_CF)

    # 구간별 offset (평균 중립화)
    bin_index = np.digitize(cf, bins=GROUP1_BIN_EDGES, right=True)
    offsets = GROUP1_BIN_OFFSETS[bin_index]
    if GROUP1_BIN_MEAN_NEUTRAL:
        offsets = offsets - offsets.mean()
    cf = np.clip(cf + offsets, 0.0, PREDICTION_MAX_CF)

    reference = np.asarray(group2_reference_cf, dtype=float)
    difference = reference - cf
    trust = np.clip(np.abs(difference) / CROSS_DIFF_SCALE_CF, 0.0, 1.0)
    cf = np.clip(cf + CROSS_MOVE_RATIO * trust * difference, 0.0, PREDICTION_MAX_CF)

    smoothed = cycle_centered_rolling_mean(cf, forecast_cycle_key(timestamps), SMOOTH_WINDOW)
    cf = np.clip((1.0 - SMOOTH_BLEND) * cf + SMOOTH_BLEND * smoothed,
                 0.0, PREDICTION_MAX_CF)

    # 평균 이동 가드: 허용치를 넘게 평균이 밀렸으면 되돌린다.
    shift = float(cf.mean()) - base_mean
    if abs(shift) > POSTPROCESS_MEAN_SHIFT_TOLERANCE_CF:
        correction = shift - np.sign(shift) * POSTPROCESS_MEAN_SHIFT_TOLERANCE_CF
        cf = np.clip(cf - correction, 0.0, PREDICTION_MAX_CF)

    return cf


def apply_group3_high85_gate(base_cf, expert_cf, high85_probability):
    """
    Group3 High-85 Settlement Gate.

    고출력 확률이 충분히 높은 시각에만 expert(q70) 방향으로 제한적으로 상승시킨다.
    expert가 base보다 낮은 시각에는 낮추지 않는다(상향 전용).

    주의: 2024 검증에서 보정된 285행 중 253행(89%)이 7월에 집중돼 있었다.
    이득이 특정 계절 조건에 의존하므로 2025년에는 거의 작동하지 않을 수 있다.
    다만 상한(CF 0.15)과 상향 전용 제약 때문에 하방 위험은 제한적이다.
    """
    base_cf = np.asarray(base_cf, dtype=float)
    expert_cf = np.asarray(expert_cf, dtype=float)
    probability = np.asarray(high85_probability, dtype=float)

    high_weight = np.clip(
        (probability - HIGH85_PROB_THRESHOLD) / (1.0 - HIGH85_PROB_THRESHOLD),
        0.0, 1.0,
    ) ** HIGH85_GAMMA

    upward_gap = np.maximum(expert_cf - base_cf, 0.0)
    correction = np.clip(HIGH85_ETA * high_weight * upward_gap,
                         0.0, HIGH85_CORRECTION_CAP_CF)

    return np.clip(base_cf + correction, 0.0, PREDICTION_MAX_CF)


def fit_high85_classifier(x_train, y_train_cf):
    """
    실제 CF >= 0.85 여부를 예측하는 분류기.

    입력을 기상 피처로만 제한한다. 모델 예측값을 입력에 넣지 않으므로
    OOF 예측이 필요 없고 누수 위험도 없다.
    """
    target = (np.asarray(y_train_cf, dtype=float) >= HIGH85_TARGET_CF).astype(int)

    if target.sum() < 30 or target.sum() == len(target):
        return None, None

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)

    classifier = LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_SEED)
    classifier.fit(x_scaled, target)
    return classifier, scaler


def predict_high85_probability(classifier, scaler, x_features):
    if classifier is None or scaler is None:
        return np.zeros(len(x_features), dtype=float)
    return classifier.predict_proba(scaler.transform(x_features))[:, 1]


def apply_postprocess_to_groups(timestamps, base_cf_by_group,
                                group3_expert_cf=None,
                                group3_high85_probability=None,
                                use_group3_gate=USE_GROUP3_GATE,
                                group3_gate_params=None):
    """
    세 그룹 base 예측(CF)에 후처리를 적용해 kWh 예측을 만든다.
    Group2는 어떤 후처리도 적용하지 않는다(검증에서 모든 변경이 실패했다).
    """
    g1_ref = np.asarray(base_cf_by_group[1], dtype=float)
    g2_ref = np.asarray(base_cf_by_group[2], dtype=float)
    g3_ref = np.asarray(base_cf_by_group[3], dtype=float)

    group1_cf = apply_group1_postprocess(g1_ref, g2_ref, timestamps)
    group2_cf = np.clip(g2_ref, 0.0, PREDICTION_MAX_CF)

    gate_ready = (use_group3_gate and group3_expert_cf is not None
                  and group3_high85_probability is not None
                  and group3_gate_params is not None)
    if gate_ready:
        group3_cf = apply_group3_gate_with_params(
            g3_ref, group3_expert_cf, group3_high85_probability, group3_gate_params)
    else:
        group3_cf = np.clip(g3_ref, 0.0, PREDICTION_MAX_CF)

    return {
        1: group1_cf * GROUPS[1].capacity_kwh,
        2: group2_cf * GROUPS[2].capacity_kwh,
        3: group3_cf * GROUPS[3].capacity_kwh,
    }


def evaluate_postprocess_on_validation(train_frames, group_recipes, output_dir):
    """
    후처리를 '적용한' 상태의 2024 검증 지표를 계산해 출력한다.

    기존 validation_metrics.csv는 후처리 이전 수치라 후처리가 실제로
    도움이 되는지 확인할 수 없었다. 2025 제출이 검증과 반대로 나온 뒤
    이를 사후에 알아차렸으므로, 실행 결과에서 바로 보이도록 추가한다.
    """
    val_frames, base_cf, actual_cf, timestamps = {}, {}, {}, None

    for group_id, config in GROUPS.items():
        frame = train_frames[group_id]
        _, val_mask = split_masks(frame["forecast_kst_dtm"])
        val_frames[group_id] = frame.loc[val_mask].reset_index(drop=True)

    # 세 그룹의 검증 시각이 동일해야 그룹간 보정이 성립한다.
    reference = val_frames[1]["forecast_kst_dtm"].to_numpy()
    for group_id in GROUPS:
        if not np.array_equal(val_frames[group_id]["forecast_kst_dtm"].to_numpy(), reference):
            print("  [경고] 그룹별 검증 시각이 달라 후처리 검증을 건너뜁니다.", flush=True)
            return None

    timestamps = val_frames[1]["forecast_kst_dtm"]

    for group_id, config in GROUPS.items():
        recipe = group_recipes[group_id]
        preds = recipe.candidate_records
        cf = recipe.validation_base_cf
        if cf is None:
            print("  [경고] 검증 base 예측이 없어 후처리 검증을 건너뜁니다.", flush=True)
            return None
        base_cf[group_id] = cf
        actual_cf[group_id] = (
            val_frames[group_id][config.target_col].to_numpy(dtype=float) / config.capacity_kwh
        )

    g3_expert = group_recipes[3].validation_expert_cf
    g3_prob = group_recipes[3].validation_high85_probability
    g3_params = group_recipes[3].group3_gate_params
    gate_on = USE_GROUP3_GATE and g3_expert is not None and g3_prob is not None and g3_params is not None

    post_kwh = apply_postprocess_to_groups(
        timestamps=timestamps,
        base_cf_by_group=base_cf,
        group3_expert_cf=g3_expert,
        group3_high85_probability=g3_prob,
        use_group3_gate=gate_on,
        group3_gate_params=g3_params,
    )

    rows = []
    print("\n=== 후처리 적용 후 검증 지표 (2024) ===", flush=True)
    for group_id, config in GROUPS.items():
        cap = config.capacity_kwh
        y_kwh = actual_cf[group_id] * cap
        before = official_metrics(y_kwh, base_cf[group_id] * cap, cap)
        after = official_metrics(y_kwh, post_kwh[group_id], cap)
        rows.append({"group": group_id,
                     "nmae_before": before["nmae"], "ficr_before": before["ficr"],
                     "score_before": before["official_score"],
                     "nmae_after": after["nmae"], "ficr_after": after["ficr"],
                     "score_after": after["official_score"],
                     "score_delta": after["official_score"] - before["official_score"]})
        print(f"  Group {group_id}: Score {before['official_score']:.6f} -> {after['official_score']:.6f} "
              f"({after['official_score'] - before['official_score']:+.6f})  "
              f"[nMAE {before['nmae']:.6f}->{after['nmae']:.6f}, "
              f"FICR {before['ficr']:.6f}->{after['ficr']:.6f}]", flush=True)

    frame = pd.DataFrame(rows)
    mean_before = 0.5 * (1 - frame["nmae_before"].mean()) + 0.5 * frame["ficr_before"].mean()
    mean_after = 0.5 * (1 - frame["nmae_after"].mean()) + 0.5 * frame["ficr_after"].mean()
    print(f"  평균: Score {mean_before:.6f} -> {mean_after:.6f} ({mean_after - mean_before:+.6f})", flush=True)

    if mean_after < mean_before:
        print("  [경고] 후처리가 검증 Score를 떨어뜨렸습니다. 적용 여부를 재검토하세요.", flush=True)

    if gate_on:
        fired = int((post_kwh[3] / GROUPS[3].capacity_kwh - base_cf[3] > 1e-9).sum())
        print(f"  Group3 Gate 발동: {fired}행 / {len(base_cf[3])}행 "
              f"({fired / len(base_cf[3]) * 100:.2f}%)", flush=True)

    frame.to_csv(output_dir / "postprocess_validation_metrics.csv",
                 index=False, encoding="utf-8-sig")
    return frame


# =============================================================================
# Group3 Gate 파라미터 자동 튜닝
#
# 배경: 기존에는 threshold/gamma/eta를 외부에서 가져온 고정값으로 썼다.
# 그 값들은 다른 분류기(AUC 0.854)에 맞춰 고른 것이라, 우리 분류기
# (AUC 0.89)에서는 확률 분포가 달라 같은 threshold가 다른 의미를 갖는다.
#
# 따라서 파라미터를 고정하지 않고, 검증 상반기(H1)에서 고른 뒤
# 하반기(H2)에서 독립 확인해 통과할 때만 채택한다.
# H2에서 개선되지 않으면 Gate를 끈다(파라미터 (0,0,0) 반환).
# =============================================================================

GATE_THRESHOLD_GRID = (0.30, 0.40, 0.50, 0.60, 0.70)
GATE_GAMMA_GRID = (0.50, 1.00, 1.50)
GATE_ETA_GRID = (0.75, 1.00, 1.25, 1.50)
GATE_CAP_CF = 0.15
GATE_MIN_H2_GAIN = 0.0


def _gate_apply(base_cf, expert_cf, probability, threshold, gamma, eta, cap):
    weight = np.clip((probability - threshold) / max(1.0 - threshold, 1e-9), 0.0, 1.0) ** gamma
    gap = np.maximum(expert_cf - base_cf, 0.0)
    correction = np.clip(eta * weight * gap, 0.0, cap)
    return np.clip(base_cf + correction, 0.0, PREDICTION_MAX_CF)


def tune_group3_gate(timestamps, actual_cf, base_cf, expert_cf, probability, capacity_kwh):
    """
    H1에서 Gate 파라미터를 고르고 H2에서 독립 검증한다.

    반환: (params_dict 또는 None, 진단 dict)
    None이면 Gate를 적용하지 않는다.
    """
    dt = pd.to_datetime(timestamps, errors="raise")
    h1 = (dt < pd.Timestamp("2024-07-01 01:00:00")).to_numpy()
    h2 = ~h1

    y_kwh = np.asarray(actual_cf, dtype=float) * capacity_kwh

    def score_on(mask, pred_cf):
        return official_metrics(y_kwh[mask], pred_cf[mask] * capacity_kwh, capacity_kwh)

    if not h1.any() or not h2.any():
        return None, {"accepted": False, "reason": "half-year mask empty"}

    base_h1 = score_on(h1, base_cf)["official_score"]
    best = None

    for threshold in GATE_THRESHOLD_GRID:
        for gamma in GATE_GAMMA_GRID:
            for eta in GATE_ETA_GRID:
                pred = _gate_apply(base_cf, expert_cf, probability,
                                   threshold, gamma, eta, GATE_CAP_CF)
                gain = score_on(h1, pred)["official_score"] - base_h1
                if best is None or gain > best[0]:
                    best = (gain, threshold, gamma, eta)

    if best is None or best[0] <= 0:
        return None, {"accepted": False, "reason": "H1에서 개선 후보 없음"}

    _, threshold, gamma, eta = best
    tuned = _gate_apply(base_cf, expert_cf, probability, threshold, gamma, eta, GATE_CAP_CF)

    h2_before = score_on(h2, base_cf)["official_score"]
    h2_after = score_on(h2, tuned)["official_score"]
    h2_gain = h2_after - h2_before

    full_before = official_metrics(y_kwh, base_cf * capacity_kwh, capacity_kwh)
    full_after = official_metrics(y_kwh, tuned * capacity_kwh, capacity_kwh)
    fired = int((tuned - base_cf > 1e-9).sum())

    diagnostics = {
        "threshold": threshold, "gamma": gamma, "eta": eta, "cap_cf": GATE_CAP_CF,
        "h1_gain": float(best[0]), "h2_before": float(h2_before),
        "h2_after": float(h2_after), "h2_gain": float(h2_gain),
        "full_score_before": float(full_before["official_score"]),
        "full_score_after": float(full_after["official_score"]),
        "fired_rows": fired, "total_rows": int(len(base_cf)),
        "fired_ratio": float(fired / max(len(base_cf), 1)),
    }

    # H2(파라미터 선택에 쓰지 않은 구간)에서 개선되지 않으면 채택하지 않는다.
    if h2_gain <= GATE_MIN_H2_GAIN:
        diagnostics["accepted"] = False
        diagnostics["reason"] = "H2 독립 검증 실패"
        return None, diagnostics

    diagnostics["accepted"] = True
    return {"threshold": threshold, "gamma": gamma, "eta": eta, "cap_cf": GATE_CAP_CF}, diagnostics


def apply_group3_gate_with_params(base_cf, expert_cf, probability, params):
    if params is None:
        return np.clip(np.asarray(base_cf, dtype=float), 0.0, PREDICTION_MAX_CF)
    return _gate_apply(np.asarray(base_cf, dtype=float),
                       np.asarray(expert_cf, dtype=float),
                       np.asarray(probability, dtype=float),
                       params["threshold"], params["gamma"], params["eta"], params["cap_cf"])
    
if __name__ == "__main__":
    main()