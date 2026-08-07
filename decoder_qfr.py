#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
BARAM 2026 — QRF/Direct FICR 기대효용 디코더 v3
   (검증 base와 적용 base가 다를 수 있는 상황용)
================================================================================

리더보드 실적
------------
    개인_6                     Score 0.63133  1-NMAE 0.87018  FICR 0.39248
    개인_6 + QRF               Score 0.63548  1-NMAE 0.86934  FICR 0.40161
    model_2 (base)             Score 0.63473  1-NMAE 0.87293  FICR 0.39653
    model_2 + QRF (G1만)       Score 0.63961  1-NMAE 0.87167  FICR 0.40755  <- 현재 최고

v3에서 추가된 것
---------------
1) --extra-test / --blend-test : 여러 테스트 base를 고정 가중으로 블렌드
      서로 다른 사람이 만든 모델의 평균은 순수 분산 축소다. 보정 계열이 아니므로
      실패 모드가 다르다. 가중치는 튜닝하지 말고 0.5 고정을 권장한다.
      (실험 5 Mean-neutral Model Bank가 실패한 건 같은 파이프라인 안에서
       오차 상관 rho가 0.95였기 때문이다. 다른 사람의 모델은 rho가 낮다.)

2) --decoder {qrf,direct,ens} : 디코더 종류
      qrf    : 조건부 밀도를 RF 잎으로 추정하고 효용을 적분 (plug-in, v2와 동일)
      direct : 효용을 직접 회귀. 후보 c마다
                   t_i = y_i * 1{y_i>=0.1} * r(|y_i - c|)
               를 타깃으로 U_c(x) = E[t|x] 를 LightGBM으로 학습한다.
               밀도 추정이 아니라 부드러운 조건부 기댓값 회귀이고, 후보마다
               전체 학습데이터를 쓴다. plug-in의 국소 표본 잡음을 피한다.
      ens    : 두 효용을 행별 정규화 후 평균

3) --allow-marginal : 반기 게이트는 통과했으나 부트스트랩 CI 하한이 0 이하인
      후보를 별도 파일 submission_marginal.csv 로 함께 출력한다.
      기본 submission.csv 는 항상 엄격 게이트 통과분만 담는다.
      (개인_6 G2가 정확히 이 경우다: dScore +0.0048, P(>0)=0.91)

4) 테스트 base 진단 : 검증 base와 테스트 base의 포화구간 진입률을 비교해
      창 크기가 어긋날 위험을 경고한다. 자동 조정은 하지 않는다.

무엇을 하는가
-------------
    ŷ* = argmax_p  E[ Y·1{Y>=0.1}·r(|Y-p|) ],  r = 4·1{e<=.06} + 3·1{.06<e<=.08}

FICR은 계단 함수라 최적 점예측이 조건부 분포의 평균이 아니라 에너지가중 최빈값이다.

지금까지의 검증 실측 (개인_6 base, train 2022–23 -> val 2024)
-----------------------------------------------------------
  G1  후처리 後 base + 창 ±0.07 블렌드 0.7
      FICR 0.4545 -> 0.4810   Score 0.6699 -> 0.6829
      H1 +0.0063 / H2 +0.0197   CI [+0.0059, +0.0199] P=1.00     -> 채택
  G2  창 ±0.03 블렌드 1.0
      dScore +0.0048  CI [-0.0020, +0.0106] P=0.91                -> CI 미달
  G3  전 조합 기각. QRF·고출력게이트·잠재출력혼합·분산확장 네 방법 모두
      H1 이득과 H2 손실이 대칭 상쇄된다(분산확장 k=1.4: H1 +0.048 / H2 -0.021).
      원인은 기간별 가동상태 분포 이동이며 NWP로 복원되지 않는다(AUC 0.687).

사용 예
-------
  # model_2 base에 3그룹 디코딩 (엄격 + marginal 둘 다 생성)
  python baram_qrf3.py \
    --preprocessed-dir  preprocessed \
    --base-validation   output_personal6/validation_predictions.csv \
    --postprocess-check output_personal6/postprocess_validation_metrics.csv \
    --base-test         model_2_output/submission_model_2.csv \
    --decoder ens --allow-marginal --output-dir out_v3

  # 개인_6 x model_2 0.5 블렌드 base에 디코딩
  python baram_qrf3.py ... \
    --base-test  model_2_output/submission_model_2.csv \
    --extra-test output_personal6/submission.csv --blend-test 0.5

  # 빠른 경로 점검
  python baram_qrf3.py ... --n-estimators 25 --n-candidates 11 --n-boot 40
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

TIME_COL = "forecast_kst_dtm"
RANDOM_SEED = 42

GROUPS = {
    1: dict(target="kpx_group_1", capacity_kwh=21600.0),
    2: dict(target="kpx_group_2", capacity_kwh=21600.0),
    3: dict(target="kpx_group_3", capacity_kwh=21000.0),
}

ACTIVE_CF = 0.10
BAND_FULL, BAND_PART = 0.06, 0.08
RATE_FULL, RATE_PART = 4.0, 3.0
PRED_MAX_CF = 1.02

GRID = np.arange(0.0, PRED_MAX_CF + 1e-9, 0.005).astype(np.float32)
NG = len(GRID)

WINDOWS = (0.02, 0.03, 0.05, 0.07, 0.10, 0.13, 0.15)
BLENDS = (1.0, 0.7, 0.5)

VAL_START = pd.Timestamp("2024-01-01 01:00:00")
VAL_MID = pd.Timestamp("2024-07-01 01:00:00")
DATA_END = pd.Timestamp("2025-01-01 01:00:00")

RF_PARAMS = dict(n_estimators=200, min_samples_leaf=25, max_features=0.35,
                 n_jobs=-1, random_state=RANDOM_SEED)
LGB_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
                  min_child_samples=40, colsample_bytree=0.7, subsample=0.9,
                  subsample_freq=1, reg_lambda=5.0, random_state=RANDOM_SEED,
                  n_jobs=-1, verbosity=-1)

# 개인_6 Group1 후처리 상수 (개인_6 postprocess 섹션과 동일해야 함)
G1_EXPANSION_K = 1.13
G1_BIN_EDGES = [0.25, 0.50, 0.75]
G1_BIN_OFFSETS = np.array([0.00000, -0.01875, 0.01875, 0.01875])
G1_CROSS_DIFF_SCALE = 0.20
G1_CROSS_MOVE_RATIO = 0.35
G1_SMOOTH_WINDOW = 5
G1_SMOOTH_BLEND = 0.20
G1_MEAN_SHIFT_TOL = 0.005
RECON_TOL = 0.002

# 2025는 2024보다 강풍이 많다(P(hub>14): 0.242 -> 0.309). 테스트 진단용 참고값.
EXPECTED_TEST_HIGH_CF_RATE = 0.14      # 2024 실측 0.126에 강풍 증가분을 반영한 추정


# ==============================================================================
# 지표
# ==============================================================================
def official_metrics(y_cf, p_cf, masked=False):
    y = np.asarray(y_cf, dtype=float)
    p = np.asarray(p_cf, dtype=float)
    if not masked:
        m = y >= ACTIVE_CF
        y, p = y[m], p[m]
    if y.size == 0:
        return dict(nmae=np.nan, ficr=np.nan, score=np.nan, n=0)
    e = np.abs(y - p)
    rate = np.where(e <= BAND_FULL, RATE_FULL, np.where(e <= BAND_PART, RATE_PART, 0.0))
    nmae = float(e.mean())
    ficr = float((rate * y).sum() / (4.0 * y.sum()))
    return dict(nmae=nmae, ficr=ficr, score=0.5 * (1 - nmae) + 0.5 * ficr, n=int(y.size))


def cycle_key(ts):
    s = pd.to_datetime(pd.Series(np.asarray(ts)).reset_index(drop=True))
    return (s - pd.Timedelta(hours=1)).dt.normalize().to_numpy()


def block_bootstrap(ts, y, pa, pb, n_boot=400, seed=RANDOM_SEED):
    """회차(하루) 블록 부트스트랩. 오차 자기상관 0.85라 행 단위는 CI를 과소추정한다."""
    y, pa, pb = np.asarray(y, float), np.asarray(pa, float), np.asarray(pb, float)
    _, inv = np.unique(cycle_key(ts), return_inverse=True)
    idx = [np.flatnonzero(inv == i) for i in range(inv.max() + 1)]
    rng = np.random.default_rng(seed)
    st = np.empty(n_boot)
    for t in range(n_boot):
        sel = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
        st[t] = (official_metrics(y[sel], pa[sel])["score"]
                 - official_metrics(y[sel], pb[sel])["score"])
    pt = official_metrics(y, pa)["score"] - official_metrics(y, pb)["score"]
    lo, hi = np.nanpercentile(st, [2.5, 97.5])
    return dict(point=float(pt), ci_low=float(lo), ci_high=float(hi),
                p_positive=float(np.nanmean(st > 0)))


# ==============================================================================
# 개인_6 Group1 후처리 재구성 (검증 base를 제출 base와 일치시키기 위함)
# ==============================================================================
def reconstruct_group1_postprocess(base1_cf, base2_cf, timestamps):
    b1, b2 = np.asarray(base1_cf, float), np.asarray(base2_cf, float)
    base_mean = float(b1.mean())
    cf = np.clip(base_mean + G1_EXPANSION_K * (b1 - base_mean), 0.0, 1.05)
    off = G1_BIN_OFFSETS[np.digitize(cf, bins=G1_BIN_EDGES, right=True)]
    cf = np.clip(cf + (off - off.mean()), 0.0, 1.05)
    diff = b2 - cf
    trust = np.clip(np.abs(diff) / G1_CROSS_DIFF_SCALE, 0.0, 1.0)
    cf = np.clip(cf + G1_CROSS_MOVE_RATIO * trust * diff, 0.0, 1.05)
    ck = cycle_key(timestamps)
    sm = (pd.Series(cf).groupby(ck)
          .transform(lambda s: s.rolling(G1_SMOOTH_WINDOW, center=True,
                                         min_periods=1).mean()).to_numpy())
    cf = np.clip((1 - G1_SMOOTH_BLEND) * cf + G1_SMOOTH_BLEND * sm, 0.0, 1.05)
    shift = float(cf.mean()) - base_mean
    if abs(shift) > G1_MEAN_SHIFT_TOL:
        cf = np.clip(cf - (shift - np.sign(shift) * G1_MEAN_SHIFT_TOL), 0.0, 1.05)
    return cf


def verify_reconstruction(metrics, check_path, group_id=1):
    if check_path is None or not Path(check_path).is_file():
        print("      재구성 대조: postprocess_validation_metrics.csv 미제공 (건너뜀)")
        return None
    df = pd.read_csv(check_path, encoding="utf-8-sig")
    row = df.loc[pd.to_numeric(df["group"], errors="coerce") == group_id]
    if row.empty:
        return None
    row = row.iloc[0]
    rs, rf_ = float(row.get("score_after", np.nan)), float(row.get("ficr_after", np.nan))
    ds, dfi = abs(metrics["score"] - rs), abs(metrics["ficr"] - rf_)
    ok = (ds <= RECON_TOL) and (dfi <= RECON_TOL)
    print(f"      재구성 대조: Score {metrics['score']:.5f} vs {rs:.5f} (d{ds:.5f}) / "
          f"FICR {metrics['ficr']:.5f} vs {rf_:.5f} (d{dfi:.5f})  "
          f"{'일치' if ok else '불일치'}")
    if not ok:
        raise SystemExit(
            "후처리 재구성이 개인_6 결과와 일치하지 않습니다.\n"
            "  개인_6 postprocess 상수를 이 파일 상단과 맞추거나,\n"
            "  후처리된 검증 예측을 직접 저장해 --no-reconstruct-postprocess 로 실행하세요.")
    return ok


# ==============================================================================
# 피처 — 사이트 격자 union (같은 발전단지이므로 누수 아님)
# ==============================================================================
_GRID_RE = re.compile(r"_grid(\d+)$")


def build_features(preprocessed_dir):
    preprocessed_dir = Path(preprocessed_dir)
    frames = {}
    for split in ("train", "test"):
        parts, taken, basedone = [], set(), False
        for g in (1, 2, 3):
            df = pd.read_csv(preprocessed_dir / f"{split}_group{g}_preprocessed.csv",
                             encoding="utf-8-sig", parse_dates=[TIME_COL])
            drop = {GROUPS[g]["target"], "is_long_outage_hour", TIME_COL}
            keep = [TIME_COL]
            for c in df.columns:
                if c in drop or c.endswith(("_roll3h", "_roll6h")):
                    continue
                m = _GRID_RE.search(c)
                if m is None:
                    if not basedone:
                        keep.append(c)
                elif int(m.group(1)) not in taken:
                    keep.append(c)
            basedone = True
            taken |= {int(m.group(1)) for c in df.columns if (m := _GRID_RE.search(c))}
            parts.append(df[keep])
        site = parts[0]
        for p in parts[1:]:
            if len(p.columns) > 1:
                site = site.merge(p, on=TIME_COL, how="outer")
        frames[split] = site.sort_values(TIME_COL).reset_index(drop=True)

    common = [c for c in frames["train"].columns if c in set(frames["test"].columns)]
    site = pd.concat([frames["train"][common], frames["test"][common]],
                     ignore_index=True).sort_values(TIME_COL).reset_index(drop=True)
    dt = site[TIME_COL]
    X = site.drop(columns=[TIME_COL]).apply(pd.to_numeric, errors="coerce")

    add, hub = {}, []
    for g in (5, 6, 10, 11, 12):
        ws, sh = (f"ldaps_50m_mean_wind_speed_grid{g}",
                  f"ldaps_shear_alpha_10m_50m_grid{g}")
        if ws in X.columns and sh in X.columns:
            add[f"hub{g}"] = (X[ws] * (117.0 / 50.0) ** X[sh].clip(-0.5, 0.7)
                              ).clip(0, 60).to_numpy()
            hub.append(f"hub{g}")
    A = pd.DataFrame(add, index=X.index)
    if hub:
        H = A[hub].to_numpy(float)
        A["hub_mean"] = np.nanmean(H, 1)
        A["hub_min"] = np.nanmin(H, 1)
        A["hub_max"] = np.nanmax(H, 1)
        A["hub_spread"] = A["hub_max"] - A["hub_min"]
    if "gfs_100m_wind_speed" in X.columns:
        ga = (X["gfs_shear_alpha_10m_100m"].clip(-0.5, 0.7)
              if "gfs_shear_alpha_10m_100m" in X.columns
              else pd.Series(0.14, index=X.index))
        A["gfs_hub"] = (X["gfs_100m_wind_speed"] * (117.0 / 100.0) ** ga
                        ).clip(0, 60).to_numpy()
        if "hub_mean" in A:
            A["gap"] = A["hub_mean"] - A["gfs_hub"]
            A["gap_abs"] = A["gap"].abs()
            A["min_model"] = np.minimum(A["hub_mean"], A["gfs_hub"])
    t2, r2 = "heightAboveGround_2_t_grid6", "heightAboveGround_2_r_grid6"
    if t2 in X.columns and r2 in X.columns:
        T, RH = X[t2].to_numpy(float), X[r2].to_numpy(float)
        A["icing_soft"] = np.maximum(273.15 - T, 0) * np.maximum(RH - 85.0, 0) / 15.0
        A["icing_flag"] = ((T < 273.15) & (RH >= 90.0)).astype(np.int8)
    A["lead_hour"] = np.where(dt.dt.hour.to_numpy() == 0, 24, dt.dt.hour.to_numpy())
    A["doy_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear.to_numpy() / 365.25)
    A["doy_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear.to_numpy() / 365.25)

    dup = [c for c in A.columns if c in X.columns]
    if dup:
        A = A.rename(columns={c: f"{c}__d" for c in dup})
    F = pd.concat([X, A], axis=1)

    ck = cycle_key(dt)
    cyc = {}
    for c in ("hub_mean", "gap", "gfs_850hpa_wind_speed"):
        if c in F.columns:
            gb = F[c].astype(float).groupby(ck, sort=False)
            cyc[f"{c}_cmean"] = gb.transform("mean").to_numpy()
            cyc[f"{c}_cmin"] = gb.transform("min").to_numpy()
            cyc[f"{c}_cstd"] = gb.transform("std").to_numpy()
    C = pd.DataFrame(cyc, index=F.index)
    C = C.rename(columns={c: f"{c}__c" for c in C.columns if c in F.columns})
    F = pd.concat([F, C], axis=1)
    F = F.loc[:, ~F.columns.duplicated()]
    F = F.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    assert F.columns.is_unique
    return dt, F


# ==============================================================================
# 효용 계산 — QRF plug-in / Direct 회귀
# ==============================================================================
def _leaf_utility(ys):
    w = ys * (ys >= ACTIVE_CF)
    if w.sum() <= 0:
        return np.zeros(NG, dtype=np.float32)
    d = np.abs(ys[None, :] - GRID[:, None])
    rate = np.where(d <= BAND_FULL, RATE_FULL, np.where(d <= BAND_PART, RATE_PART, 0.0))
    return ((rate @ w).astype(np.float32)) / len(ys)


def qrf_utility(X_fit, y_fit, X_pred, rf_params, tag=""):
    """조건부 밀도를 RF 잎으로 추정하고 FICR 효용을 적분 (plug-in)."""
    t0 = time.time()
    rf = RandomForestRegressor(**rf_params).fit(X_fit, y_fit)
    Lf, Lp = rf.apply(X_fit), rf.apply(X_pred)
    del rf
    gc.collect()
    util = np.zeros((len(X_pred), NG), dtype=np.float32)
    for t in range(Lf.shape[1]):
        lt = Lf[:, t]
        nl = int(lt.max()) + 1
        prof = np.zeros((nl, NG), dtype=np.float32)
        o = np.argsort(lt, kind="stable")
        ls, yo = lt[o], y_fit[o]
        b = np.searchsorted(ls, np.arange(nl + 1))
        for leaf in range(nl):
            if b[leaf + 1] > b[leaf]:
                prof[leaf] = _leaf_utility(yo[b[leaf]:b[leaf + 1]])
        util += prof[Lp[:, t]]
        del prof
    print(f"      QRF {tag} {rf_params['n_estimators']}트리 학습행 {len(y_fit)} "
          f"({time.time() - t0:.0f}s)", flush=True)
    gc.collect()
    return util


def direct_utility(X_fit, y_fit, X_pred, n_candidates=21, tag=""):
    """
    효용을 직접 회귀한다. 후보 c마다
        t_i = y_i * 1{y_i>=0.1} * r(|y_i - c|)
    를 타깃으로 U_c(x) = E[t|x] 를 학습한다.
    밀도 추정이 아니라 조건부 기댓값 회귀이므로 국소 표본 잡음을 피하고
    후보마다 전체 학습데이터를 쓴다.
    """
    import lightgbm as lgb
    t0 = time.time()
    cand = np.linspace(0.02, 1.02, int(n_candidates))
    U = np.zeros((len(X_pred), len(cand)), dtype=np.float32)
    for j, c in enumerate(cand):
        e = np.abs(y_fit - c)
        t = y_fit * (y_fit >= ACTIVE_CF) * np.where(
            e <= BAND_FULL, RATE_FULL, np.where(e <= BAND_PART, RATE_PART, 0.0))
        m = lgb.LGBMRegressor(**LGB_PARAMS).fit(X_fit, t)
        U[:, j] = m.predict(X_pred)
        del m
    out = np.empty((len(X_pred), NG), dtype=np.float32)
    for i in range(len(X_pred)):
        out[i] = np.interp(GRID, cand, U[i])
    print(f"      DIRECT {tag} 후보 {len(cand)}개 학습행 {len(y_fit)} "
          f"({time.time() - t0:.0f}s)", flush=True)
    gc.collect()
    return out


def _row_normalize(u):
    mx = u.max(axis=1, keepdims=True)
    return u / np.maximum(mx, 1e-9)


def compute_utility(decoder, X_fit, y_fit, X_pred, rf_params, n_candidates, tag=""):
    if decoder == "qrf":
        return qrf_utility(X_fit, y_fit, X_pred, rf_params, tag)
    if decoder == "direct":
        return direct_utility(X_fit, y_fit, X_pred, n_candidates, tag)
    uq = qrf_utility(X_fit, y_fit, X_pred, rf_params, tag)
    ud = direct_utility(X_fit, y_fit, X_pred, n_candidates, tag)
    return 0.5 * _row_normalize(uq) + 0.5 * _row_normalize(ud)


def decode(util, base, window, blend):
    """기준 예측 주변 ±window 안에서 기대효용 최대점을 찾고 기준과 blend."""
    mask = np.abs(GRID[None, :] - np.asarray(base, float)[:, None]) <= window
    u = np.where(mask, util, -np.inf)
    eu = GRID[u.argmax(1)]
    return np.clip(blend * eu + (1.0 - blend) * base, 0.0, PRED_MAX_CF)


# ==============================================================================
# base 로딩
# ==============================================================================
def load_base_validation(path):
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=[TIME_COL])
    need = {"group", "actual_kwh", "prediction_kwh", "official_eval_row"}
    if not need.issubset(df.columns):
        raise KeyError(f"validation 예측 컬럼 부족: {need - set(df.columns)}")
    out = {}
    for g, s in df.groupby("group"):
        cap = GROUPS[int(g)]["capacity_kwh"]
        s = s.sort_values(TIME_COL).reset_index(drop=True)
        out[int(g)] = pd.DataFrame({
            TIME_COL: s[TIME_COL],
            "actual_cf": s["actual_kwh"].to_numpy(float) / cap,
            "base_cf": s["prediction_kwh"].to_numpy(float) / cap,
            "eval": s["official_eval_row"].to_numpy().astype(bool)})
    return out


def load_submission_cf(path, expect_time=None):
    """제출 형식 CSV -> {group: (time, base_cf)}"""
    df = pd.read_csv(Path(path), encoding="utf-8-sig")
    if TIME_COL not in df.columns:
        raise KeyError(f"{Path(path).name}: {TIME_COL} 컬럼이 필요합니다.")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="raise")
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    if expect_time is not None:
        if len(df) != len(expect_time) or not np.array_equal(
                df[TIME_COL].to_numpy(), np.asarray(expect_time)):
            raise ValueError(f"{Path(path).name}: 테스트 시간축이 전처리와 다릅니다.")
    out = {}
    for g, cfg in GROUPS.items():
        if cfg["target"] not in df.columns:
            raise KeyError(f"{Path(path).name}: {cfg['target']} 컬럼 없음")
        out[g] = pd.to_numeric(df[cfg["target"]], errors="raise").to_numpy(float) \
            / cfg["capacity_kwh"]
    return df, out


def saturation_entry(pred_cf, actual_rate):
    p = np.asarray(pred_cf, float)
    return float(np.mean(p >= 0.85)) / max(actual_rate, 1e-9)


# ==============================================================================
# 메인
# ==============================================================================
def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rf_params = {**RF_PARAMS, "n_estimators": int(args.n_estimators)}

    print("=" * 78)
    print("BARAM — FICR 기대효용 디코더 v3 (검증 base != 적용 base 지원)")
    print(f"  디코더={args.decoder}  RF {args.n_estimators}트리  "
          f"direct 후보 {args.n_candidates}개")
    print(f"  창 {WINDOWS} / 블렌드 {BLENDS}")
    print(f"  레시피 선택: 개인_6 2024 검증 (반기 게이트 + 부트스트랩 CI 하한>0)")
    print(f"  marginal 출력: {'ON' if args.allow_marginal else 'OFF'}")
    print("=" * 78)

    dt, F = build_features(args.preprocessed_dir)
    is_test = (dt >= DATA_END).to_numpy()
    test_time = dt[is_test].reset_index(drop=True)
    print(f"  피처 {F.shape}  (train {int((~is_test).sum())} / test {int(is_test.sum())})")

    # ---- 테스트 base (필요하면 블렌드) ----
    sub_df, base_test = load_submission_cf(args.base_test, test_time.to_numpy())
    blend_info = dict(primary=str(args.base_test), extra=None, weight=0.0)
    if args.extra_test is not None:
        _, extra = load_submission_cf(args.extra_test, test_time.to_numpy())
        w = float(args.blend_test)
        if not (0.0 <= w <= 1.0):
            raise SystemExit("--blend-test 는 0~1 이어야 합니다.")
        print(f"\n  [테스트 base 블렌드] {Path(args.base_test).name} x {1-w:.2f} + "
              f"{Path(args.extra_test).name} x {w:.2f}")
        for g in sorted(GROUPS):
            corr = np.corrcoef(base_test[g], extra[g])[0, 1]
            mad = np.mean(np.abs(base_test[g] - extra[g]))
            base_test[g] = (1 - w) * base_test[g] + w * extra[g]
            print(f"    G{g}: 두 base 상관 {corr:.4f}, 평균|차이| {mad:.4f} CF")
        blend_info.update(extra=str(args.extra_test), weight=w)

    # ---- 검증 base (개인_6) ----
    valbase = load_base_validation(args.base_validation)
    if args.reconstruct:
        v1 = valbase[1].set_index(TIME_COL)
        v2 = valbase[2].set_index(TIME_COL)
        idx = v1.index
        b2 = v2["base_cf"].reindex(idx).to_numpy(float)
        if np.isnan(b2).any():
            raise ValueError("Group2 검증 예측 시간축이 Group1과 다릅니다.")
        pre = v1["base_cf"].to_numpy(float)
        post = reconstruct_group1_postprocess(pre, b2, idx)
        y1 = v1["actual_cf"].to_numpy(float)
        m_pre, m_post = official_metrics(y1, pre), official_metrics(y1, post)
        print(f"\n  [Group1 후처리 재구성] Score {m_pre['score']:.5f} -> {m_post['score']:.5f}, "
              f"FICR {m_pre['ficr']:.5f} -> {m_post['ficr']:.5f}")
        verify_reconstruction(m_post, args.postprocess_check, 1)
        valbase[1] = valbase[1].copy()
        valbase[1]["base_cf"] = post

    targets = {}
    for g, cfg in GROUPS.items():
        t = pd.read_csv(Path(args.preprocessed_dir) / f"train_group{g}_preprocessed.csv",
                        encoding="utf-8-sig", usecols=[TIME_COL, cfg["target"]],
                        parse_dates=[TIME_COL])
        targets[g] = (pd.DataFrame({TIME_COL: dt}).merge(t, on=TIME_COL, how="left")
                      [cfg["target"]].to_numpy(float) / cfg["capacity_kwh"])

    # ---- 진단: 검증 base와 테스트 base의 포화 진입률 비교 ----
    print("\n  [진단] 포화구간 진입률 P(pred>=0.85). 검증 base와 테스트 base가 다르면")
    print("         최적 창이 어긋날 수 있다 (진입률이 높을수록 창은 좁아야 한다).")
    for g in sorted(GROUPS):
        vb = valbase[g]
        m = vb["actual_cf"].to_numpy() >= ACTIVE_CF
        ar = float(np.mean(vb["actual_cf"].to_numpy()[m] >= 0.85))
        rv = saturation_entry(vb["base_cf"].to_numpy()[m], ar)
        rt = saturation_entry(base_test[g], EXPECTED_TEST_HIGH_CF_RATE)
        flag = "" if abs(rt - rv) < 0.25 else "  <- 차이 큼, 창 확인 필요"
        print(f"    G{g}: 검증 {rv:.2f} (P={np.mean(vb['base_cf'].to_numpy()[m]>=0.85):.3f}/"
              f"{ar:.3f})   테스트 {rt:.2f} "
              f"(P={np.mean(base_test[g]>=0.85):.3f}/{EXPECTED_TEST_HIGH_CF_RATE:.3f}){flag}")

    have = {g: np.isfinite(targets[g]) & (~is_test) for g in GROUPS}
    fit_m = {g: have[g] & (dt < VAL_START).to_numpy() for g in GROUPS}
    val_m = {g: have[g] & (dt >= VAL_START).to_numpy() for g in GROUPS}
    Fv = F.fillna(-999)

    strict_cf, marg_cf, recipes, report = {}, {}, {}, {}
    for g in sorted(GROUPS):
        print(f"\n{'=' * 78}\n  Group {g}\n{'=' * 78}")
        y = targets[g]
        vt = dt[val_m[g]]
        vb = valbase[g].set_index(TIME_COL)
        base_v = vb["base_cf"].reindex(vt).to_numpy(float)
        if np.isnan(base_v).any():
            raise ValueError(f"G{g}: 검증 기준 예측에 결측. 시간축 정렬을 확인하세요.")
        ev = vb["eval"].reindex(vt).to_numpy().astype(bool)
        yv = y[val_m[g]]
        h1 = (vt < VAL_MID).to_numpy()

        b = official_metrics(yv[ev], base_v[ev], masked=True)
        a0 = official_metrics(yv[h1 & ev], base_v[h1 & ev], masked=True)
        c0 = official_metrics(yv[~h1 & ev], base_v[~h1 & ev], masked=True)
        print(f"    검증 base (개인_6)  FICR={b['ficr']:.4f} NMAE={b['nmae']:.4f} "
              f"Score={b['score']:.4f}")

        util_v = compute_utility(args.decoder, Fv[fit_m[g]].values, y[fit_m[g]],
                                 Fv[val_m[g]].values, rf_params,
                                 args.n_candidates, tag=f"G{g} val")

        cands = []
        for win in WINDOWS:
            for bl in BLENDS:
                p = decode(util_v, base_v, win, bl)
                m = official_metrics(yv[ev], p[ev], masked=True)
                d1 = official_metrics(yv[h1 & ev], p[h1 & ev],
                                      masked=True)["score"] - a0["score"]
                d2 = official_metrics(yv[~h1 & ev], p[~h1 & ev],
                                      masked=True)["score"] - c0["score"]
                stable = (d1 > 0) and (d2 > 0)
                cands.append(dict(window=win, blend=bl, **m, h1=d1, h2=d2, stable=stable))
                print(f"    창±{win:.2f} 블렌드{bl:.1f}  FICR={m['ficr']:.4f} "
                      f"NMAE={m['nmae']:.4f} Score={m['score']:.4f} | "
                      f"H1 {d1:+.4f} H2 {d2:+.4f} {'안정' if stable else ''}")

        ok = [c for c in cands if c["stable"] and c["score"] - b["score"] >= args.min_gain]
        strict_cf[g] = base_test[g].copy()
        marg_cf[g] = base_test[g].copy()
        recipes[g] = dict(strict=None, marginal=None)
        report[f"group{g}"] = dict(base=b, candidates=cands)

        if not ok:
            print("    -> 반기 안정 기준 통과 후보 없음. 테스트 base 그대로 유지.")
            report[f"group{g}"]["verdict"] = "반기 안정 기준 미통과"
            continue

        best = max(ok, key=lambda c: c["score"])
        p = decode(util_v, base_v, best["window"], best["blend"])
        ci = block_bootstrap(vt[ev], yv[ev], p[ev], base_v[ev], n_boot=args.n_boot)
        print(f"    최선 창±{best['window']:.2f} 블렌드{best['blend']:.1f}  "
              f"dScore={best['score'] - b['score']:+.5f}  "
              f"CI[{ci['ci_low']:+.5f},{ci['ci_high']:+.5f}] P={ci['p_positive']:.2f}  "
              f"{'엄격 통과' if ci['ci_low'] > 0 else 'marginal (CI 미달)'}")
        report[f"group{g}"].update(best=best, ci=ci)
        strict_pass = ci["ci_low"] > 0
        report[f"group{g}"]["verdict"] = "엄격 통과" if strict_pass else "marginal"

        if not strict_pass and not args.allow_marginal:
            print("    -> CI 미달이고 --allow-marginal 이 꺼져 있어 적용하지 않습니다.")
            continue

        # ---- 테스트 디코딩 (전체 train으로 효용 재학습) ----
        print("    전체 학습구간으로 효용 재학습 -> 테스트 디코딩")
        util_t = compute_utility(args.decoder, Fv[have[g]].values, y[have[g]],
                                 Fv[is_test].values, rf_params,
                                 args.n_candidates, tag=f"G{g} full")
        dec = decode(util_t, base_test[g], best["window"], best["blend"])
        d = dec - base_test[g]
        print(f"    테스트 변화: 평균d={d.mean():+.4f}  평균|d|={np.abs(d).mean():.4f}  "
              f"|d|>0.10 비율={np.mean(np.abs(d) > 0.10):.3f}")
        print(f"    P(CF>=0.85) {np.mean(base_test[g] >= 0.85):.3f} -> "
              f"{np.mean(dec >= 0.85):.3f}   "
              f"P(CF<0.10) {np.mean(base_test[g] < 0.10):.3f} -> "
              f"{np.mean(dec < 0.10):.3f}")
        report[f"group{g}"]["test_shift"] = dict(
            mean_delta=float(d.mean()), mean_abs_delta=float(np.abs(d).mean()),
            p_abs_gt_010=float(np.mean(np.abs(d) > 0.10)),
            p_ge_085_before=float(np.mean(base_test[g] >= 0.85)),
            p_ge_085_after=float(np.mean(dec >= 0.85)),
            p_lt_010_before=float(np.mean(base_test[g] < 0.10)),
            p_lt_010_after=float(np.mean(dec < 0.10)))

        marg_cf[g] = dec
        recipes[g]["marginal"] = dict(window=best["window"], blend=best["blend"])
        if strict_pass:
            strict_cf[g] = dec
            recipes[g]["strict"] = dict(window=best["window"], blend=best["blend"])
        del util_t
        gc.collect()
        del util_v
        gc.collect()

    # ---- 제출 파일 ----
    def write(cf_map, name):
        out = (sub_df[["forecast_id", TIME_COL]].copy()
               if "forecast_id" in sub_df.columns
               else pd.DataFrame({TIME_COL: test_time}))
        piece = pd.DataFrame({TIME_COL: test_time})
        for g, cfg in GROUPS.items():
            piece[cfg["target"]] = np.clip(cf_map[g], 0.0, PRED_MAX_CF) * cfg["capacity_kwh"]
        out = out.merge(piece, on=TIME_COL, how="left", validate="one_to_one")
        cols = [GROUPS[g]["target"] for g in sorted(GROUPS)]
        if out[cols].isna().any().any():
            raise ValueError(f"{name}: 결측 예측이 있습니다.")
        path = output_dir / name
        out.to_csv(path, index=False, encoding="utf-8-sig",
                   date_format="%Y-%m-%d %H:%M:%S")
        return path

    p_strict = write(strict_cf, "submission.csv")
    paths = {"strict": str(p_strict)}
    if args.allow_marginal and any(recipes[g]["marginal"] and not recipes[g]["strict"]
                                   for g in GROUPS):
        p_marg = write(marg_cf, "submission_marginal.csv")
        paths["marginal"] = str(p_marg)

    (output_dir / "qrf_report.json").write_text(
        json.dumps({"decoder": args.decoder, "blend": blend_info,
                    "recipes": recipes, "paths": paths, "report": report},
                   ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    print("\n" + "=" * 78)
    for g in sorted(GROUPS):
        r = recipes[g]
        v = report[f"group{g}"].get("verdict", "-")
        if r["strict"]:
            print(f"  G{g}: 엄격 채택  창±{r['strict']['window']:.2f} "
                  f"블렌드{r['strict']['blend']:.1f}")
        elif r["marginal"]:
            print(f"  G{g}: marginal만  창±{r['marginal']['window']:.2f} "
                  f"블렌드{r['marginal']['blend']:.1f}  ({v})")
        else:
            print(f"  G{g}: 미적용 ({v})")
    print(f"\n  제출(엄격): {p_strict}")
    if "marginal" in paths:
        print(f"  제출(marginal 포함): {paths['marginal']}")
        print("  두 파일의 차이는 CI 미달 그룹뿐입니다. 여유 제출이 있을 때만 marginal 사용.")
    print(f"  리포트: {output_dir / 'qrf_report.json'}")
    print("\n  제출 전 확인: test_shift의 mean_delta(+0.01~+0.03), p_abs_gt_010(<0.10)")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(
        description="BARAM FICR 기대효용 디코더 v3 (검증 base != 적용 base 지원)")
    ap.add_argument("--preprocessed-dir", type=Path, required=True)
    ap.add_argument("--base-validation", type=Path, required=True,
                    help="개인_6 validation_predictions.csv (레시피 선택의 유일한 근거)")
    ap.add_argument("--base-test", type=Path, required=True,
                    help="적용할 테스트 base 제출 파일 (예: model_2 submission)")
    ap.add_argument("--extra-test", type=Path, default=None,
                    help="선택. 블렌드할 두 번째 테스트 base 제출 파일")
    ap.add_argument("--blend-test", type=float, default=0.5,
                    help="--extra-test 의 가중치. 튜닝하지 말고 0.5 권장")
    ap.add_argument("--postprocess-check", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("out_qrf3"))
    ap.add_argument("--decoder", choices=["qrf", "direct", "ens"], default="qrf")
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--n-candidates", type=int, default=21,
                    help="direct 디코더의 후보 개수")
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--min-gain", type=float, default=0.002)
    ap.add_argument("--allow-marginal", action="store_true",
                    help="반기 게이트는 통과했으나 CI 미달인 후보를 별도 파일로 출력")
    ap.add_argument("--no-reconstruct-postprocess", dest="reconstruct",
                    action="store_false")
    ap.set_defaults(reconstruct=True)
    a = ap.parse_args()
    t0 = time.time()
    run(a)
    print(f"총 소요 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()