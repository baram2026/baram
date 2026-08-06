# -*- coding: utf-8 -*-
"""김은호 모델 2의 단일 파일 실행 코드.

저장소 루트의 wind_data에서 시작하여 전처리, 검증, 전체 재학습,
Group별 최종 모델 선택과 submission 생성을 한 번에 수행합니다.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
WIND_DATA_DIR = REPO_ROOT / "wind_data"

if not WIND_DATA_DIR.is_dir():
    raise FileNotFoundError(
        f"wind_data 폴더를 찾지 못했습니다: {WIND_DATA_DIR}"
    )
if not (WIND_DATA_DIR / "sample_submission.csv").exists():
    raise FileNotFoundError(
        f"sample_submission.csv를 찾지 못했습니다: {WIND_DATA_DIR}"
    )

TOTAL_STARTED = time.perf_counter()
print("저장소 루트:", REPO_ROOT)
print("데이터 폴더:", WIND_DATA_DIR)
print("Random Seed: 42")



# =============================================================================
# 1. 전처리 함수
# =============================================================================

import gc
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# SCADA 0·저출력 원인 분류 및 장기 고장·정비 결측 처리
#
# 저장소 wind_data에 필요한 파일
# 1. scada_vestas_train.csv
# 2. scada_unison_train.csv
# 3. ldaps_train.csv
#
# 생성 파일
# - 지정한 save_dir/scada_vestas_train_cleaned.csv
# - 지정한 save_dir/scada_unison_train_cleaned.csv
#
# 진단 CSV는 생성하지 않으며,
# 상태별 개수와 새로 NaN 처리된 수는 실행 결과로 출력합니다.
# =========================================================


# =========================================================
# 0. 실행 설정
# =========================================================

EFFICIENCY_THRESHOLD = 0.02       # 정격 대비 2% 미만
ICING_TEMP_K = 273.15             # 0°C 미만
ICING_HUMIDITY = 90.0             # 상대습도 90% 이상
LOW_WIND_MS = 4.0                 # 터빈 SCADA 풍속 4m/s 미만
PEER_ACTIVE_RATIO = 0.50          # 타 터빈 절반 이상 정상발전
MIN_VALID_PEERS = 2               # 비교 가능한 타 터빈 최소 수

# 최소 하루 이상 지속된 저출력만 장기 고장·정비로 확정
LONG_OUTAGE_HOURS = 24

SCADA_INTERVAL_MINUTES = 10
SAVE_DIAGNOSTICS = False

#극단값을 처리하기 위한 설정값 추가
SCADA_POWER_UPPER_RATIO = 1.05
SCADA_NEGATIVE_TOLERANCE_RATIO = 0.05

# =========================================================
# 1. 제조사·터빈 설정
# =========================================================
SCADA_CONFIG = {
    "vestas": {
        "file_name": "scada_vestas_train.csv",
        "n_turbines": 12,
        "rated_power_kw": 3600.0,
        "group_map": {
            **{
                i: 1
                for i in range(1, 7)
            },
            **{
                i: 2
                for i in range(7, 13)
            }
        }
    },

    "unison": {
        "file_name": "scada_unison_train.csv",
        "n_turbines": 5,
        "rated_power_kw": 4200.0,
        "group_map": {
            i: 3
            for i in range(1, 6)
        }
    }
}


# KPX 그룹별 터빈 수
# - Group 1·2: VESTAS 각 6기
# - Group 3: UNISON 5기
KPX_GROUP_TURBINE_COUNTS = {
    1: 6,
    2: 6,
    3: 5
}


# 터빈 좌표와 LDAPS 격자 중심을 비교해 정한 최근접 격자
TURBINE_GRID_MAP = {
    "vestas_wtg01": 10,
    "vestas_wtg02": 5,
    "vestas_wtg03": 5,
    "vestas_wtg04": 5,
    "vestas_wtg05": 6,
    "vestas_wtg06": 6,

    "vestas_wtg07": 6,
    "vestas_wtg08": 6,
    "vestas_wtg09": 6,
    "vestas_wtg10": 11,
    "vestas_wtg11": 11,
    "vestas_wtg12": 11,

    "unison_wtg01": 6,
    "unison_wtg02": 12,
    "unison_wtg03": 12,
    "unison_wtg04": 12,
    "unison_wtg05": 12
}


# 실제 발전량을 NaN 처리할 최종 확정 상태
#
# 개별정지_고장정비후보는 중간 후보일 뿐이므로
# 여기에는 포함하지 않습니다.
FAULT_STATUSES = {
    "장기연속정지"
}


# 장기연속정지 여부를 검사할 수 있는 상태
#
# 착빙과 저풍속은 명시적으로 제외합니다.
LONG_OUTAGE_ELIGIBLE_STATUSES = {
    "판정불가",
    "판정불가_풍속결측",
    "판정불가_기상결측",
    "개별정지_고장정비후보"
}


# =========================================================
# 3. 착빙 판정용 LDAPS 기상 데이터 준비
# =========================================================
def prepare_ldaps_icing_weather(
    ldaps_path
):
    """
    착빙 판정에 필요한 기온·상대습도만 읽고 전처리합니다.

    - 대상 격자: 5, 6, 10, 11, 12
    - 같은 예보 회차·격자 안에서 시간순 보간
    - 남은 결측치는 격자별 중앙값으로 보완
    - 상대습도는 0~100으로 clipping
    """

    usecols = [
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "grid_id",
        "heightAboveGround_2_t",
        "heightAboveGround_2_r"
    ]

    weather = pd.read_csv(
        ldaps_path,
        usecols=usecols,
        encoding="utf-8-sig"
    )

    weather["forecast_kst_dtm"] = pd.to_datetime(
        weather["forecast_kst_dtm"],
        errors="raise"
    )

    weather["data_available_kst_dtm"] = pd.to_datetime(
        weather["data_available_kst_dtm"],
        errors="raise"
    )

    weather["grid_id"] = pd.to_numeric(
        weather["grid_id"],
        errors="raise"
    ).astype("int16")

    weather = weather[
        weather["grid_id"].isin(
            sorted(
                set(
                    TURBINE_GRID_MAP.values()
                )
            )
        )
    ].copy()

    weather["temp_kelvin"] = pd.to_numeric(
        weather["heightAboveGround_2_t"],
        errors="coerce"
    )

    weather["humidity"] = pd.to_numeric(
        weather["heightAboveGround_2_r"],
        errors="coerce"
    )

    # 물리적으로 잘못된 습도는 NaN으로 만든 후 보간
    invalid_humidity = (
        (weather["humidity"] < 0)
        | (weather["humidity"] > 100)
    )

    weather.loc[
        invalid_humidity,
        "humidity"
    ] = np.nan

    weather = weather.sort_values(
        [
            "grid_id",
            "data_available_kst_dtm",
            "forecast_kst_dtm"
        ]
    )

    group_cols = [
        "grid_id",
        "data_available_kst_dtm"
    ]

    for col in [
        "temp_kelvin",
        "humidity"
    ]:
        weather[col] = (
            weather
            .groupby(
                group_cols,
                sort=False,
                observed=True
            )[col]
            .transform(
                lambda s: s.interpolate(
                    method="linear",
                    limit_direction="both"
                )
            )
        )

        # 같은 예보 회차 안에서도 채워지지 않은 값은
        # 동일 격자의 Train 중앙값으로 보완
        grid_median = (
            weather
            .groupby(
                "grid_id"
            )[col]
            .transform("median")
        )

        weather[col] = weather[col].fillna(
            grid_median
        )

        weather[col] = weather[col].fillna(
            weather[col].median()
        )

    weather["humidity"] = (
        weather["humidity"]
        .clip(
            lower=0,
            upper=100
        )
    )

    weather = weather[
        [
            "forecast_kst_dtm",
            "grid_id",
            "temp_kelvin",
            "humidity"
        ]
    ].copy()

    if weather.duplicated(
        subset=[
            "forecast_kst_dtm",
            "grid_id"
        ]
    ).any():
        raise ValueError(
            "LDAPS에 중복된 "
            "forecast_kst_dtm × grid_id가 있습니다."
        )

    return (
        weather
        .sort_values(
            [
                "forecast_kst_dtm",
                "grid_id"
            ]
        )
        .reset_index(drop=True)
    )


# =========================================================
# 4. SCADA Wide → Long
# =========================================================
def scada_wide_to_long(
    scada_df,
    manufacturer
):
    """
    원본 터빈별 Wide SCADA를
    시각 × 터빈 Long 형식으로 변환합니다.
    """

    config = SCADA_CONFIG[
        manufacturer
    ]

    frames = []

    for turbine_no in range(
        1,
        config["n_turbines"] + 1
    ):
        turbine_id = (
            f"{manufacturer}_wtg"
            f"{turbine_no:02d}"
        )

        power_col = (
            f"{turbine_id}_power_kw10m"
        )

        ws_col = (
            f"{turbine_id}_ws"
        )

        wd_col = (
            f"{turbine_id}_wd"
        )

        missing = [
            col
            for col in [
                power_col,
                ws_col,
                wd_col
            ]
            if col not in scada_df.columns
        ]

        if missing:
            raise KeyError(
                f"{turbine_id} 컬럼 누락: "
                f"{missing}"
            )

        part = scada_df[
            [
                "kst_dtm",
                power_col,
                ws_col,
                wd_col
            ]
        ].copy()

        part.columns = [
            "kst_dtm",
            "power",
            "ws",
            "wd"
        ]

        part["manufacturer"] = (
            manufacturer
        )

        part["turbine_no"] = (
            turbine_no
        )

        part["turbine_id"] = (
            turbine_id
        )

        part["kpx_group"] = (
            config["group_map"][
                turbine_no
            ]
        )

        part["grid_id"] = np.int64(
            TURBINE_GRID_MAP[
                turbine_id
            ]
        )

        part["source_power_col"] = (
            power_col
        )

        # 10분 단위 정격 발전량
        #
        # VESTAS:
        # 3600 kW × 10/60 = 600 kWh
        #
        # UNISON:
        # 4200 kW × 10/60 = 700 kWh
        part["rated_10min"] = (
            config["rated_power_kw"]
            * SCADA_INTERVAL_MINUTES
            / 60.0
        )

        frames.append(part)

    long_df = pd.concat(
        frames,
        ignore_index=True
    )

    long_df["kst_dtm"] = pd.to_datetime(
        long_df["kst_dtm"],
        errors="raise"
    )

    for col in [
        "power",
        "ws",
        "wd"
    ]:
        long_df[col] = pd.to_numeric(
            long_df[col],
            errors="coerce"
        )
        

    # 원본값 보존
    long_df["power_raw"] = long_df["power"]
    lower_bound = (
    -long_df["rated_10min"]
    * SCADA_NEGATIVE_TOLERANCE_RATIO)
    upper_bound = (
    long_df["rated_10min"]
    * SCADA_POWER_UPPER_RATIO)
    # 물리적으로 불가능한 극단값
    long_df["is_power_outlier"] = (
    long_df["power_raw"].notna()
    & (
        (long_df["power_raw"] < lower_bound)
        | (long_df["power_raw"] > upper_bound)
    )
)
    #정상 범위 안의 작은 음수는 0으로 처리
    long_df["is_negative_clipped"] = (
    long_df["power_raw"].notna()
    & ~long_df["is_power_outlier"]
    & (long_df["power_raw"] < 0))
    long_df["power"] = (
    long_df["power_raw"]
    .mask(long_df["is_power_outlier"])
    .clip(lower=0)
)
    return long_df

# =========================================================
# 5. 10분 SCADA에 시간별 LDAPS 기상값 결합
# =========================================================
def attach_weather(
    scada_long,
    ldaps_weather
):
    """
    터빈별 최근접 격자의 가장 가까운 시간별 LDAPS 값을
    10분 SCADA 데이터에 결합합니다.

    SCADA가 10분, LDAPS가 1시간이므로
    ±31분 범위에서 가장 가까운 예보 시각을 연결합니다.
    """

    left = scada_long.copy()
    right = ldaps_weather.copy()

    left["kst_dtm"] = pd.to_datetime(
        left["kst_dtm"],
        errors="raise"
    )

    right["forecast_kst_dtm"] = pd.to_datetime(
        right["forecast_kst_dtm"],
        errors="raise"
    )

    left["grid_id"] = pd.to_numeric(
        left["grid_id"],
        errors="raise"
    ).astype("int64")

    right["grid_id"] = pd.to_numeric(
        right["grid_id"],
        errors="raise"
    ).astype("int64")

    # merge_asof는 on 컬럼이 전체적으로
    # 오름차순 정렬되어 있어야 합니다.
    left = (
        left
        .sort_values("kst_dtm")
        .reset_index(drop=True)
    )

    right = (
        right
        .sort_values(
            "forecast_kst_dtm"
        )
        .reset_index(drop=True)
    )

    return pd.merge_asof(
        left=left,
        right=right,
        left_on="kst_dtm",
        right_on="forecast_kst_dtm",
        by="grid_id",
        direction="nearest",
        tolerance=pd.Timedelta(
            "31min"
        )
    )


# =========================================================
# 6. 터빈별 장기 연속 저출력 구간 탐지
# =========================================================
def add_long_outage_flag(df):
    """
    착빙과 저풍속으로 판정된 행은 제외하고,
    미분류 상태 또는 개별 고장·정비 후보 상태에서만
    장기 연속 저출력을 검사합니다.

    LONG_OUTAGE_HOURS 이상 연속된 구간의 전체 행을
    장기연속정지로 표시합니다.

    착빙·저풍속·정상발전·원본결측 행을 만나거나
    10분 간격이 끊기면 연속 구간이 종료됩니다.
    """

    result = (
        df
        .sort_values(
            [
                "turbine_id",
                "kst_dtm"
            ]
        )
        .reset_index(drop=True)
        .copy()
    )

    expected_gap = pd.Timedelta(
        minutes=SCADA_INTERVAL_MINUTES
    )

    # 장기정지 연속성 계산 대상
    result[
        "is_long_outage_candidate"
    ] = (
        result["is_low_output"]
        & result["status"].isin(
            LONG_OUTAGE_ELIGIBLE_STATUSES
        )
    )

    previous_state = (
        result
        .groupby(
            "turbine_id",
            sort=False
        )[
            "is_long_outage_candidate"
        ]
        .shift()
    )

    state_changed = (
        previous_state.isna()
        | result[
            "is_long_outage_candidate"
        ].ne(previous_state)
    )

    time_broken = (
        result
        .groupby(
            "turbine_id",
            sort=False
        )["kst_dtm"]
        .diff()
        .ne(expected_gap)
    )

    new_streak = (
        state_changed
        | time_broken
    )

    result["_streak_id"] = (
        new_streak
        .groupby(
            result["turbine_id"],
            sort=False
        )
        .cumsum()
    )

    streak_size = (
        result
        .groupby(
            [
                "turbine_id",
                "_streak_id"
            ],
            sort=False
        )[
            "is_long_outage_candidate"
        ]
        .transform("size")
    )

    result[
        "low_output_streak_hours"
    ] = np.where(
        result[
            "is_long_outage_candidate"
        ],
        (
            streak_size
            * SCADA_INTERVAL_MINUTES
            / 60.0
        ),
        0.0
    )

    result["is_long_outage"] = (
        result[
            "is_long_outage_candidate"
        ]
        & (
            result[
                "low_output_streak_hours"
            ]
            >= LONG_OUTAGE_HOURS
        )
    )

    return result.drop(
        columns="_streak_id"
    )


# =========================================================
# 7. 저출력 원인 분류 및 발전량 정제
# =========================================================
def classify_and_clean(
    scada_weather
):
    """
    판단 순서

    1. 정격 대비 발전 효율 2% 이상
       → 정상발전

    2. 2% 미만 + 0°C 미만 + 습도 90% 이상
       → 착빙

    3. 착빙 아님 + SCADA 풍속 4m/s 미만
       → 저풍속

    4. 나머지에서 같은 KPX 그룹의
       타 터빈 절반 이상 정상발전
       → 개별정지_고장정비후보

    5. 착빙·저풍속을 제외한 저출력 상태가
       최소 24시간 이상 연속
       → 장기연속정지

    결측 처리

    - 장기연속정지
      → power_clean을 NaN 처리

    - 개별정지_고장정비후보
      → 중간 후보일 뿐이므로 원래 발전량 유지

    - 착빙·저풍속·판정불가
      → 원래 발전량 유지
    """

    result = scada_weather.copy()

    result["power_efficiency"] = (
        result["power"]
        / result["rated_10min"]
    )

    result["is_low_output"] = (
        result["power"].notna()
        & (
            result[
                "power_efficiency"
            ]
            < EFFICIENCY_THRESHOLD
        )
    )

    result["is_active"] = (
        result["power"].notna()
        & (
            result[
                "power_efficiency"
            ]
            >= EFFICIENCY_THRESHOLD
        )
    )

    # 같은 시각·같은 KPX 그룹의
    # 다른 터빈과 비교
    peer_group = result.groupby(
        [
            "kst_dtm",
            "kpx_group"
        ],
        sort=False
    )

    all_active = peer_group[
        "is_active"
    ].transform("sum")

    all_valid = peer_group[
        "power"
    ].transform("count")

    # 자기 자신 제외
    result["peer_active_count"] = (
        all_active
        - result[
            "is_active"
        ].astype("int8")
    )

    result["peer_valid_count"] = (
        all_valid
        - result[
            "power"
        ].notna().astype("int8")
    )

    result["peer_active_ratio"] = (
        result["peer_active_count"]
        / result[
            "peer_valid_count"
        ].replace(
            0,
            np.nan
        )
    )

    result["is_icing"] = (
        (
            result["temp_kelvin"]
            < ICING_TEMP_K
        )
        & (
            result["humidity"]
            >= ICING_HUMIDITY
        )
    )

    result["is_low_wind"] = (
        result["ws"]
        < LOW_WIND_MS
    )

    result["status"] = (
        "정상발전"
    )

    original_missing = (
        result["power"].isna()
    )

    power_outlier = result["is_power_outlier"].fillna(False)

    candidate = (
        result["is_low_output"]
    )

    result.loc[
        original_missing,
        "status"
    ] = "원본결측"

    result.loc[
        power_outlier,
        "status"
    ] = "원본이상치"

    # -----------------------------------------------------
    # 1순위: 착빙
    # -----------------------------------------------------
    icing_mask = (
        candidate
        & result["is_icing"]
    )

    result.loc[
        icing_mask,
        "status"
    ] = "착빙"

    # -----------------------------------------------------
    # 2순위: 저풍속
    # -----------------------------------------------------
    remaining = (
        candidate
        & ~result["is_icing"]
    )

    low_wind_mask = (
        remaining
        & result["is_low_wind"]
    )

    result.loc[
        low_wind_mask,
        "status"
    ] = "저풍속"

    # -----------------------------------------------------
    # 풍속 또는 기상 결측
    # -----------------------------------------------------
    remaining = (
        remaining
        & ~result["is_low_wind"]
    )

    result.loc[
        (
            remaining
            & result["ws"].isna()
        ),
        "status"
    ] = "판정불가_풍속결측"

    weather_missing = (
        result["temp_kelvin"].isna()
        | result["humidity"].isna()
    )

    result.loc[
        (
            remaining
            & result["ws"].notna()
            & weather_missing
        ),
        "status"
    ] = "판정불가_기상결측"

    # -----------------------------------------------------
    # 3순위: 타 터빈 비교
    # -----------------------------------------------------
    fault_candidate_mask = (
        remaining
        & result["ws"].notna()
        & ~weather_missing
        & (
            result["peer_valid_count"]
            >= MIN_VALID_PEERS
        )
        & (
            result["peer_active_ratio"]
            >= PEER_ACTIVE_RATIO
        )
    )

    result.loc[
        fault_candidate_mask,
        "status"
    ] = "개별정지_고장정비후보"

    # -----------------------------------------------------
    # 4순위: 최소 24시간 이상 장기 연속 저출력
    # -----------------------------------------------------
    result = add_long_outage_flag(
        result
    )

    result.loc[
        result["is_long_outage"],
        "status"
    ] = "장기연속정지"

    # -----------------------------------------------------
    # 장기연속정지만 NaN 처리
    # -----------------------------------------------------
    result["power_clean"] = (
        result["power"]
    )

    result.loc[
        result["status"].isin(
            FAULT_STATUSES
        ),
        "power_clean"
    ] = np.nan

    return result


# =========================================================
# 8. 터빈별 장기정지 상태를 시간별 그룹 요약으로 변환
# =========================================================
def build_hourly_outage_summary(
    diagnostics,
    group_ids
):
    """
    지정한 KPX 그룹의 터빈별 10분 장기연속정지 결과를
    시간별 장기정지 터빈 수로 변환합니다.

    예:
    - VESTAS diagnostics + group_ids=[1, 2]
    - UNISON diagnostics + group_ids=[3]

    시간 대응:
    - 01:10, 01:20, ..., 02:00
      → 02:00 발전량 Label

    이 요약값은 해당 그룹의 장기정지 행 제외에만 사용하며,
    최종 모델 입력 피처로는 남기지 않습니다.
    """

    group_ids = list(group_ids)

    invalid_group_ids = [
        group_id
        for group_id in group_ids
        if group_id not in KPX_GROUP_TURBINE_COUNTS
    ]

    if invalid_group_ids:
        raise ValueError(
            "알 수 없는 KPX 그룹입니다: "
            f"{invalid_group_ids}"
        )

    if not group_ids:
        raise ValueError(
            "시간별 장기정지 요약에 group_ids가 필요합니다."
        )

    required_cols = [
        "kst_dtm",
        "turbine_id",
        "kpx_group",
        "is_long_outage"
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in diagnostics.columns
    ]

    if missing_cols:
        raise KeyError(
            "시간별 장기정지 요약 생성에 "
            "필요한 컬럼이 없습니다: "
            f"{missing_cols}"
        )

    temp = diagnostics[
        required_cols
    ].copy()

    temp["kst_dtm"] = pd.to_datetime(
        temp["kst_dtm"],
        errors="raise"
    )

    temp["is_long_outage"] = (
        temp["is_long_outage"]
        .fillna(False)
        .astype(bool)
    )

    # 현재 제작사의 diagnostics에서
    # 요청받은 KPX 그룹만 선택
    temp = temp[
        temp["kpx_group"].isin(
            group_ids
        )
    ].copy()

    if temp.empty:
        raise ValueError(
            "요청한 KPX 그룹의 SCADA diagnostics가 없습니다: "
            f"{group_ids}"
        )

    # 10분 SCADA를 시간 종료 Label 시각에 대응
    temp["forecast_kst_dtm"] = (
        temp["kst_dtm"]
        .dt.ceil("h")
    )

    # 같은 시간·같은 터빈의 10분 행 중 하나라도
    # 장기연속정지이면 그 시간에는 해당 터빈이
    # 장기정지 상태였던 것으로 처리
    hourly_turbine = (
        temp
        .groupby(
            [
                "forecast_kst_dtm",
                "kpx_group",
                "turbine_id"
            ],
            as_index=False,
            sort=False
        )["is_long_outage"]
        .any()
    )

    # 시간·그룹별 장기정지 터빈 수
    hourly_group = (
        hourly_turbine
        .groupby(
            [
                "forecast_kst_dtm",
                "kpx_group"
            ],
            as_index=False,
            sort=False
        )["is_long_outage"]
        .sum()
        .rename(
            columns={
                "is_long_outage":
                "outage_turbine_count"
            }
        )
    )

    summary = (
        hourly_group
        .pivot(
            index="forecast_kst_dtm",
            columns="kpx_group",
            values="outage_turbine_count"
        )
        .reset_index()
    )

    summary.columns.name = None

    summary = summary.rename(
        columns={
            group_id:
            f"group{group_id}_outage_turbine_count"
            for group_id in group_ids
        }
    )

    for group_id, total_turbines in (
        (
            (
                group_id,
                KPX_GROUP_TURBINE_COUNTS[group_id]
            )
            for group_id in group_ids
        )
    ):
        outage_col = (
            f"group{group_id}"
            "_outage_turbine_count"
        )

        available_col = (
            f"group{group_id}"
            "_available_turbine_count"
        )

        flag_col = (
            f"group{group_id}"
            "_fault_maintenance_flag"
        )

        if outage_col not in summary.columns:
            summary[outage_col] = 0

        summary[outage_col] = (
            summary[outage_col]
            .fillna(0)
            .astype("int8")
        )

        invalid_count = (
            (summary[outage_col] < 0)
            | (
                summary[outage_col]
                > total_turbines
            )
        )

        if invalid_count.any():
            raise ValueError(
                f"Group {group_id} 장기정지 터빈 수가 "
                "유효 범위를 벗어났습니다."
            )

        summary[available_col] = (
            total_turbines
            - summary[outage_col]
        ).astype("int8")

        summary[flag_col] = (
            summary[outage_col] > 0
        )

    output_cols = ["forecast_kst_dtm"]

    for group_id in group_ids:
        output_cols.extend(
            [
                f"group{group_id}_outage_turbine_count",
                f"group{group_id}_available_turbine_count",
                f"group{group_id}_fault_maintenance_flag"
            ]
        )

    summary = summary[
        output_cols
    ].copy()

    if summary[
        "forecast_kst_dtm"
    ].duplicated().any():
        raise ValueError(
            "시간별 장기정지 요약에 "
            "중복 시각이 있습니다."
        )

    print(
        "\n[시간별 장기정지 터빈 수]"
    )

    for group_id in group_ids:
        outage_col = (
            f"group{group_id}"
            "_outage_turbine_count"
        )

        print(
            f"Group {group_id} 장기정지 포함 시각:",
            int(
                (
                    summary[outage_col] > 0
                ).sum()
            )
        )

        print(
            f"Group {group_id} 최대 동시 장기정지 터빈 수:",
            int(
                summary[outage_col].max()
            )
        )

    return (
        summary
        .sort_values("forecast_kst_dtm")
        .reset_index(drop=True)
    )


# =========================================================
# 9. 정제된 발전량을 원본 Wide 구조로 복원
# =========================================================
def rebuild_cleaned_wide(
    original_df,
    classified_long
):
    power_wide = (
        classified_long
        .pivot(
            index="kst_dtm",
            columns="source_power_col",
            values="power_clean"
        )
        .reset_index()
    )

    power_cols = [
        col
        for col in original_df.columns
        if col.endswith(
            "_power_kw10m"
        )
    ]

    base = original_df.drop(
        columns=power_cols
    ).copy()

    base["kst_dtm"] = pd.to_datetime(
        base["kst_dtm"],
        errors="raise"
    )

    cleaned = base.merge(
        power_wide,
        on="kst_dtm",
        how="left",
        validate="one_to_one"
    )

    # 원본과 동일한 컬럼 순서 유지
    return cleaned[
        original_df.columns
    ]

# =========================================================
# 0. 데이터별 설정
# - 전처리 로직은 preprocess_weather() 하나를 공통 사용
# - LDAPS/GFS의 컬럼 차이만 설정으로 관리
# =========================================================
WEATHER_CONFIG = {
    "ldaps": {
        "humidity_cols": [
            "heightAboveGround_2_r"
        ],
        "nonnegative_cols": [
            "surface_0_SNOM"
        ],
        "temperature_col": "heightAboveGround_2_t",
        # 실제 제공 파일의 값이 약 250~310이므로 Kelvin으로 처리
        "temperature_unit": "kelvin",
        "pressure_col": "surface_0_sp",
        "rho_col": "ldaps_rho",
        "shortwave_col": "surface_0_NDNSW",
        "night_flag_col": "ldaps_is_night",
        "wind_pairs": [
            (
                "heightAboveGround_10_10u",
                "heightAboveGround_10_10v",
                "ldaps_10m"
            ),
            (
                "heightAboveGround_5_XBLWS",
                "heightAboveGround_5_YBLWS",
                "ldaps_5m_blws"
            ),
            (
                "ldaps_50m_mean_u",
                "ldaps_50m_mean_v",
                "ldaps_50m_mean"
            )
        ],
        "wind_feature_prefixes": [
            "ldaps_10m",
            "ldaps_50m_mean"
        ],# 5m BLWS는 wind_pairs에는 유지하지만 wind_feature에서는 삭제
        "shear": {
            "lower_speed_col": "ldaps_10m_wind_speed",
            "upper_speed_col": "ldaps_50m_mean_wind_speed",
            "lower_height": 10.0,
            "upper_height": 50.0,
            "output_col": "ldaps_shear_alpha_10m_50m"
        },
        "veer": {
            "lower_direction_col": "ldaps_10m_wind_direction_deg",
            "upper_direction_col": "ldaps_50m_mean_wind_direction_deg",
            "lower_speed_col": "ldaps_10m_wind_speed",
            "upper_speed_col": "ldaps_50m_mean_wind_speed",
            "output_col": "ldaps_veer_50m_10m_deg"
        },
        "drop_cols": [
            "surface_0_lsm",
            "surface_0_h",
            "data_available_kst_dtm",
            "ldaps_50m_mean_u",# LDAPS 50m 평균 계산용 임시 U/V이므로 삭제
            "ldaps_50m_mean_v",
            "ldaps_10m_wind_direction_deg",
            "ldaps_5m_blws_wind_direction_deg",
            "ldaps_50m_mean_wind_direction_deg"
        ]# degree 풍향은 veer 계산 후 삭제
    },

    "gfs": {
        "humidity_cols": [
            "heightAboveGround_2_2r",
            "isobaricInhPa_850_r"
        ],
        "nonnegative_cols": [],
        "temperature_col": "heightAboveGround_2_2t",
        # 실제 제공 파일의 값이 약 250~310이므로 Kelvin으로 처리
        "temperature_unit": "kelvin",
        "pressure_col": "surface_0_sp",
        "rho_col": "gfs_rho",
        "shortwave_col": "surface_0_dswrf",
        "night_flag_col": "gfs_is_night",
        "wind_pairs": [
            (
                "heightAboveGround_10_10u",
                "heightAboveGround_10_10v",
                "gfs_10m"
            ),
            (
                "heightAboveGround_80_u",
                "heightAboveGround_80_v",
                "gfs_80m"
            ),
            (
                "heightAboveGround_100_100u",
                "heightAboveGround_100_100v",
                "gfs_100m"
            ),
            (
                "planetaryBoundaryLayer_0_u",
                "planetaryBoundaryLayer_0_v",
                "gfs_pbl"
            ),
            (
                "isobaricInhPa_850_u",
                "isobaricInhPa_850_v",
                "gfs_850hpa"
            ),
            (
                "isobaricInhPa_700_u",
                "isobaricInhPa_700_v",
                "gfs_700hpa"
            )
        ],
        "wind_feature_prefixes": [
            "gfs_10m",
            "gfs_80m",
            "gfs_100m",
            "gfs_pbl",
            "gfs_850hpa",
            "gfs_700hpa"
        ],
        "shear": {
            "lower_speed_col": "gfs_10m_wind_speed",
            "upper_speed_col": "gfs_100m_wind_speed",
            "lower_height": 10.0,
            "upper_height": 100.0,
            "output_col": "gfs_shear_alpha_10m_100m"
        },
        "veer": {
            "lower_direction_col": "gfs_10m_wind_direction_deg",
            "upper_direction_col": "gfs_100m_wind_direction_deg",
            "lower_speed_col": "gfs_10m_wind_speed",
            "upper_speed_col": "gfs_100m_wind_speed",
            "output_col": "gfs_veer_100m_10m_deg"
        },
        "drop_cols": [
            "data_available_kst_dtm",
            "gfs_10m_wind_direction_deg",
            "gfs_80m_wind_direction_deg",
            "gfs_100m_wind_direction_deg",
            "gfs_pbl_wind_direction_deg",
            "gfs_850hpa_wind_direction_deg",
            "gfs_700hpa_wind_direction_deg"
        ]#degree 풍향은 veer 계산 후 삭제
    }
}


GROUP_CAPACITY_KWH = {
    "kpx_group_1": 21600.0,
    "kpx_group_2": 21600.0,
    "kpx_group_3": 21000.0
}


GLOBAL_TIME_FEATURES = [
    "month",
    "day_sin",
    "day_cos",
    "hour_sin",
    "hour_cos"
]


# =========================================================
# 1. 풍속·풍향 및 추가 피처 보조 함수
# =========================================================
def calculate_uv_direction(
    u,
    v,
    calm_threshold=0.3
):
    """
    U, V 성분으로 풍속과 기상학적 풍향을 계산합니다.

    theta = degrees(atan2(-U, -V)) % 360
    """

    u = np.asarray(u, dtype="float64")
    v = np.asarray(v, dtype="float64")

    wind_speed = np.hypot(u, v) 

    wind_direction_deg = (
        np.degrees(np.arctan2(-u, -v))
        % 360.0
    )

    direction_rad = np.deg2rad(wind_direction_deg)
    dir_sin = np.sin(direction_rad)
    dir_cos = np.cos(direction_rad)

    # 무풍에서는 풍향을 0으로 통일
    calm_mask = (
        np.isfinite(wind_speed)
        & (wind_speed <= calm_threshold)
    )

    wind_direction_deg[calm_mask] = 0.0
    dir_sin[calm_mask] = 0.0
    dir_cos[calm_mask] = 0.0

    return (
        wind_speed,
        wind_direction_deg,
        dir_sin,
        dir_cos
    )


def add_uv_wind_features(
    df,
    u_col,
    v_col,
    prefix,
    calm_threshold=1e-12
):
    """
    생성 피처
    - {prefix}_wind_speed
    - {prefix}_wind_direction_deg
    - {prefix}_dir_sin
    - {prefix}_dir_cos
    """

    missing_cols = [
        col
        for col in [u_col, v_col]
        if col not in df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"[{prefix}] 필요한 컬럼이 없습니다: "
            f"{missing_cols}"
        )

    u = pd.to_numeric(
        df[u_col],
        errors="coerce"
    ).to_numpy(dtype="float64")

    v = pd.to_numeric(
        df[v_col],
        errors="coerce"
    ).to_numpy(dtype="float64")

    (
        wind_speed,
        wind_direction_deg,
        dir_sin,
        dir_cos
    ) = calculate_uv_direction(
        u=u,
        v=v,
        calm_threshold=calm_threshold
    )

    df[f"{prefix}_wind_speed"] = wind_speed
    df[f"{prefix}_wind_direction_deg"] = wind_direction_deg
    df[f"{prefix}_dir_sin"] = dir_sin
    df[f"{prefix}_dir_cos"] = dir_cos

    return df


def calculate_shear_alpha(
    upper_speed,
    lower_speed,
    upper_height,
    lower_height,
    epsilon=1e-6,
    low_speed_threshold=0.3 # shear 저풍속 처리
):
    """
    alpha =
        ln((v_upper + epsilon) / (v_lower + epsilon))
        / ln(z_upper / z_lower)

    상층 또는 하층 풍속이 0.3 m/s보다 작으면
    저풍속 구간의 불안정한 shear를 0으로 처리합니다.
    """

    upper_speed = pd.to_numeric(
        upper_speed,
        errors="coerce"
    )

    lower_speed = pd.to_numeric(
        lower_speed,
        errors="coerce"
    )

    shear_alpha = (
        np.log(
            (upper_speed + epsilon)
            / (lower_speed + epsilon)
        )
        / np.log(upper_height / lower_height)
    )

    low_speed_mask = (
        (upper_speed < low_speed_threshold)
        | (lower_speed < low_speed_threshold)
    )

    return shear_alpha.mask(
        low_speed_mask,
        0.0
    )


def calculate_signed_veer(
    upper_direction,
    lower_direction
):
    """
    ((wd_upper - wd_lower + 180) % 360) - 180
    """

    upper_direction = pd.to_numeric(
        upper_direction,
        errors="coerce"
    )

    lower_direction = pd.to_numeric(
        lower_direction,
        errors="coerce"
    )

    return (
        (
            upper_direction
            - lower_direction
            + 180.0
        )
        % 360.0
    ) - 180.0


def add_speed_direction_interactions( #풍향x풍속 함수 정의부는 그대로 두되, 아래 호출부 삭제
    df,
    wind_prefixes
):
    """
    모든 풍속에 대해 다음 피처를 생성합니다.
    - wind_speed × dir_sin
    - wind_speed × dir_cos
    """

    for prefix in wind_prefixes:
        speed_col = f"{prefix}_wind_speed"
        sin_col = f"{prefix}_dir_sin"
        cos_col = f"{prefix}_dir_cos"

        missing_cols = [
            col
            for col in [speed_col, sin_col, cos_col]
            if col not in df.columns
        ]

        if missing_cols:
            raise KeyError(
                f"[{prefix}] 상호작용 생성에 필요한 "
                f"컬럼이 없습니다: {missing_cols}"
            )

        df[f"{prefix}_speed_x_dir_sin"] = (
            df[speed_col] * df[sin_col]
        )

        df[f"{prefix}_speed_x_dir_cos"] = (
            df[speed_col] * df[cos_col]
        )

    return df


def add_night_speed_interactions(
    df,
    night_flag_col,
    wind_prefixes
):
    """
    is_night × wind_speed 피처를 생성합니다.
    """

    for prefix in wind_prefixes:
        speed_col = f"{prefix}_wind_speed"

        if speed_col not in df.columns:
            raise KeyError(
                f"야간 풍속 생성에 필요한 컬럼이 없습니다: "
                f"{speed_col}"
            )

        df[f"{prefix}_night_speed"] = (
            df[night_flag_col]
            * df[speed_col]
        )

    return df


# =========================================================
# 2. 발전량 Label 전처리
# =========================================================
def preprocess_train_labels(
    df,
    upper_ratio=1.05
):
    """
    - 하한: 0 kWh
    - 상한: 그룹 설비용량의 105%
    - 결측치는 그대로 유지
    """

    result = df.copy()

    if "kst_dtm" not in result.columns:
        raise KeyError(
            "train_labels에 'kst_dtm' 컬럼이 없습니다."
        )

    result["kst_dtm"] = pd.to_datetime(
        result["kst_dtm"],
        errors="raise"
    )

    original_row_count = len(result)

    for target_col, capacity_kwh in GROUP_CAPACITY_KWH.items():
        if target_col not in result.columns:
            raise KeyError(
                f"train_labels에 '{target_col}' 컬럼이 없습니다."
            )

        result[target_col] = pd.to_numeric(
            result[target_col],
            errors="coerce"
        ).clip(
            lower=0,
            upper=capacity_kwh * upper_ratio
        )

    if len(result) != original_row_count:
        raise RuntimeError(
            "Label 전처리 과정에서 행 수가 변경되었습니다."
        )

    return result

# =========================================================
# 3. 이상치 처리, 보간 및 Train fallback 통계치 함수
# =========================================================
def get_weather_value_columns(df, source):
    """
    시간·격자 키를 제외한 기상값 컬럼을 반환합니다.
    """

    source = source.lower()

    if source not in WEATHER_CONFIG:
        raise ValueError(
            f"알 수 없는 source입니다: {source}"
        )

    exclude_cols = {
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "grid_id",
        *WEATHER_CONFIG[source]["drop_cols"]
    }

    return [
        col
        for col in df.columns
        if col not in exclude_cols
    ]


def replace_weather_outliers_with_nan(
    df,
    source
):
    """
    물리적으로 유효하지 않은 값을 NaN으로 변환합니다.

    현재 WEATHER_CONFIG에 정의된 기준:
    - 상대습도: 0~100 범위 밖 → NaN
    - 음수가 불가능한 변수: 0 미만 → NaN
    - 양의 무한대, 음의 무한대 → NaN

    이 함수에서는 clip하지 않습니다.
    이상치를 NaN으로 만든 뒤 보간할 수 있도록 처리합니다.
    """

    source = source.lower()

    if source not in WEATHER_CONFIG:
        raise ValueError(
            f"알 수 없는 source입니다: {source}"
        )

    config = WEATHER_CONFIG[source]
    result = df.copy()

    value_cols = get_weather_value_columns(
        result,
        source
    )

    # 기상값을 숫자형으로 변환하고
    # inf와 -inf도 결측치로 처리
    for col in value_cols:
        result[col] = (
            pd.to_numeric(
                result[col],
                errors="coerce"
            )
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

    # 상대습도는 0~100 범위 밖을 NaN 처리
    for col in config["humidity_cols"]:
        if col in result.columns:
            invalid_mask = (
                (result[col] < 0)
                | (result[col] > 100)
            )

            result.loc[
                invalid_mask,
                col
            ] = np.nan

    # 물리적으로 음수가 불가능한 변수는
    # 0 미만 값을 NaN 처리
    for col in config["nonnegative_cols"]:
        if col in result.columns:
            invalid_mask = (
                result[col] < 0
            )

            result.loc[
                invalid_mask,
                col
            ] = np.nan

    return result


def calculate_train_fallbacks(
    train_df,
    source
):
    """
    보간 후에도 남는 결측치에 사용할 중앙값을
    Train에서만 계산합니다.

    이상치는 먼저 NaN으로 바꾼 후 제외하므로,
    정상 범위의 Train 값만 중앙값 계산에 사용됩니다.
    """

    source = source.lower()

    if source not in WEATHER_CONFIG:
        raise ValueError(
            f"알 수 없는 source입니다: {source}"
        )

    # 이상치를 경계값으로 clip하지 않고
    # NaN으로 변환한 상태에서 fallback 계산
    temp = replace_weather_outliers_with_nan(
        df=train_df,
        source=source
    )

    fallback_values = {}

    for col in get_weather_value_columns(
        temp,
        source
    ):
        numeric_col = pd.to_numeric(
            temp[col],
            errors="coerce"
        )

        if numeric_col.notna().any():
            fallback_values[col] = float(
                numeric_col.median()
            )

    return fallback_values


def interpolate_missing_by_forecast_cycle(
    df,
    source,
    fallback_values=None
):
    """
    처리 순서
    1. 물리적 이상치와 무한대를 NaN으로 변환
    2. 동일한 grid_id 및 data_available_kst_dtm 안에서
       forecast_kst_dtm 순으로 선형 보간
    3. 보간 후에도 남는 NaN은 Train에서 계산한
       fallback 중앙값으로 대체

    최종 clip은 preprocess_weather()에서 수행합니다.
    """

    source = source.lower()

    if source not in WEATHER_CONFIG:
        raise ValueError(
            f"알 수 없는 source입니다: {source}"
        )

    result = df.copy()

    required_cols = [
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "grid_id"
    ]

    missing_required = [
        col
        for col in required_cols
        if col not in result.columns
    ]

    if missing_required:
        raise KeyError(
            "보간에 필요한 키 컬럼이 없습니다: "
            f"{missing_required}"
        )

    result["forecast_kst_dtm"] = pd.to_datetime(
        result["forecast_kst_dtm"],
        errors="raise"
    )

    result["data_available_kst_dtm"] = pd.to_datetime(
        result["data_available_kst_dtm"],
        errors="raise"
    )

    # 전처리 후 원래 행 순서를 복원하기 위한 임시 컬럼
    result["__original_order"] = np.arange(
        len(result)
    )

    # -----------------------------------------------------
    # 1단계: 이상치와 무한대를 NaN으로 변환
    # -----------------------------------------------------
    result = replace_weather_outliers_with_nan(
        df=result,
        source=source
    )

    value_cols = get_weather_value_columns(
        result,
        source
    )

    cols_with_missing = [
        col
        for col in value_cols
        if result[col].isna().any()
    ]

    if cols_with_missing:
        group_cols = [
            "grid_id",
            "data_available_kst_dtm"
        ]

        # 같은 예보 주기 및 격자 안에서
        # forecast 시각 순으로 보간하기 위해 정렬
        result = result.sort_values(
            group_cols
            + ["forecast_kst_dtm"]
        )

        grouped = result.groupby(
            group_cols,
            sort=False,
            observed=True
        )

        # -------------------------------------------------
        # 2단계: 예보 주기 내부 선형 보간
        # -------------------------------------------------
        for col in cols_with_missing:
            result[col] = grouped[col].transform(
                lambda series: series.interpolate(
                    method="linear",
                    limit_direction="both"
                )
            )

        # -------------------------------------------------
        # 3단계: 보간 후 남은 NaN을
        #         Train 중앙값으로 대체
        # -------------------------------------------------
        if fallback_values is not None:
            for col in cols_with_missing:
                if col in fallback_values:
                    result[col] = result[col].fillna(
                        fallback_values[col]
                    )

        # 원래 행 순서 복원
        result = result.sort_values(
            "__original_order"
        )

    result = result.drop(
        columns="__original_order"
    )

    return result

# =========================================================
# 4. Train/Test 공통 기상 전처리 함수
# =========================================================
def preprocess_weather(
    df,
    source,
    fallback_values=None
):
    """
    LDAPS/GFS 및 Train/Test에 공통으로 적용합니다.

    source:
        "ldaps" 또는 "gfs"
    """

    source = source.lower()

    if source not in WEATHER_CONFIG:
        raise ValueError(
            f"source는 {list(WEATHER_CONFIG)} 중 "
            "하나여야 합니다."
        )

    config = WEATHER_CONFIG[source]
    result = df.copy()

    original_row_count = len(result)
    original_unique_times = result[
        "forecast_kst_dtm"
    ].nunique(dropna=False)

    # -----------------------------------------------------
    # 1단계: 결측치 보간
    # -----------------------------------------------------
    result = interpolate_missing_by_forecast_cycle(
        df=result,
        source=source,
        fallback_values=fallback_values
    )

    # -----------------------------------------------------
    # 2단계: 보간 후 최종 물리적 범위 보정
    # -----------------------------------------------------
    for col in config["humidity_cols"]:
        if col in result.columns:
            result[col] = pd.to_numeric(
                result[col],
                errors="coerce"
            ).clip(
                lower=0,
                upper=100
            )

    for col in config["nonnegative_cols"]:
        if col in result.columns:
            result[col] = pd.to_numeric(
                result[col],
                errors="coerce"
            ).clip(lower=0)

    # -----------------------------------------------------
    # 3단계: U, V 기반 풍속·풍향 피처
    # -----------------------------------------------------
    if source == "ldaps":
        u50_max = pd.to_numeric(
            result[
                "heightAboveGround_50_50MUmax"
            ],
            errors="coerce"
        )

        u50_min = pd.to_numeric(
            result[
                "heightAboveGround_50_50MUmin"
            ],
            errors="coerce"
        )

        v50_max = pd.to_numeric(
            result[
                "heightAboveGround_50_50MVmax"
            ],
            errors="coerce"
        )

        v50_min = pd.to_numeric(
            result[
                "heightAboveGround_50_50MVmin"
            ],
            errors="coerce"
        )

        # U/V 성분을 각각 산술평균한 뒤,
        # 다른 풍속과 동일하게 hypot(U, V)로 평균 풍속 계산(hypot은 calculate_uv_direction함수에 있음)
        result["ldaps_50m_mean_u"] = (
            u50_max + u50_min
        ) / 2.0

        result["ldaps_50m_mean_v"] = (
            v50_max + v50_min
        ) / 2.0

        # 50m U/V 성분의 max-min 범위 피처
        result["ldaps_50m_u_max_minus_min"] = (
            u50_max - u50_min
        )

        result["ldaps_50m_v_max_minus_min"] = (
            v50_max - v50_min
        )

    for u_col, v_col, prefix in config["wind_pairs"]:
        result = add_uv_wind_features(
            df=result,
            u_col=u_col,
            v_col=v_col,
            prefix=prefix
        )

    # -----------------------------------------------------
    # 3.5-1단계: 시간 주기성 피처
    # -----------------------------------------------------
    forecast_datetime = pd.to_datetime(
        result["forecast_kst_dtm"],
        errors="raise"
    )

    dayofyear = forecast_datetime.dt.dayofyear
    hour = forecast_datetime.dt.hour

    result["month"] = (
        forecast_datetime.dt.month
        .astype("int8")
    )

    result["day_sin"] = np.sin(
        2.0 * np.pi * dayofyear / 365.25
    )

    result["day_cos"] = np.cos(
        2.0 * np.pi * dayofyear / 365.25
    )

    result["hour_sin"] = np.sin(
        2.0 * np.pi * hour / 24.0
    )

    result["hour_cos"] = np.cos(
        2.0 * np.pi * hour / 24.0
    )

    # -----------------------------------------------------
    # 3.5-2단계: 공기 밀도
    # -----------------------------------------------------
    temperature = pd.to_numeric(
        result[config["temperature_col"]],
        errors="coerce"
    )

    if config["temperature_unit"] == "celsius":
        temperature_kelvin = temperature + 273.15
    elif config["temperature_unit"] == "kelvin":
        temperature_kelvin = temperature
    else:
        raise ValueError(
            "temperature_unit은 'celsius' 또는 "
            "'kelvin'이어야 합니다."
        )

    temperature_kelvin = (
        temperature_kelvin.clip(lower=1e-6)
    )

    surface_pressure = pd.to_numeric(
        result[config["pressure_col"]],
        errors="coerce"
    )

    result[config["rho_col"]] = (
        surface_pressure
        / (
            287.058
            * temperature_kelvin
        )
    )

    # -----------------------------------------------------
    # 3.5-3단계: Wind Shear Alpha
    # -----------------------------------------------------
    shear_config = config["shear"]

    result[
        shear_config["output_col"]
    ] = calculate_shear_alpha(
        upper_speed=result[
            shear_config["upper_speed_col"]
        ],
        lower_speed=result[
            shear_config["lower_speed_col"]
        ],
        upper_height=shear_config[
            "upper_height"
        ],
        lower_height=shear_config[
            "lower_height"
        ],
        epsilon=1e-6
    )

    # -----------------------------------------------------
    # 3.5-4단계: 야간 플래그
    # -----------------------------------------------------
    shortwave_radiation = pd.to_numeric(
        result[config["shortwave_col"]],
        errors="coerce"
    )

    result[config["night_flag_col"]] = (
        shortwave_radiation <= 0
    ).astype("int8")

    # -----------------------------------------------------
    # 3.5-5단계: 야간 풍속 상호작용
    # -----------------------------------------------------
    result = add_night_speed_interactions( #LDAPS wind feature에서 경계층 5m 바람 삭제했으므로 
        df=result,
        night_flag_col=config[
            "night_flag_col"
        ],
        wind_prefixes=config[
            "wind_feature_prefixes"
        ]
    )

    # -----------------------------------------------------
    # 3.5-6단계: Veer
    # -----------------------------------------------------
    veer_config = config["veer"]
     
    veer = calculate_signed_veer(
        upper_direction=result[
            veer_config[
                "upper_direction_col"
            ]
        ],
        lower_direction=result[
            veer_config[
                "lower_direction_col"
            ]
        ]
    )
    low_speed_thershold=0.3
    # 어느 한쪽이라도 무풍이면 풍향 변화는 0으로 처리
    calm_mask = (
        (
            result[
                veer_config["lower_speed_col"]
            ] <= low_speed_thershold
        )
        | (
            result[
                veer_config["upper_speed_col"]
            ] <= low_speed_thershold
        )
    )

    result[
        veer_config["output_col"]
    ] = veer.mask(calm_mask, 0.0)

    # -----------------------------------------------------
    # 4단계: 불필요 컬럼 삭제
    # -----------------------------------------------------
    result = result.drop(
        columns=config["drop_cols"],
        errors="ignore"
    )

    # -----------------------------------------------------
    # 5단계: 구조 및 결측치 검증
    # -----------------------------------------------------
    if len(result) != original_row_count:
        raise RuntimeError(
            "전처리 중 행 수가 변경되었습니다."
        )

    processed_unique_times = result[
        "forecast_kst_dtm"
    ].nunique(dropna=False)

    if processed_unique_times != original_unique_times:
        raise RuntimeError(
            "전처리 중 예측 시각 수가 변경되었습니다."
        )

    remaining_missing = result.isna().sum()
    remaining_missing = remaining_missing[
        remaining_missing > 0
    ]

    if not remaining_missing.empty:
        raise ValueError(
            "전처리 후 결측치가 남아 있습니다:\n"
            + remaining_missing.to_string()
        )

    numeric_cols = result.select_dtypes(
        include=[np.number]
    ).columns

    infinite_count = int(
        np.isinf(
            result[numeric_cols].to_numpy()
        ).sum()
    )

    if infinite_count > 0:
        raise ValueError(
            f"전처리 후 무한대 값이 "
            f"{infinite_count}개 남아 있습니다."
        )

    return result


# =========================================================
# 5. 그룹별 LDAPS Wide 변환 및 GFS 5번 격자 준비
# =========================================================
LDAPS_GROUP_GRIDS = {
    1: [5, 6, 10],
    2: [6, 11],
    3: [6, 12]
}

GROUP_TARGET_COLS = {
    1: "kpx_group_1",
    2: "kpx_group_2",
    3: "kpx_group_3"
}


GRID_METADATA_COLS = {
    "latitude",
    "longitude"
}#여기에서 latitude, longitude 포함됨


def normalize_grid_id(grid_series):
    """
    grid_id를 정수형으로 통일합니다.
    """

    numeric_grid = pd.to_numeric(
        grid_series,
        errors="coerce"
    )

    if numeric_grid.isna().any():
        invalid_values = (
            grid_series[
                numeric_grid.isna()
            ]
            .astype(str)
            .unique()[:5]
        )

        raise ValueError(
            "숫자로 변환할 수 없는 grid_id가 있습니다: "
            f"{invalid_values.tolist()}"
        )

    return numeric_grid.astype("int16")


def get_pivot_weather_columns(df):
    """
    Wide 변환 또는 GFS 단일 격자 병합에 사용할
    기상 변수 컬럼을 반환합니다.

    시각, 격자 ID, 위·경도 및 공통 시간 피처는 제외합니다.
    """

    exclude_cols = {
        "forecast_kst_dtm",
        "grid_id",
        *GRID_METADATA_COLS,#GRID_METADATA_COLS 언패킹
        *GLOBAL_TIME_FEATURES
    }

    return [
        col
        for col in df.columns
        if col not in exclude_cols
    ]


def validate_time_grid_unique(
    df,
    data_name
):
    """
    forecast_kst_dtm × grid_id 조합의 중복을 검증합니다.
    """

    duplicated_mask = df.duplicated(
        subset=[
            "forecast_kst_dtm",
            "grid_id"
        ]
    )

    if duplicated_mask.any():
        raise ValueError(
            f"{data_name}에 중복된 "
            "'forecast_kst_dtm × grid_id' 조합이 "
            f"{int(duplicated_mask.sum())}개 있습니다."
        )


def assert_same_forecast_times(
    left_df,
    right_df,
    left_name,
    right_name
):
    """
    두 데이터프레임의 forecast_kst_dtm 구성이 같은지 검증합니다.
    """

    left_times = set(
        pd.to_datetime(
            left_df["forecast_kst_dtm"],
            errors="raise"
        )
    )

    right_times = set(
        pd.to_datetime(
            right_df["forecast_kst_dtm"],
            errors="raise"
        )
    )

    if left_times != right_times:
        only_left = len(
            left_times - right_times
        )

        only_right = len(
            right_times - left_times
        )

        raise ValueError(
            f"{left_name}와 {right_name}의 "
            "예측 시각 구성이 다릅니다. "
            f"{left_name}에만 존재: {only_left}, "
            f"{right_name}에만 존재: {only_right}"
        )


def extract_global_time_features(df):
    """
    모든 격자에서 동일한 공통 시간 피처를
    시각당 한 행으로 추출합니다.
    """

    available_cols = [
        col
        for col in GLOBAL_TIME_FEATURES
        if col in df.columns
    ]

    time_features = (
        df[
            [
                "forecast_kst_dtm",
                *available_cols
            ]
        ]
        .drop_duplicates()
    )

    if time_features[
        "forecast_kst_dtm"
    ].duplicated().any():
        raise ValueError(
            "동일한 시각에 서로 다른 시간 파생 피처 값이 "
            "존재합니다."
        )

    return time_features


def pivot_ldaps_group_to_wide(
    ldaps_df,
    group_id
):
    """
    그룹과 관련된 LDAPS 격자만 필터링하여
    Long Format을 시각당 한 행의 Wide Format으로 변환합니다.

    Group 1: grid 5, 6, 10
    Group 2: grid 6, 11
    Group 3: grid 6, 12

    생성 컬럼 예:
        ldaps_10m_wind_speed_grid6
        ldaps_10m_wind_speed_grid11
    """

    if group_id not in LDAPS_GROUP_GRIDS:
        raise ValueError(
            "group_id는 1, 2, 3 중 하나여야 합니다."
        )

    result = ldaps_df.copy()

    required_cols = [
        "forecast_kst_dtm",
        "grid_id"
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in result.columns
    ]

    if missing_cols:
        raise KeyError(
            "LDAPS Wide 변환에 필요한 컬럼이 없습니다: "
            f"{missing_cols}"
        )

    result["forecast_kst_dtm"] = pd.to_datetime(
        result["forecast_kst_dtm"],
        errors="raise"
    )

    result["grid_id"] = normalize_grid_id(
        result["grid_id"]
    )

    group_grids = LDAPS_GROUP_GRIDS[group_id]

    selected = result[
        result["grid_id"].isin(group_grids)
    ].copy()

    if selected.empty:
        raise ValueError(
            f"Group {group_id} 관련 LDAPS 격자 데이터가 없습니다."
        )

    found_grids = set(
        selected["grid_id"].unique()
    )

    missing_grids = (
        set(group_grids)
        - found_grids
    )

    if missing_grids:
        raise ValueError(
            f"Group {group_id}에 필요한 LDAPS 격자가 없습니다: "
            f"{sorted(missing_grids)}"
        )

    validate_time_grid_unique(
        selected,
        data_name=f"LDAPS Group {group_id}"
    )

    # 모든 시각에 해당 그룹의 관련 격자가 모두 존재하는지 확인
    grid_count_by_time = (
        selected
        .groupby("forecast_kst_dtm")["grid_id"]
        .nunique()
    )

    incomplete_times = grid_count_by_time[
        grid_count_by_time
        != len(group_grids)
    ]

    if not incomplete_times.empty:
        raise ValueError(
            f"Group {group_id}에서 일부 시각의 LDAPS 격자가 "
            f"누락되었습니다. 누락 시각 수: {len(incomplete_times)}"
        )

    time_features = extract_global_time_features(
        selected
    )

    weather_cols = get_pivot_weather_columns(
        selected
    )

    if not weather_cols:
        raise ValueError(
            "LDAPS Wide 변환에 사용할 기상 변수 컬럼이 없습니다."
        )

    # Long → Wide
    wide_df = selected.pivot(
        index="forecast_kst_dtm",
        columns="grid_id",
        values=weather_cols
    )

    # MultiIndex를 변수명_grid{id} 형태로 변경
    wide_df.columns = [
        f"{feature}_grid{int(grid_id)}"
        for feature, grid_id in wide_df.columns
    ]

    wide_df = wide_df.reset_index()

    # 공통 시간 피처는 격자별로 반복하지 않고 한 번만 유지
    wide_df = time_features.merge(
        wide_df,
        on="forecast_kst_dtm",
        how="inner",
        validate="one_to_one"
    )

    wide_df = (
        wide_df
        .sort_values("forecast_kst_dtm")
        .reset_index(drop=True)
    )

    if not wide_df[
        "forecast_kst_dtm"
    ].is_unique:
        raise RuntimeError(
            f"Group {group_id} LDAPS Wide 변환 후 "
            "예측 시각이 중복되었습니다."
        )

    return wide_df


def prepare_gfs_grid5(gfs_df):
    """
    GFS에서 grid_id=5인 행만 선택합니다.

    GFS는 단일 격자를 사용하므로 피벗하지 않으며,
    시각당 한 행의 기상 변수 컬럼을 그대로 사용합니다.

    LDAPS 컬럼과 이름이 충돌하지 않도록 원본 GFS 변수에는
    gfs_ 접두사를 붙입니다. 이미 gfs_로 시작하는 파생 피처는
    그대로 유지합니다.
    """

    result = gfs_df.copy()

    required_cols = [
        "forecast_kst_dtm",
        "grid_id"
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in result.columns
    ]

    if missing_cols:
        raise KeyError(
            "GFS 처리에 필요한 컬럼이 없습니다: "
            f"{missing_cols}"
        )

    result["forecast_kst_dtm"] = pd.to_datetime(
        result["forecast_kst_dtm"],
        errors="raise"
    )

    result["grid_id"] = normalize_grid_id(
        result["grid_id"]
    )

    selected = result[
        result["grid_id"] == 5
    ].copy()

    if selected.empty:
        raise ValueError(
            "GFS 데이터에 5번 격자가 없습니다."
        )

    if selected[
        "forecast_kst_dtm"
    ].duplicated().any():
        raise ValueError(
            "GFS 5번 격자에 중복된 예측 시각이 있습니다."
        )

    weather_cols = get_pivot_weather_columns(
        selected
    )

    gfs_grid5 = selected[
        [
            "forecast_kst_dtm",
            *weather_cols
        ]
    ].copy()

    rename_map = {}

    for col in weather_cols:
        if col.startswith("gfs_"):
            rename_map[col] = col
        else:
            rename_map[col] = f"gfs_{col}"

    gfs_grid5 = gfs_grid5.rename(
        columns=rename_map
    )

    gfs_grid5 = (
        gfs_grid5
        .sort_values("forecast_kst_dtm")
        .reset_index(drop=True)
    )

    return gfs_grid5

# =========================================================
# 이동 평균 구하기
# =========================================================

def add_wind_rolling_features(
    df,
    grid_ids,
    time_col="forecast_kst_dtm"
):
    """
    Wide 변환 및 LDAPS-GFS 병합이 완료된 데이터에
    중심 이동평균 풍속 피처를 추가합니다.

    대상:
    - LDAPS 10m 풍속
    - LDAPS 50m 평균 풍속
    - GFS 80m 풍속

    생성:
    - window=7, center=True → _roll3h
    - window=13, center=True → _roll6h
    """

    result = df.copy()

    if time_col not in result.columns:
        raise KeyError(
            f"이동평균 계산에 필요한 '{time_col}' "
            "컬럼이 없습니다."
        )

    result[time_col] = pd.to_datetime(
        result[time_col],
        errors="raise"
    )

    # Wide 데이터는 시각당 한 행이어야 함
    if result[time_col].duplicated().any():
        raise ValueError(
            "이동평균 계산 전에 예측 시각이 "
            "시각당 한 행이어야 합니다."
        )

    # 이동평균은 반드시 시간순으로 계산
    result = (
        result
        .sort_values(time_col)
        .reset_index(drop=True)
    )

    target_wind_cols = []

    # 그룹에 해당하는 LDAPS 격자별 풍속 컬럼 구성
    for grid_id in grid_ids:
        target_wind_cols.extend([
            f"ldaps_10m_wind_speed_grid{grid_id}",
            f"ldaps_50m_mean_wind_speed_grid{grid_id}"
        ])

    # GFS는 모든 그룹이 동일한 grid 5를 사용
    target_wind_cols.append(
        "gfs_80m_wind_speed"
    )

    # 대상 컬럼 존재 여부 확인
    missing_cols = [
        col
        for col in target_wind_cols
        if col not in result.columns
    ]

    if missing_cols:
        raise KeyError(
            "이동평균 대상 풍속 컬럼이 없습니다: "
            f"{missing_cols}"
        )

    rolling_windows = {
        "roll3h": 7,
        "roll6h": 13
    }

    for wind_col in target_wind_cols:
        wind_values = pd.to_numeric(
            result[wind_col],
            errors="coerce"
        )

        for suffix, window_size in rolling_windows.items():
            result[
                f"{wind_col}_{suffix}"
            ] = wind_values.rolling(
                window=window_size,
                center=True,
                min_periods=1
            ).mean()

    # 이동평균으로 인해 새 결측치가 생기지 않았는지 확인
    rolling_cols = [
        col
        for col in result.columns
        if col.endswith(
            ("_roll3h", "_roll6h")
        )
    ]

    if result[rolling_cols].isna().any().any():
        raise ValueError(
            "이동평균 생성 후 결측치가 남아 있습니다."
        )

    return result


# =========================================================
# 6. 공통 파일 처리 및 그룹별 병합 함수
# =========================================================
def read_csv_utf8(file_path):
    return pd.read_csv(
        file_path,
        encoding="utf-8-sig"
    )


def check_required_files(
    file_paths,
    description
):
    missing_files = [
        path.name
        for path in file_paths
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"{description} 파일이 없습니다: "
            + ", ".join(missing_files)
        )


def resolve_split_dir(input_dir, split):
    """
    Return the directory containing a split's input files.

    Both layouts below are supported:
    - <input_dir>/train/*.csv and <input_dir>/test/*.csv
    - <input_dir>/*_train.csv and <input_dir>/*_test.csv
    """

    input_dir = Path(input_dir)
    split_dir = input_dir / split

    if split_dir.is_dir():
        return split_dir

    return input_dir


def preprocess_weather_split(
    split,
    input_dir,
    fallback_values
):
    """
    split만 train 또는 test로 바꾸어
    같은 전처리 함수로 LDAPS와 GFS를 처리합니다.

    반환값은 아직 Long Format인 전처리 완료 데이터입니다.
    """

    split = split.lower()

    if split not in {"train", "test"}:
        raise ValueError(
            "split은 'train' 또는 'test'여야 합니다."
        )

    ldaps_path = (
        input_dir
        / f"ldaps_{split}.csv"
    )

    gfs_path = (
        input_dir
        / f"gfs_{split}.csv"
    )

    check_required_files(
        [ldaps_path, gfs_path],
        f"{split.upper()} 기상"
    )

    ldaps_raw = read_csv_utf8(
        ldaps_path
    )

    gfs_raw = read_csv_utf8(
        gfs_path
    )

    ldaps_processed = preprocess_weather(
        df=ldaps_raw,
        source="ldaps",
        fallback_values=fallback_values[
            "ldaps"
        ]
    )

    gfs_processed = preprocess_weather(
        df=gfs_raw,
        source="gfs",
        fallback_values=fallback_values[
            "gfs"
        ]
    )

    del ldaps_raw, gfs_raw
    gc.collect()

    return ldaps_processed, gfs_processed



# =========================================================
# 7. 그룹별 결과 검증 및 출력
# =========================================================
def print_group_dataset_summary(
    group_results,
    split
):
    split = split.lower()

    print(
        f"\n[최종 {split.upper()} 그룹별 데이터]"
    )

    for group_name, group_df in group_results.items():
        group_id = int(
            group_name.replace(
                "group",
                ""
            )
        )

        print(
            f"\n- Group {group_id}"
        )

        print(
            "  Shape:",
            group_df.shape
        )

        print(
            "  시간 중복 수:",
            int(
                group_df[
                    "forecast_kst_dtm"
                ].duplicated().sum()
            )
        )

        if split == "train":
            target_col = GROUP_TARGET_COLS[
                group_id
            ]

            feature_cols = [
                col
                for col in group_df.columns
                if col != target_col
            ]

            print(
                "  기상 피처 결측치 수:",
                int(
                    group_df[
                        feature_cols
                    ].isna().sum().sum()
                )
            )

            print(
                "  Target 결측치 수:",
                int(
                    group_df[
                        target_col
                    ].isna().sum()
                )
            )

        else:
            print(
                "  전체 결측치 수:",
                int(
                    group_df.isna().sum().sum()
                )
            )


# =========================================================
# 통합 전 확인
# =========================================================
# 위쪽 SCADA 설정에서 반드시 다음 상태여야 합니다.
#
# LONG_OUTAGE_HOURS = 24
#
# FAULT_STATUSES = {
#     "장기연속정지"
# }
#
# 즉, 24시간 이상 이어진 장기연속정지만 NaN 처리하고
# 개별정지_고장정비후보는 원본값을 유지합니다.


# =========================================================
# 1. 제조사별 SCADA 실행 함수
# 기존 process_manufacturer()를 이 함수로 교체
# =========================================================
def validate_train_test_feature_schema(train_results, test_results):
    """
    Ensure every test group has exactly the same ordered feature columns
    as its matching train group. The train-only target is excluded.
    """

    for group_id in [1, 2, 3]:
        group_name = f"group{group_id}"
        target_col = GROUP_TARGET_COLS[group_id]

        train_feature_cols = [
            col
            for col in train_results[group_name].columns
            if col != target_col
        ]

        test_feature_cols = (
            test_results[group_name]
            .columns
            .tolist()
        )

        if train_feature_cols != test_feature_cols:
            missing_in_test = [
                col
                for col in train_feature_cols
                if col not in test_feature_cols
            ]

            extra_in_test = [
                col
                for col in test_feature_cols
                if col not in train_feature_cols
            ]

            raise ValueError(
                f"Group {group_id} train/test feature schema mismatch. "
                f"Missing in test: {missing_in_test}; "
                f"Extra in test: {extra_in_test}"
            )


def process_manufacturer(
    manufacturer,
    ldaps_weather,
    input_dir,
    save_dir
):
    input_dir = Path(input_dir)
    save_dir = Path(save_dir)

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    config = SCADA_CONFIG[
        manufacturer
    ]

    input_path = (
        input_dir
        / config["file_name"]
    )

    print(
        f"\n[{manufacturer.upper()} 처리 시작]"
    )

    raw = pd.read_csv(
        input_path,
        encoding="utf-8-sig"
    )

    if "kst_dtm" not in raw.columns:
        raise KeyError(
            f"{config['file_name']}에 "
            "kst_dtm이 없습니다."
        )

    long_df = scada_wide_to_long(
        scada_df=raw,
        manufacturer=manufacturer
    )

    long_df = attach_weather(
        scada_long=long_df,
        ldaps_weather=ldaps_weather
    )

    diagnostics = classify_and_clean(
        long_df
    )

    cleaned = rebuild_cleaned_wide(
        original_df=raw,
        classified_long=diagnostics
    )

    cleaned_path = (
        save_dir
        / (
            f"scada_{manufacturer}"
            "_train_cleaned.csv"
        )
    )

    cleaned.to_csv(
        cleaned_path,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S"
    )

    diagnostic_cols = [
        "kst_dtm",
        "manufacturer",
        "turbine_id",
        "turbine_no",
        "kpx_group",
        "grid_id",
        "power_raw",
        "power",
        "power_clean",
        "is_power_outlier",
        "is_negative_clipped",
        "rated_10min",
        "power_efficiency",
        "ws",
        "wd",
        "temp_kelvin",
        "humidity",
        "peer_active_count",
        "peer_valid_count",
        "peer_active_ratio",
        "is_low_output",
        "is_icing",
        "is_low_wind",
        "is_long_outage_candidate",
        "is_long_outage",
        "low_output_streak_hours",
        "status"
    ]

    if SAVE_DIAGNOSTICS:
        diagnostics_path = (
            save_dir
            / (
                f"scada_{manufacturer}"
                "_zero_diagnostics.csv"
            )
        )

        diagnostics[
            diagnostic_cols
        ].to_csv(
            diagnostics_path,
            index=False,
            encoding="utf-8-sig",
            date_format="%Y-%m-%d %H:%M:%S"
        )

    original_nan = int(
        diagnostics[
            "power_raw"
        ].isna().sum()
    )

    cleaned_nan = int(
        diagnostics[
            "power_clean"
        ].isna().sum()
    )

    print(
        diagnostics[
            "status"
        ].value_counts(
            dropna=False
        )
    )

    print(
        "새로 NaN 처리된 발전량:",
        cleaned_nan - original_nan
    )

    print(
        "장기연속정지 터빈·시각 행 수:",
        int(
            diagnostics[
                "is_long_outage"
            ].sum()
        )
    )

    print(
        "저장 완료:",
        cleaned_path.name
    )

    del raw, long_df
    gc.collect()

    return cleaned, diagnostics


# =========================================================
# 2. 최종 그룹 파일에서 SCADA 관련 열 차단
# =========================================================
FINAL_FORBIDDEN_COLUMN_TOKENS = (
    "scada",
    "vestas_wtg",
    "unison_wtg",
    "outage_turbine_count",
    "available_turbine_count",
    "fault_maintenance_flag",
    "long_outage",
    "power_clean",
    "power_raw",
    "peer_active"
)


def assert_no_scada_columns(
    df,
    data_name
):
    """
    최종 모델 입력 파일에 SCADA 원자료나
    장기정지 판단용 보조 열이 남아 있지 않은지 검사합니다.

    최종 그룹별 Train/Test 파일은 다음으로만 구성되어야 합니다.
    - LDAPS/GFS 및 파생 기상 피처
    - forecast_kst_dtm
    - Train의 해당 그룹 Target
    """

    forbidden_cols = [
        col
        for col in df.columns
        if any(
            token in str(col).lower()
            for token in FINAL_FORBIDDEN_COLUMN_TOKENS
        )
    ]

    if forbidden_cols:
        raise RuntimeError(
            f"{data_name}에 SCADA 또는 장기정지 보조 열이 "
            f"남아 있습니다: {forbidden_cols}"
        )


# =========================================================
# 3. 그룹별 장기정지 행 제외
# =========================================================
def remove_group_long_outage_rows(
    group_df,
    group_id,
    hourly_outage_summary
):
    """
    Group 1·2·3에서 해당 그룹의 장기연속정지 터빈이
    한 대라도 존재하는 시간의 학습 행을 제외합니다.

    중요:
    - Group 1 장기정지는 Group 1 행에만 적용
    - Group 2 장기정지는 Group 2 행에만 적용
    - Group 3 UNISON 장기정지는 Group 3 행에만 적용
    - 동일 시각의 다른 그룹 행은 유지
    - SCADA 요약 열은 필터링 직후 삭제
    """

    result = group_df.copy()

    if group_id not in KPX_GROUP_TURBINE_COUNTS:
        raise ValueError(
            "group_id는 1, 2, 3 중 하나여야 합니다."
        )

    if hourly_outage_summary is None:
        raise ValueError(
            f"Train Group {group_id} 장기정지 행 제외에 "
            "hourly_outage_summary가 필요합니다."
        )

    summary = hourly_outage_summary.copy()
    outage_col = (
        f"group{group_id}"
        "_outage_turbine_count"
    )

    required_cols = [
        "forecast_kst_dtm",
        outage_col
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in summary.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Group {group_id} 장기정지 행 제외에 필요한 "
            f"요약 컬럼이 없습니다: {missing_cols}"
        )

    result["forecast_kst_dtm"] = pd.to_datetime(
        result["forecast_kst_dtm"],
        errors="raise"
    )

    summary["forecast_kst_dtm"] = pd.to_datetime(
        summary["forecast_kst_dtm"],
        errors="raise"
    )

    if result[
        "forecast_kst_dtm"
    ].duplicated().any():
        raise ValueError(
            f"Group {group_id} 데이터에 중복 시각이 있습니다."
        )

    if summary[
        "forecast_kst_dtm"
    ].duplicated().any():
        raise ValueError(
            "시간별 장기정지 요약에 중복 시각이 있습니다."
        )

    before_count = len(result)

    result = result.merge(
        summary[required_cols],
        on="forecast_kst_dtm",
        how="left",
        validate="one_to_one"
    )

    if len(result) != before_count:
        raise RuntimeError(
            f"Group {group_id} 장기정지 요약 병합 중 "
            "행 수가 변경되었습니다."
        )

    # SCADA와 시각이 매칭되지 않으면 장기정지 없음으로 처리
    result[outage_col] = (
        pd.to_numeric(
            result[outage_col],
            errors="coerce"
        )
        .fillna(0)
    )

    invalid_mask = (
        (result[outage_col] < 0)
        | (
            result[outage_col]
            > KPX_GROUP_TURBINE_COUNTS[group_id]
        )
    )

    if invalid_mask.any():
        raise ValueError(
            f"Group {group_id} 장기정지 터빈 수가 "
            "유효 범위를 벗어났습니다."
        )

    long_outage_mask = (
        result[outage_col] > 0
    )

    removed_count = int(
        long_outage_mask.sum()
    )

    # 장기정지 터빈이 한 대라도 있는 해당 그룹 행만 제외
    result = (
        result.loc[
            ~long_outage_mask
        ]
        .drop(columns=outage_col)
        .reset_index(drop=True)
    )

    return result, removed_count


# =========================================================
# 4. 그룹별 최종 데이터 생성
# 기존 build_group_dataset()을 이 함수로 교체
# =========================================================
def build_group_dataset(
    ldaps_processed,
    gfs_grid5,
    group_id,
    train_labels=None,
    hourly_outage_summary=None
):
    """
    그룹별 최종 데이터프레임을 생성합니다.

    Train:
    - 그룹 관련 LDAPS Wide
    - GFS grid 5
    - 이동평균 피처
    - 원래 Group Target
    - Group 1·2·3은 장기연속정지 터빈이 한 대라도 있는
      해당 그룹·시간 행을 삭제
    - Group 3은 UNISON 장기정지 정보를 사용
    - 원래 Target 결측 행 제외

    Test:
    - 기상 피처만 반환
    - SCADA 정보를 사용하지 않음

    최종 반환값에는 SCADA 원자료, 장기정지 플래그,
    장기정지/가동 터빈 수가 포함되지 않습니다.
    """

    if group_id not in GROUP_TARGET_COLS:
        raise ValueError(
            "group_id는 1, 2, 3 중 하나여야 합니다."
        )

    ldaps_wide = pivot_ldaps_group_to_wide(
        ldaps_df=ldaps_processed,
        group_id=group_id
    )

    assert_same_forecast_times(
        left_df=ldaps_wide,
        right_df=gfs_grid5,
        left_name=f"LDAPS Group {group_id}",
        right_name="GFS Grid 5"
    )

    group_df = ldaps_wide.merge(
        gfs_grid5,
        on="forecast_kst_dtm",
        how="inner",
        validate="one_to_one"
    )

    group_df = add_wind_rolling_features(
        df=group_df,
        grid_ids=LDAPS_GROUP_GRIDS[
            group_id
        ]
    )

    # Test는 Label과 SCADA가 없으므로 기상 피처만 반환
    if train_labels is None:
        assert_no_scada_columns(
            df=group_df,
            data_name=f"Test Group {group_id}"
        )
        return group_df

    target_col = GROUP_TARGET_COLS[
        group_id
    ]

    group_labels = train_labels[
        [
            "forecast_kst_dtm",
            target_col
        ]
    ].copy()

    assert_same_forecast_times(
        left_df=group_df,
        right_df=group_labels,
        left_name=f"Group {group_id} 기상",
        right_name=target_col
    )

    final_group_df = group_df.merge(
        group_labels,
        on="forecast_kst_dtm",
        how="left",
        validate="one_to_one"
    )

    # -----------------------------------------------------
    # Group 1·2·3 장기연속정지 행을 그룹별로 제외
    # -----------------------------------------------------
    final_group_df, outage_removed_count = (
        remove_group_long_outage_rows(
            group_df=final_group_df,
            group_id=group_id,
            hourly_outage_summary=(
                hourly_outage_summary
            )
        )
    )

    print(
        f"Group {group_id} 장기연속정지 행 제외:",
        f"{outage_removed_count:,}행"
    )

    # -----------------------------------------------------
    # 원래 Target 결측 행 제외
    # - Group 3의 2022년 Label 결측 포함
    # -----------------------------------------------------
    before_target_drop = len(
        final_group_df
    )

    final_group_df = (
        final_group_df[
            final_group_df[
                target_col
            ].notna()
        ]
        .copy()
        .reset_index(drop=True)
    )

    target_missing_removed = (
        before_target_drop
        - len(final_group_df)
    )

    print(
        f"Group {group_id} 원래 Target 결측 행 제외:",
        f"{target_missing_removed:,}행"
    )

    # Target을 마지막 컬럼에 배치
    feature_cols = [
        col
        for col in final_group_df.columns
        if col != target_col
    ]

    final_group_df = final_group_df[
        feature_cols + [target_col]
    ]

    # 최종 파일에 SCADA 관련 열이 남는 것을 명시적으로 차단
    assert_no_scada_columns(
        df=final_group_df,
        data_name=f"Train Group {group_id}"
    )

    return final_group_df


# =========================================================
# 5. Train/Test 그룹별 처리
# 기존 process_grouped_dataset()을 이 함수로 교체
# =========================================================
def process_grouped_dataset(
    split,
    input_dir,
    save_dir,
    fallback_values,
    hourly_outage_summary=None,
    label_upper_ratio=1.05
):
    """
    Train/Test를 동일한 기상 전처리 구조로 처리합니다.

    Train:
    - train_labels 값은 비례 보정하지 않음
    - Group 1·2·3의 장기연속정지 행을 그룹별로 제외
    - Group 3은 UNISON 장기정지 정보를 사용
    - 최종 파일에 SCADA 관련 열을 포함하지 않음

    Test:
    - SCADA 정보를 사용하지 않음
    - 기상 피처만 저장
    """

    split = split.lower()
    input_dir = Path(input_dir)
    save_dir = Path(save_dir)

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    ldaps_processed, gfs_processed = (
        preprocess_weather_split(
            split=split,
            input_dir=input_dir,
            fallback_values=fallback_values
        )
    )

    gfs_grid5 = prepare_gfs_grid5(
        gfs_processed
    )

    train_labels = None

    if split == "train":
        labels_path = (
            input_dir
            / "train_labels.csv"
        )

        check_required_files(
            [labels_path],
            "TRAIN Label"
        )

        # 하한/상한 clip만 적용하고 비례 보정은 하지 않음
        train_labels = (
            preprocess_train_labels(
                df=read_csv_utf8(
                    labels_path
                ),
                upper_ratio=label_upper_ratio
            )
            .rename(
                columns={
                    "kst_dtm":
                    "forecast_kst_dtm"
                }
            )
        )

        if train_labels[
            "forecast_kst_dtm"
        ].duplicated().any():
            raise ValueError(
                "train_labels에 중복된 시각이 있습니다."
            )

        if hourly_outage_summary is None:
            raise ValueError(
                "Train Group 1·2·3 장기정지 행 제외에 "
                "hourly_outage_summary가 필요합니다."
            )

    elif split == "test":
        if hourly_outage_summary is not None:
            raise ValueError(
                "Test에는 SCADA 장기정지 요약을 "
                "전달하면 안 됩니다."
            )

    else:
        raise ValueError(
            "split은 'train' 또는 'test'여야 합니다."
        )

    group_results = {}

    for group_id in [1, 2, 3]:
        group_name = (
            f"group{group_id}"
        )

        group_df = build_group_dataset(
            ldaps_processed=ldaps_processed,
            gfs_grid5=gfs_grid5,
            group_id=group_id,
            train_labels=train_labels,
            hourly_outage_summary=(
                hourly_outage_summary
                if split == "train"
                else None
            )
        )

        # 저장 직전에도 한 번 더 검사
        assert_no_scada_columns(
            df=group_df,
            data_name=(
                f"{split.upper()} Group {group_id}"
            )
        )

        output_path = (
            save_dir
            / (
                f"{split}_{group_name}"
                "_preprocessed.csv"
            )
        )

        group_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
            date_format="%Y-%m-%d %H:%M:%S"
        )

        group_results[
            group_name
        ] = group_df

    del (
        ldaps_processed,
        gfs_processed,
        gfs_grid5
    )

    gc.collect()

    return group_results


# =========================================================
# 6. SCADA + 기상 + Label 전체 통합 실행 함수
# 기존 run_preprocessing()을 이 함수로 교체
# =========================================================
def run_preprocessing(
    input_dir=".",
    save_dir="./preprocessed",
    process_test_if_available=True
):
    """
    하나의 함수에서 다음을 순서대로 수행합니다.

    1. SCADA 저출력 원인 분류
    2. 24시간 이상 장기연속정지 판정
    3. VESTAS Group 1·2와 UNISON Group 3의
       시간별 장기정지 터빈 수 계산
    4. SCADA cleaned CSV와 장기정지 요약을 별도 저장
    5. Train 기상변수 전처리
    6. Group 1·2·3에서 장기정지 터빈이 한 대라도 있는
       해당 그룹·시간 학습 행 삭제
    7. train_labels 값은 비례 보정하지 않고 원값 사용
    8. Group 3 원래 Target 결측 제외
    9. Test 기상변수 전처리
    10. SCADA 열이 없는 그룹별 최종 CSV 저장

    주의:
    - Group 1 장기정지는 Group 1 Train에서만 삭제합니다.
    - Group 2 장기정지는 Group 2 Train에서만 삭제합니다.
    - Group 3 UNISON 장기정지는 Group 3 Train에서만 삭제합니다.
    - 같은 시각의 다른 그룹 행은 유지합니다.
    - Test에는 SCADA 정보를 사용하지 않습니다.
    """

    input_dir = Path(
        input_dir
    )

    save_dir = Path(
        save_dir
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_dir = resolve_split_dir(input_dir, "train")
    test_dir = resolve_split_dir(input_dir, "test")

    required_train_paths = [
        train_dir / "ldaps_train.csv",
        train_dir / "gfs_train.csv",
        train_dir / "train_labels.csv",
        train_dir / "scada_vestas_train.csv",
        train_dir / "scada_unison_train.csv"
    ]

    check_required_files(
        required_train_paths,
        "필수 TRAIN 및 SCADA"
    )

    # =====================================================
    # 1단계: SCADA 전처리
    # =====================================================
    print("\n" + "=" * 70)
    print("1단계: SCADA 장기 고장·정비 판정")
    print("=" * 70)

    ldaps_icing_weather = (
        prepare_ldaps_icing_weather(
            train_dir
            / "ldaps_train.csv"
        )
    )

    # VESTAS: Group 1·2의 장기정지 판단에 사용
    (
        vestas_cleaned,
        vestas_diagnostics
    ) = process_manufacturer(
        manufacturer="vestas",
        ldaps_weather=ldaps_icing_weather,
        input_dir=train_dir,
        save_dir=save_dir
    )

    vestas_hourly_outage_summary = (
        build_hourly_outage_summary(
            diagnostics=(
                vestas_diagnostics
            ),
            group_ids=[1, 2]
        )
    )

    # UNISON: Group 3의 장기정지 판단에 사용
    (
        unison_cleaned,
        unison_diagnostics
    ) = process_manufacturer(
        manufacturer="unison",
        ldaps_weather=ldaps_icing_weather,
        input_dir=train_dir,
        save_dir=save_dir
    )

    unison_hourly_outage_summary = (
        build_hourly_outage_summary(
            diagnostics=(
                unison_diagnostics
            ),
            group_ids=[3]
        )
    )

    # VESTAS Group 1·2와 UNISON Group 3 요약을
    # 하나의 시간별 요약으로 통합
    hourly_outage_summary = (
        vestas_hourly_outage_summary
        .merge(
            unison_hourly_outage_summary,
            on="forecast_kst_dtm",
            how="outer",
            validate="one_to_one"
        )
        .sort_values("forecast_kst_dtm")
        .reset_index(drop=True)
    )

    # UNISON은 2023년부터 제공되므로 2022년의 Group 3
    # 요약은 장기정지 0대로 채웁니다.
    for group_id, total_turbines in (
        KPX_GROUP_TURBINE_COUNTS.items()
    ):
        outage_col = (
            f"group{group_id}_outage_turbine_count"
        )
        available_col = (
            f"group{group_id}_available_turbine_count"
        )
        flag_col = (
            f"group{group_id}_fault_maintenance_flag"
        )

        hourly_outage_summary[outage_col] = (
            hourly_outage_summary[outage_col]
            .fillna(0)
            .astype("int8")
        )
        hourly_outage_summary[available_col] = (
            total_turbines
            - hourly_outage_summary[outage_col]
        ).astype("int8")
        hourly_outage_summary[flag_col] = (
            hourly_outage_summary[outage_col] > 0
        )

    print(
        "\n[시간별 장기정지 요약]"
    )

    print(
        hourly_outage_summary[
            [
                "group1_outage_turbine_count",
                "group2_outage_turbine_count",
                "group3_outage_turbine_count"
            ]
        ].describe()
    )

    outage_summary_path = (
        save_dir
        / "scada_hourly_outage_summary.csv"
    )

    # 이 파일은 진단용 별도 파일이며,
    # 최종 모델 입력 파일에는 병합하지 않음
    hourly_outage_summary.to_csv(
        outage_summary_path,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S"
    )

    print(
        "시간별 장기정지 요약 저장:",
        outage_summary_path.name
    )

    del (
        ldaps_icing_weather,
        vestas_cleaned,
        vestas_diagnostics,
        vestas_hourly_outage_summary,
        unison_cleaned,
        unison_diagnostics,
        unison_hourly_outage_summary
    )

    gc.collect()

    # =====================================================
    # 2단계: 기상변수 fallback 계산
    # =====================================================
    print("\n" + "=" * 70)
    print("2단계: Train 기상 fallback 계산")
    print("=" * 70)

    ldaps_train_raw = read_csv_utf8(
        train_dir
        / "ldaps_train.csv"
    )

    gfs_train_raw = read_csv_utf8(
        train_dir
        / "gfs_train.csv"
    )

    fallback_values = {
        "ldaps":
        calculate_train_fallbacks(
            train_df=ldaps_train_raw,
            source="ldaps"
        ),

        "gfs":
        calculate_train_fallbacks(
            train_df=gfs_train_raw,
            source="gfs"
        )
    }

    del (
        ldaps_train_raw,
        gfs_train_raw
    )

    gc.collect()

    results = {
        "hourly_outage_summary":
        hourly_outage_summary
    }

    # =====================================================
    # 3단계: Train 전처리
    # =====================================================
    print("\n" + "=" * 70)
    print("3단계: Train 그룹별 전처리")
    print("=" * 70)

    results["train"] = (
        process_grouped_dataset(
            split="train",
            input_dir=train_dir,
            save_dir=save_dir,
            fallback_values=fallback_values,
            hourly_outage_summary=(
                hourly_outage_summary
            )
        )
    )

    # =====================================================
    # 4단계: Test 전처리
    # =====================================================
    ldaps_test_path = (
        test_dir
        / "ldaps_test.csv"
    )

    gfs_test_path = (
        test_dir
        / "gfs_test.csv"
    )

    test_file_exists = [
        ldaps_test_path.exists(),
        gfs_test_path.exists()
    ]

    if process_test_if_available:
        if all(test_file_exists):
            print("\n" + "=" * 70)
            print("4단계: Test 그룹별 전처리")
            print("=" * 70)

            results["test"] = (
                process_grouped_dataset(
                    split="test",
                    input_dir=test_dir,
                    save_dir=save_dir,
                    fallback_values=(
                        fallback_values
                    ),
                    hourly_outage_summary=None
                )
            )

            validate_train_test_feature_schema(
                train_results=results["train"],
                test_results=results["test"]
            )

        elif any(test_file_exists):
            raise FileNotFoundError(
                "Test 파일은 LDAPS와 GFS 두 개가 "
                "모두 필요합니다. 현재 하나만 존재합니다."
            )

    # =====================================================
    # 5단계: 최종 결과 출력
    # =====================================================
    print("\n" + "=" * 70)
    print("전체 전처리 및 저장 완료")
    print("저장 위치:", save_dir)
    print("=" * 70)

    print_group_dataset_summary(
        results["train"],
        split="train"
    )

    if "test" in results:
        print_group_dataset_summary(
            results["test"],
            split="test"
        )
    else:
        print(
            "\n[Test 데이터 없음: "
            "Train 그룹별 데이터만 전처리했습니다.]"
        )

    print("\n저장된 파일:")

    for file_path in sorted(
        save_dir.glob("*.csv")
    ):
        print(
            "-",
            file_path.name
        )

    return results


# =========================================================


# =============================================================================
# 2. 장기정지 행을 삭제하지 않는 전처리 실행
# =============================================================================

PREPROCESSED_DIR = REPO_ROOT / "wind_baram_v2_preprocessed"
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TIME_COLUMN = "forecast_kst_dtm"
GROUP_TARGETS = {1: "kpx_group_1", 2: "kpx_group_2", 3: "kpx_group_3"}


def _keep_long_outage_rows(group_df, group_id, hourly_outage_summary):
    """장기정지 행을 제거하지 않고 모델 학습 후보로 복원합니다."""
    return group_df.reset_index(drop=True).copy(), 0


def _attach_long_outage_flag(frame, summary, group_id):
    """학습 가중치 계산에 필요한 장기정지 여부만 붙입니다."""
    outage_column = f"group{group_id}_outage_turbine_count"
    if outage_column not in summary.columns:
        raise KeyError(f"장기정지 요약에 {outage_column} 열이 없습니다.")
    result = frame.copy()
    result[TIME_COLUMN] = pd.to_datetime(result[TIME_COLUMN], errors="raise")
    small = summary[[TIME_COLUMN, outage_column]].copy()
    small[TIME_COLUMN] = pd.to_datetime(small[TIME_COLUMN], errors="raise")
    result = result.merge(small, on=TIME_COLUMN, how="left", validate="one_to_one")
    result["is_long_outage_hour"] = (
        result[outage_column].fillna(0).gt(0).astype("int8")
    )
    return result.drop(columns=outage_column)


print("\n[1/6] 원본 데이터 전처리")
_original_outage_removal = remove_group_long_outage_rows
remove_group_long_outage_rows = _keep_long_outage_rows
try:
    _restored = run_preprocessing(
        input_dir=WIND_DATA_DIR,
        save_dir=PREPROCESSED_DIR,
        process_test_if_available=True,
    )
finally:
    remove_group_long_outage_rows = _original_outage_removal

_outage_summary = _restored["hourly_outage_summary"].copy()
for _group_id in GROUP_TARGETS:
    _train = _attach_long_outage_flag(
        _restored["train"][f"group{_group_id}"], _outage_summary, _group_id
    )
    _test = _restored["test"][f"group{_group_id}"].copy()
    # 미래 Test에는 실제 장기정지 여부를 알 수 없으므로 입력 피처로 사용하지 않습니다.
    _test["is_long_outage_hour"] = np.int8(0)
    _train.to_csv(
        PREPROCESSED_DIR / f"train_group{_group_id}_preprocessed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _test.to_csv(
        PREPROCESSED_DIR / f"test_group{_group_id}_preprocessed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"  Group {_group_id}: train={len(_train):,}, "
        f"outage={int(_train['is_long_outage_hour'].sum()):,}, test={len(_test):,}"
    )

del _restored, _outage_summary, _train, _test
gc.collect()



# =============================================================================
# 3. 기준 모델 함수
# =============================================================================

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
    parser.add_argument("--preprocessed-dir", type=Path, default=Path.cwd() / "wind_baram_v2_preprocessed")
    parser.add_argument("--sample-submission", type=Path, default=Path.cwd() / "wind_data" / "sample_submission.csv")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "wind_baram_v2_output")
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


# =============================================================================
# 4. 기준 앙상블 모델 학습
# =============================================================================

print("\n[2/6] 기준 앙상블 모델 학습")
RANDOM_SEED = 42
OUTAGE_MODE = "weight"
OUTAGE_WEIGHT = 0.50
APPLY_PIPELINE_CALIBRATION = False
USE_GROUP3_GATE = True

_baseline_result = run_pipeline(
    preprocessed_dir=PREPROCESSED_DIR,
    sample_submission_path=WIND_DATA_DIR / "sample_submission.csv",
    output_dir=REPO_ROOT / "wind_baram_v2_output",
    make_submission=True,
    save_models=False,
)
del _baseline_result
gc.collect()



# =============================================================================
# 5. 통합 2-stage 후보 학습
# =============================================================================

# ---- 원본 노트북 코드 셀 2 ----
import gc
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from IPython.display import display
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = REPO_ROOT
DATA_DIR = ROOT / "wind_baram_v2_preprocessed"
SAMPLE_PATH = ROOT / "wind_data" / "sample_submission.csv"
OUTPUT_DIR = ROOT / "wind_hybrid_2stage_output"

SEED = 42
QUICK_MODE = False
SAVE_MODELS = False

TIME_COL = "forecast_kst_dtm"
OUTAGE_COL = "is_long_outage_hour"
TARGETS = {1: "kpx_group_1", 2: "kpx_group_2", 3: "kpx_group_3"}
CAPACITIES = {1: 21600.0, 2: 21600.0, 3: 21000.0}

ACTIVE_CF = 0.10
PREDICTION_MAX_CF = 1.05
TRAIN_END = pd.Timestamp("2024-01-01 01:00:00")
VALIDATION_END = pd.Timestamp("2025-01-01 01:00:00")
H1_END = pd.Timestamp("2024-07-01 01:00:00")

ANCHOR_FOLDS = 2 if QUICK_MODE else 4
AUXILIARY_FOLDS = 2
RESIDUAL_FOLDS = ANCHOR_FOLDS
EARLY_STOPPING_ROUNDS = 40 if QUICK_MODE else 180
LGB_MAX_ESTIMATORS = 180 if QUICK_MODE else 3500
CAT_MAX_ITERATIONS = 160 if QUICK_MODE else 2800
XGB_MAX_ESTIMATORS = 160 if QUICK_MODE else 2400
RESIDUAL_MAX_ESTIMATORS = 140 if QUICK_MODE else 1800

# 새 구조를 채택하기 위한 보수적인 독립 검증 기준입니다.
MIN_H1_SCORE_GAIN = 0.0000 if QUICK_MODE else 0.0010
MIN_H2_SCORE_GAIN = 0.0000 if QUICK_MODE else 0.0005
MAX_NMAE_WORSEN = 0.0020 if QUICK_MODE else 0.0003
MAX_FICR_WORSEN = 0.0100 if QUICK_MODE else 0.0010
MIN_TOTAL_MONTH_WINS = 0 if QUICK_MODE else 8
MIN_H2_MONTH_WINS = 0 if QUICK_MODE else 4

if not DATA_DIR.exists():
    raise FileNotFoundError(DATA_DIR)
if not SAMPLE_PATH.exists():
    raise FileNotFoundError(SAMPLE_PATH)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
started_all = time.perf_counter()

print("시드:", SEED)
print("QUICK_MODE:", QUICK_MODE)
print("Anchor Fold 수:", ANCHOR_FOLDS)
print("보조 후보 Fold 수:", AUXILIARY_FOLDS)
print("Group 1 잔차 Fold 수:", RESIDUAL_FOLDS)
print("출력 폴더:", OUTPUT_DIR)


# ---- 원본 노트북 코드 셀 4 ----
def read_group_csv(group_id, split):
    path = DATA_DIR / f"{split}_group{group_id}_preprocessed.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if TIME_COL not in frame:
        raise KeyError(f"{path.name}: {TIME_COL} 열이 없습니다.")

    frame[TIME_COL] = pd.to_datetime(frame[TIME_COL], errors="raise")
    frame = frame.sort_values(TIME_COL).reset_index(drop=True)

    if frame[TIME_COL].duplicated().any():
        raise ValueError(f"{path.name}: 중복 시간이 있습니다.")

    if split == "train":
        target = TARGETS[group_id]
        if target not in frame:
            raise KeyError(f"{path.name}: {target} 열이 없습니다.")
        frame[target] = pd.to_numeric(frame[target], errors="raise")

    return frame


train_frames = {g: read_group_csv(g, "train") for g in TARGETS}
test_frames = {g: read_group_csv(g, "test") for g in TARGETS}

audit_rows = []
for group_id in TARGETS:
    train = train_frames[group_id]
    times = train[TIME_COL]
    history_mask = times < TRAIN_END
    validation_mask = (times >= TRAIN_END) & (times < VALIDATION_END)
    outage = pd.to_numeric(
        train.get(OUTAGE_COL, pd.Series(0, index=train.index)),
        errors="coerce",
    ).fillna(0).gt(0)

    audit_rows.append(
        {
            "group": group_id,
            "전체_라벨행": len(train),
            "2022_2023_내부학습행": int(history_mask.sum()),
            "2024_외부검증행": int(validation_mask.sum()),
            "장기정지_복원행": int(outage.sum()),
            "삭제행": 0,
            "새모델_장기정지가중치": 1.0,
            "test행": len(test_frames[group_id]),
        }
    )

audit_df = pd.DataFrame(audit_rows)
display(audit_df)


# ---- 원본 노트북 코드 셀 6 ----
def official_metrics_cf(y_true_cf, y_pred_cf, capacity_kwh):
    y_true_cf = np.asarray(y_true_cf, dtype=float)
    y_pred_cf = np.asarray(y_pred_cf, dtype=float)

    finite = np.isfinite(y_true_cf) & np.isfinite(y_pred_cf)
    active = finite & (y_true_cf >= ACTIVE_CF)
    if not active.any():
        raise ValueError("공식 평가 대상 행이 없습니다.")

    actual = y_true_cf[active]
    pred = y_pred_cf[active]
    error = np.abs(pred - actual)

    rate = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
    actual_kwh = actual * capacity_kwh
    maximum = float((4.0 * actual_kwh).sum())

    nmae = float(error.mean())
    ficr = float((rate * actual_kwh).sum() / maximum)
    score = float(0.5 * (1.0 - nmae) + 0.5 * ficr)

    return {
        "rows": int(active.sum()),
        "MAE_kWh": float((error * capacity_kwh).mean()),
        "NMAE": nmae,
        "one_minus_NMAE": 1.0 - nmae,
        "FICR": ficr,
        "Score": score,
        "within_6_rate": float((error <= 0.06).mean()),
        "within_8_rate": float((error <= 0.08).mean()),
        "signed_bias_cf": float((pred - actual).mean()),
    }


def metrics_on_mask(y, pred, capacity, mask):
    mask = np.asarray(mask, dtype=bool)
    return official_metrics_cf(np.asarray(y)[mask], np.asarray(pred)[mask], capacity)


def monthly_score_wins(y, base, candidate, times, mask):
    table = pd.DataFrame(
        {
            "time": pd.to_datetime(times),
            "y": np.asarray(y, dtype=float),
            "base": np.asarray(base, dtype=float),
            "candidate": np.asarray(candidate, dtype=float),
            "use": np.asarray(mask, dtype=bool),
        }
    )
    table = table.loc[table["use"]].copy()
    table["month_key"] = table["time"].dt.to_period("M")

    wins = 0
    rows = []
    for month_key, part in table.groupby("month_key", sort=True):
        base_m = official_metrics_cf(part["y"], part["base"], 1.0)
        candidate_m = official_metrics_cf(part["y"], part["candidate"], 1.0)
        gain = candidate_m["Score"] - base_m["Score"]
        wins += int(gain > 0)
        rows.append(
            {
                "month": str(month_key),
                "base_score": base_m["Score"],
                "candidate_score": candidate_m["Score"],
                "gain": gain,
            }
        )
    return wins, rows


def threshold_transitions(y, base, candidate, mask):
    y = np.asarray(y, dtype=float)
    base = np.asarray(base, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    mask = np.asarray(mask, dtype=bool) & (y >= ACTIVE_CF)

    base_inside = np.abs(base[mask] - y[mask]) <= 0.08
    candidate_inside = np.abs(candidate[mask] - y[mask]) <= 0.08
    return {
        "rescued_outside_to_inside": int((~base_inside & candidate_inside).sum()),
        "lost_inside_to_outside": int((base_inside & ~candidate_inside).sum()),
    }


def approval_report(group_id, y, times, anchor, candidate, h1_mask, h2_mask):
    capacity = CAPACITIES[group_id]
    anchor_h1 = metrics_on_mask(y, anchor, capacity, h1_mask)
    candidate_h1 = metrics_on_mask(y, candidate, capacity, h1_mask)
    anchor_h2 = metrics_on_mask(y, anchor, capacity, h2_mask)
    candidate_h2 = metrics_on_mask(y, candidate, capacity, h2_mask)

    all_mask = np.asarray(h1_mask) | np.asarray(h2_mask)
    total_month_wins, monthly_rows = monthly_score_wins(
        y, anchor, candidate, times, all_mask
    )
    h2_month_wins, _ = monthly_score_wins(
        y, anchor, candidate, times, h2_mask
    )
    transitions = threshold_transitions(y, anchor, candidate, all_mask)

    passed = bool(
        candidate_h1["Score"] >= anchor_h1["Score"] + MIN_H1_SCORE_GAIN
        and candidate_h2["Score"] >= anchor_h2["Score"] + MIN_H2_SCORE_GAIN
        and candidate_h1["NMAE"] <= anchor_h1["NMAE"] + MAX_NMAE_WORSEN
        and candidate_h2["NMAE"] <= anchor_h2["NMAE"] + MAX_NMAE_WORSEN
        and candidate_h1["FICR"] >= anchor_h1["FICR"] - MAX_FICR_WORSEN
        and candidate_h2["FICR"] >= anchor_h2["FICR"] - MAX_FICR_WORSEN
        and total_month_wins >= MIN_TOTAL_MONTH_WINS
        and h2_month_wins >= MIN_H2_MONTH_WINS
        and transitions["rescued_outside_to_inside"]
            >= transitions["lost_inside_to_outside"]
    )

    return {
        "passed": passed,
        "anchor_H1": anchor_h1,
        "candidate_H1": candidate_h1,
        "anchor_H2": anchor_h2,
        "candidate_H2": candidate_h2,
        "H1_score_gain": candidate_h1["Score"] - anchor_h1["Score"],
        "H2_score_gain": candidate_h2["Score"] - anchor_h2["Score"],
        "total_month_wins": total_month_wins,
        "H2_month_wins": h2_month_wins,
        "transitions": transitions,
        "monthly": monthly_rows,
    }


# ---- 원본 노트북 코드 셀 8 ----
EXCLUDED_COLUMNS = {
    TIME_COL,
    OUTAGE_COL,
    "forecast_id",
    "index",
    "_split",
    *TARGETS.values(),
}


def matching_numeric_columns(frame, include_tokens):
    columns = []
    for column in frame.columns:
        lower = column.lower()
        if column in EXCLUDED_COLUMNS:
            continue
        if any(token in lower for token in include_tokens):
            if pd.api.types.is_numeric_dtype(frame[column]):
                columns.append(column)
    return columns


def add_row_statistics(features, source, columns, prefix):
    if not columns:
        return
    values = source[columns].apply(pd.to_numeric, errors="coerce")
    features[f"{prefix}_mean"] = values.mean(axis=1)
    features[f"{prefix}_std"] = values.std(axis=1)
    features[f"{prefix}_min"] = values.min(axis=1)
    features[f"{prefix}_max"] = values.max(axis=1)
    features[f"{prefix}_q25"] = values.quantile(0.25, axis=1)
    features[f"{prefix}_q75"] = values.quantile(0.75, axis=1)


def build_features(frame):
    timestamps = pd.to_datetime(frame[TIME_COL], errors="raise")
    numeric_columns = [
        column
        for column in frame.columns
        if column not in EXCLUDED_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    features = frame[numeric_columns].copy()

    hour = timestamps.dt.hour + timestamps.dt.minute / 60.0
    day_of_week = timestamps.dt.dayofweek
    day_of_year = timestamps.dt.dayofyear

    features["time_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    features["time_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    features["time_dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    features["time_dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
    features["time_doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    features["time_doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)
    features["time_year"] = timestamps.dt.year.astype(float)
    features["time_month"] = timestamps.dt.month.astype(float)

    wind_columns = matching_numeric_columns(
        frame,
        ["wind_speed", "50mumax", "50mumin", "50mvmax", "50mvmin"],
    )
    ldaps_wind = [c for c in wind_columns if c.lower().startswith("ldaps")]
    gfs_wind = [c for c in wind_columns if c.lower().startswith("gfs")]

    add_row_statistics(features, frame, wind_columns, "wind_all")
    add_row_statistics(features, frame, ldaps_wind, "wind_ldaps")
    add_row_statistics(features, frame, gfs_wind, "wind_gfs")

    add_row_statistics(
        features,
        frame,
        matching_numeric_columns(frame, ["_t_grid", "_dpt_", "_2t", "_2d"]),
        "temperature",
    )
    add_row_statistics(
        features,
        frame,
        matching_numeric_columns(frame, ["_r_grid", "_q_grid", "_2r", "_2sh"]),
        "humidity",
    )
    add_row_statistics(
        features,
        frame,
        matching_numeric_columns(frame, ["_sp_", "prmsl"]),
        "pressure",
    )
    add_row_statistics(
        features,
        frame,
        matching_numeric_columns(frame, ["_hcc", "_mcc", "_lcc", "_tcc", "vlcdc"]),
        "cloud",
    )

    for prefix in ["wind_all", "wind_ldaps", "wind_gfs"]:
        mean_column = f"{prefix}_mean"
        if mean_column in features:
            features[f"{prefix}_mean_sq"] = features[mean_column].pow(2)
            features[f"{prefix}_mean_cube"] = features[mean_column].pow(3)

    if "wind_ldaps_mean" in features and "wind_gfs_mean" in features:
        features["wind_source_difference"] = (
            features["wind_ldaps_mean"] - features["wind_gfs_mean"]
        )
        features["wind_source_ratio"] = (
            features["wind_ldaps_mean"]
            / (features["wind_gfs_mean"].abs() + 0.2)
        )

    sequence_columns = [
        column
        for column in [
            "wind_all_mean",
            "wind_ldaps_mean",
            "wind_gfs_mean",
            "wind_all_std",
        ]
        if column in features
    ]
    for column in sequence_columns:
        series = features[column]
        for lag in [1, 3]:
            features[f"{column}_lag{lag}"] = series.shift(lag)
            features[f"{column}_lead{lag}"] = series.shift(-lag)
            features[f"{column}_diff{lag}"] = series - series.shift(lag)
        features[f"{column}_roll3"] = series.rolling(
            3, center=True, min_periods=1
        ).mean()
        features[f"{column}_roll6"] = series.rolling(
            6, center=True, min_periods=1
        ).mean()

    return features.replace([np.inf, -np.inf], np.nan).astype("float32")


def prepare_group_data(group_id):
    train = train_frames[group_id].copy()
    test = test_frames[group_id].copy()
    target = TARGETS[group_id]

    train["_split"] = 0
    test["_split"] = 1
    combined = pd.concat([train, test], ignore_index=True, sort=False)
    features = build_features(combined)

    train_count = len(train)
    x_train = features.iloc[:train_count].reset_index(drop=True)
    x_test = features.iloc[train_count:].reset_index(drop=True)

    medians = x_train.median(numeric_only=True).fillna(0.0)
    x_train = x_train.fillna(medians).fillna(0.0)
    x_test = x_test.reindex(columns=x_train.columns).fillna(medians).fillna(0.0)

    y_cf = train[target].to_numpy(dtype=float) / CAPACITIES[group_id]
    outage = pd.to_numeric(
        train.get(OUTAGE_COL, pd.Series(0, index=train.index)),
        errors="coerce",
    ).fillna(0).gt(0).to_numpy()

    return {
        "x_train": x_train.astype("float32"),
        "x_test": x_test.astype("float32"),
        "y_cf": y_cf,
        "outage": outage,
        "times": train[TIME_COL].reset_index(drop=True),
        "test_times": test[TIME_COL].reset_index(drop=True),
    }


group_data = {g: prepare_group_data(g) for g in TARGETS}
for group_id, data in group_data.items():
    print(
        f"Group {group_id}: 피처 {data['x_train'].shape[1]:,}개, "
        f"라벨 {len(data['y_cf']):,}행"
    )


# ---- 원본 노트북 코드 셀 10 ----
@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    objective: str
    alpha: float | None = None
    outage_weight: float = 1.0
    weight_scheme: str = "uniform"
    variant: str = "standard"
    pooled: bool = False


GROUP_SPECS = {
    1: [
        ModelSpec("cat_q55_w05", "catboost", "quantile", 0.55, 0.5),
        ModelSpec("lgb_q65_w05", "lightgbm", "quantile", 0.65, 0.5),
        ModelSpec("cat_q55_w1", "catboost", "quantile", 0.55, 1.0),
        ModelSpec("lgb_q65_w1", "lightgbm", "quantile", 0.65, 1.0),
        ModelSpec(
            "uploaded_lgb_high_w1",
            "lightgbm",
            "mae",
            outage_weight=1.0,
            weight_scheme="group1_bins",
            variant="uploaded",
        ),
    ],
    2: [
        ModelSpec("lgb_deep_w05", "lightgbm", "mae", outage_weight=0.5, variant="deep"),
        ModelSpec("lgb_mae_w05", "lightgbm", "mae", outage_weight=0.5),
        ModelSpec("lgb_q58_w05", "lightgbm", "quantile", 0.58, 0.5),
        ModelSpec("cat_mae_w05", "catboost", "mae", outage_weight=0.5),
        ModelSpec(
            "uploaded_xgb_mse_w1",
            "xgboost",
            "mse",
            outage_weight=1.0,
            variant="uploaded",
        ),
    ],
    3: [
        ModelSpec("cat_q55_w05", "catboost", "quantile", 0.55, 0.5),
        ModelSpec(
            "uploaded_xgb_high_w1",
            "xgboost",
            "mse",
            outage_weight=1.0,
            weight_scheme="group3_bins",
            variant="group3",
        ),
    ],
}

POOLED_SPECS = [
    ModelSpec(
        "pooled_cat_w05",
        "catboost",
        "mae",
        outage_weight=0.5,
        variant="pooled",
        pooled=True,
    ),
    ModelSpec(
        "uploaded_pooled_xgb_w1",
        "xgboost",
        "mse",
        outage_weight=1.0,
        variant="uploaded",
        pooled=True,
    ),
]

ANCHOR_RECIPES = {
    1: {"cat_q55_w05": 0.40, "lgb_q65_w05": 0.60},
    2: {
        "lgb_deep_w05": 0.1431515556863846,
        "lgb_mae_w05": 0.29022053336280174,
        "lgb_q58_w05": 0.25907454872702446,
        "cat_mae_w05": 0.22922975196180245,
    },
    3: {"pooled_cat_w05": 0.90, "cat_q55_w05": 0.10},
}

CLONE_RECIPES = {
    1: {"cat_q55_w1": 0.40, "lgb_q65_w1": 0.60},
}

UPLOADED_RECIPES = {
    1: {"uploaded_lgb_high_w1": 1.00},
    # Group 2는 2024 장기정지 검증표본이 없으므로 별도 weight-1 Clone 대신
    # 기존 Anchor LightGBM을 재사용해 불필요한 세 모델 학습을 줄입니다.
    2: {"uploaded_xgb_mse_w1": 0.50, "lgb_mae_w05": 0.50},
    3: {"uploaded_xgb_high_w1": 0.75, "uploaded_pooled_xgb_w1": 0.25},
}


def normalize_recipe(recipe):
    total = float(sum(recipe.values()))
    if total <= 0:
        raise ValueError("레시피 가중치 합이 0입니다.")
    return {name: float(weight / total) for name, weight in recipe.items()}


ANCHOR_RECIPES = {g: normalize_recipe(r) for g, r in ANCHOR_RECIPES.items()}
CLONE_RECIPES = {g: normalize_recipe(r) for g, r in CLONE_RECIPES.items()}
UPLOADED_RECIPES = {g: normalize_recipe(r) for g, r in UPLOADED_RECIPES.items()}

RECIPE_MAPS = {}
for group_id in TARGETS:
    RECIPE_MAPS[group_id] = {
        "anchor": ANCHOR_RECIPES[group_id],
        "uploaded": UPLOADED_RECIPES[group_id],
    }
    if group_id in CLONE_RECIPES:
        RECIPE_MAPS[group_id]["clone"] = CLONE_RECIPES[group_id]

for group_id in TARGETS:
    print("Group", group_id)
    print("  anchor :", ANCHOR_RECIPES[group_id])
    if group_id in CLONE_RECIPES:
        print("  clone  :", CLONE_RECIPES[group_id])
    print("  upload :", UPLOADED_RECIPES[group_id])


# ---- 원본 노트북 코드 셀 12 ----
def make_training_weight(y_cf, outage, spec, group_ids=None):
    y_cf = np.asarray(y_cf, dtype=float)
    outage = np.asarray(outage, dtype=bool)
    weight = np.ones(len(y_cf), dtype=float)

    if spec.weight_scheme == "group1_bins":
        weight = np.select(
            [
                y_cf < 0.50,
                (y_cf >= 0.50) & (y_cf < 0.70),
                (y_cf >= 0.70) & (y_cf < 0.90),
                y_cf >= 0.90,
            ],
            [1.0, 1.2, 2.0, 6.0],
            default=1.0,
        ).astype(float)
    elif spec.weight_scheme == "group3_bins":
        weight = np.select(
            [
                y_cf < 0.50,
                (y_cf >= 0.50) & (y_cf < 0.70),
                (y_cf >= 0.70) & (y_cf < 0.90),
                y_cf >= 0.90,
            ],
            [1.0, 1.5, 2.5, 4.0],
            default=1.0,
        ).astype(float)
    elif spec.weight_scheme == "residual_high":
        weight = 1.0 + 1.5 * np.square(np.clip(y_cf, 0.0, 1.1))

    weight *= np.where(outage, spec.outage_weight, 1.0)

    if group_ids is not None:
        group_ids = np.asarray(group_ids)
        for group_id in np.unique(group_ids):
            mask = group_ids == group_id
            total = weight[mask].sum()
            if total > 0:
                weight[mask] /= total
        weight /= weight.mean()
    else:
        weight /= weight.mean()

    return weight


def max_iterations_for_spec(spec):
    if spec.variant == "residual":
        return RESIDUAL_MAX_ESTIMATORS
    if spec.family == "lightgbm":
        return LGB_MAX_ESTIMATORS
    if spec.family == "catboost":
        return CAT_MAX_ITERATIONS
    if spec.family == "xgboost":
        return XGB_MAX_ESTIMATORS
    raise ValueError(spec.family)


def create_model(spec, n_estimators, early_stopping):
    n_estimators = int(max(20, n_estimators))

    if spec.family == "lightgbm":
        params = {
            "objective": (
                "quantile"
                if spec.objective == "quantile"
                else ("regression" if spec.objective == "mse" else "regression_l1")
            ),
            "n_estimators": n_estimators,
            "learning_rate": 0.02,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 35,
            "subsample": 0.88,
            "subsample_freq": 1,
            "colsample_bytree": 0.82,
            "reg_alpha": 0.25,
            "reg_lambda": 2.5,
            "random_state": SEED,
            "n_jobs": -1,
            "verbosity": -1,
        }
        if spec.variant == "deep":
            params.update(
                learning_rate=0.015,
                num_leaves=95,
                min_child_samples=28,
                reg_alpha=0.35,
                reg_lambda=3.5,
            )
        elif spec.variant == "uploaded":
            params.update(
                learning_rate=0.03,
                num_leaves=31,
                max_depth=6,
                min_child_samples=100,
                subsample=0.80,
                colsample_bytree=0.75,
                reg_alpha=0.10,
                reg_lambda=10.0,
            )
        elif spec.variant == "residual":
            params.update(
                learning_rate=0.025,
                num_leaves=15,
                max_depth=5,
                min_child_samples=100,
                subsample=0.85,
                colsample_bytree=0.80,
                reg_alpha=0.50,
                reg_lambda=4.0,
            )
        if spec.alpha is not None:
            params["alpha"] = spec.alpha
        return lgb.LGBMRegressor(**params)

    if spec.family == "catboost":
        loss_function = (
            f"Quantile:alpha={spec.alpha}"
            if spec.objective == "quantile"
            else ("RMSE" if spec.objective == "mse" else "MAE")
        )
        params = {
            "loss_function": loss_function,
            "eval_metric": "MAE",
            "iterations": n_estimators,
            "learning_rate": 0.03,
            "depth": 8,
            "l2_leaf_reg": 8.0,
            "random_strength": 0.4,
            "random_seed": SEED,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.88,
            "rsm": 0.85,
            "allow_writing_files": False,
            "thread_count": -1,
            "verbose": False,
        }
        if spec.variant == "residual":
            params.update(depth=6, l2_leaf_reg=10.0, random_strength=0.25)
        if early_stopping:
            params.update(od_type="Iter", od_wait=EARLY_STOPPING_ROUNDS)
        return CatBoostRegressor(**params)

    if spec.family == "xgboost":
        group3_regularized = spec.variant == "group3"
        params = {
            "objective": "reg:squarederror",
            "n_estimators": n_estimators,
            "learning_rate": 0.03,
            "max_depth": 4 if group3_regularized else 5,
            "min_child_weight": 15 if group3_regularized else 10,
            "subsample": 0.80,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.10,
            "reg_lambda": 15.0 if group3_regularized else 10.0,
            "tree_method": "hist",
            "eval_metric": "mae",
            "random_state": SEED,
            "n_jobs": -1,
            "verbosity": 0,
        }
        if early_stopping:
            params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
        return XGBRegressor(**params)

    raise ValueError(spec.family)


def fit_model(
    model,
    spec,
    x_train,
    target_train,
    x_valid,
    target_valid,
    weight_target_train,
    outage_train,
    group_ids_train=None,
):
    sample_weight = make_training_weight(
        weight_target_train,
        outage_train,
        spec,
        group_ids=group_ids_train,
    )

    if spec.family == "lightgbm":
        callbacks = [
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ]
        model.fit(
            x_train,
            target_train,
            sample_weight=sample_weight,
            eval_set=[(x_valid, target_valid)],
            callbacks=callbacks,
        )
        best_iteration = int(model.best_iteration_ or model.n_estimators)

    elif spec.family == "catboost":
        model.fit(
            x_train,
            target_train,
            sample_weight=sample_weight,
            eval_set=(x_valid, target_valid),
            use_best_model=True,
            verbose=False,
        )
        best_iteration = int(model.get_best_iteration() + 1)
        if best_iteration <= 0:
            best_iteration = int(model.get_params()["iterations"])

    elif spec.family == "xgboost":
        model.fit(
            x_train,
            target_train,
            sample_weight=sample_weight,
            eval_set=[(x_valid, target_valid)],
            verbose=False,
        )
        try:
            best_iteration = int(model.best_iteration + 1)
        except (AttributeError, TypeError):
            best_iteration = int(model.get_params()["n_estimators"])
    else:
        raise ValueError(spec.family)

    return best_iteration


def fit_fixed_model(
    model,
    spec,
    x_train,
    target_train,
    weight_target_train,
    outage_train,
    group_ids_train=None,
):
    sample_weight = make_training_weight(
        weight_target_train,
        outage_train,
        spec,
        group_ids=group_ids_train,
    )
    if spec.family == "lightgbm":
        model.fit(
            x_train,
            target_train,
            sample_weight=sample_weight,
        )
    else:
        model.fit(
            x_train,
            target_train,
            sample_weight=sample_weight,
            verbose=False,
        )
    return model


def predict_cf(model, features):
    return np.clip(
        np.asarray(model.predict(features), dtype=float),
        0.0,
        PREDICTION_MAX_CF,
    )


# ---- 원본 노트북 코드 셀 14 ----
def fold_indices(n_rows, n_folds):
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    return list(splitter.split(np.arange(n_rows)))


def crossfit_group_component(
    spec,
    x_all,
    y_all,
    outage_all,
    x_external,
    fixed_iterations=None,
    keep_models=False,
    n_folds=ANCHOR_FOLDS,
):
    x_all = x_all.reset_index(drop=True)
    x_external = x_external.reindex(columns=x_all.columns).reset_index(drop=True)
    y_all = np.asarray(y_all, dtype=float)
    outage_all = np.asarray(outage_all, dtype=bool)
    active = y_all >= ACTIVE_CF

    oof = np.zeros(len(x_all), dtype=float)
    external_predictions = []
    iterations = []
    models = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        fold_indices(len(x_all), n_folds), start=1
    ):
        train_active_idx = train_idx[active[train_idx]]
        valid_active_idx = valid_idx[active[valid_idx]]
        if len(valid_active_idx) == 0:
            raise ValueError("Fold 안에 활성 검증행이 없습니다.")

        use_early_stopping = fixed_iterations is None
        n_estimators = (
            max_iterations_for_spec(spec)
            if use_early_stopping
            else int(fixed_iterations)
        )
        model = create_model(spec, n_estimators, use_early_stopping)

        if use_early_stopping:
            best_iteration = fit_model(
                model,
                spec,
                x_all.iloc[train_active_idx],
                y_all[train_active_idx],
                x_all.iloc[valid_active_idx],
                y_all[valid_active_idx],
                y_all[train_active_idx],
                outage_all[train_active_idx],
            )
        else:
            model = fit_fixed_model(
                model,
                spec,
                x_all.iloc[train_active_idx],
                y_all[train_active_idx],
                y_all[train_active_idx],
                outage_all[train_active_idx],
            )
            best_iteration = int(fixed_iterations)

        oof[valid_idx] = predict_cf(model, x_all.iloc[valid_idx])
        external_predictions.append(predict_cf(model, x_external))
        iterations.append(best_iteration)
        if keep_models:
            models.append(model)

        print(
            f"    {spec.name} fold {fold_number}/{n_folds} "
            f"완료 (iter={best_iteration})"
        )
        if not keep_models:
            del model
            gc.collect()

    return {
        "oof": oof,
        "external": np.mean(external_predictions, axis=0),
        "iterations": iterations,
        "median_iteration": int(np.median(iterations)),
        "models": models,
    }


def add_group_indicators(frame, group_id):
    result = frame.copy()
    for candidate_group in TARGETS:
        result[f"pooled_group_{candidate_group}"] = float(
            candidate_group == group_id
        )
    return result


def pooled_union_columns(period_data):
    columns = set()
    for data in period_data.values():
        columns.update(data["x_all"].columns)
    columns.update({f"pooled_group_{g}" for g in TARGETS})
    return sorted(columns)


def crossfit_pooled_component(
    spec,
    period_data,
    target_group,
    fixed_iterations=None,
    keep_models=False,
    n_folds=ANCHOR_FOLDS,
):
    union_columns = pooled_union_columns(period_data)
    folds_by_group = {
        g: fold_indices(len(data["x_all"]), n_folds)
        for g, data in period_data.items()
    }

    target_rows = len(period_data[target_group]["x_all"])
    oof = np.zeros(target_rows, dtype=float)
    external_predictions = []
    iterations = []
    models = []

    target_external = add_group_indicators(
        period_data[target_group]["x_external"], target_group
    ).reindex(columns=union_columns, fill_value=0.0)

    for fold_number in range(n_folds):
        x_train_parts = []
        y_train_parts = []
        outage_parts = []
        group_parts = []
        x_valid_parts = []
        y_valid_parts = []

        for group_id, data in period_data.items():
            train_idx, valid_idx = folds_by_group[group_id][fold_number]
            y_group = np.asarray(data["y_all"], dtype=float)
            active = y_group >= ACTIVE_CF
            train_active_idx = train_idx[active[train_idx]]
            valid_active_idx = valid_idx[active[valid_idx]]

            x_group = add_group_indicators(data["x_all"], group_id).reindex(
                columns=union_columns, fill_value=0.0
            )
            x_train_parts.append(x_group.iloc[train_active_idx])
            y_train_parts.append(y_group[train_active_idx])
            outage_parts.append(np.asarray(data["outage"])[train_active_idx])
            group_parts.append(
                np.full(len(train_active_idx), group_id, dtype=int)
            )
            x_valid_parts.append(x_group.iloc[valid_active_idx])
            y_valid_parts.append(y_group[valid_active_idx])

            if group_id == target_group:
                oof[valid_idx] = 0.0
                target_valid_all = x_group.iloc[valid_idx]
                target_valid_idx = valid_idx

        x_train = pd.concat(x_train_parts, ignore_index=True)
        y_train = np.concatenate(y_train_parts)
        outage_train = np.concatenate(outage_parts)
        group_train = np.concatenate(group_parts)
        x_valid = pd.concat(x_valid_parts, ignore_index=True)
        y_valid = np.concatenate(y_valid_parts)

        use_early_stopping = fixed_iterations is None
        n_estimators = (
            max_iterations_for_spec(spec)
            if use_early_stopping
            else int(fixed_iterations)
        )
        model = create_model(spec, n_estimators, use_early_stopping)

        if use_early_stopping:
            best_iteration = fit_model(
                model,
                spec,
                x_train,
                y_train,
                x_valid,
                y_valid,
                y_train,
                outage_train,
                group_ids_train=group_train,
            )
        else:
            model = fit_fixed_model(
                model,
                spec,
                x_train,
                y_train,
                y_train,
                outage_train,
                group_ids_train=group_train,
            )
            best_iteration = int(fixed_iterations)

        oof[target_valid_idx] = predict_cf(model, target_valid_all)
        external_predictions.append(predict_cf(model, target_external))
        iterations.append(best_iteration)
        if keep_models:
            models.append(model)

        print(
            f"    {spec.name} pooled fold {fold_number + 1}/{n_folds} "
            f"완료 (iter={best_iteration})"
        )
        if not keep_models:
            del model
            gc.collect()

    return {
        "oof": oof,
        "external": np.mean(external_predictions, axis=0),
        "iterations": iterations,
        "median_iteration": int(np.median(iterations)),
        "models": models,
    }


def make_period_data(history_end, external_kind):
    result = {}
    for group_id, data in group_data.items():
        times = data["times"]
        history_mask = (times < history_end).to_numpy()

        if external_kind == "validation":
            external_mask = (
                (times >= TRAIN_END) & (times < VALIDATION_END)
            ).to_numpy()
            x_external = data["x_train"].loc[external_mask].reset_index(drop=True)
            external_times = times.loc[external_mask].reset_index(drop=True)
        elif external_kind == "test":
            x_external = data["x_test"].reset_index(drop=True)
            external_times = data["test_times"].reset_index(drop=True)
        else:
            raise ValueError(external_kind)

        result[group_id] = {
            "x_all": data["x_train"].loc[history_mask].reset_index(drop=True),
            "y_all": data["y_cf"][history_mask],
            "outage": data["outage"][history_mask],
            "times": times.loc[history_mask].reset_index(drop=True),
            "x_external": x_external,
            "external_times": external_times,
        }
    return result


# ---- 원본 노트북 코드 셀 16 ----
validation_period = make_period_data(TRAIN_END, "validation")
component_results = {g: {} for g in TARGETS}
iteration_map = {}
component_fold_map = {}
candidate_metric_rows = []

for group_id in TARGETS:
    print("=" * 78)
    print(f"Group {group_id} 개별 후보")
    data = validation_period[group_id]

    for spec in GROUP_SPECS[group_id]:
        model_folds = (
            ANCHOR_FOLDS
            if spec.name in ANCHOR_RECIPES[group_id]
            else AUXILIARY_FOLDS
        )
        result = crossfit_group_component(
            spec,
            data["x_all"],
            data["y_all"],
            data["outage"],
            data["x_external"],
            n_folds=model_folds,
        )
        component_results[group_id][spec.name] = result
        iteration_map[(group_id, spec.name)] = result["median_iteration"]
        component_fold_map[(group_id, spec.name)] = model_folds

        y_val = group_data[group_id]["y_cf"][
            ((group_data[group_id]["times"] >= TRAIN_END)
             & (group_data[group_id]["times"] < VALIDATION_END)).to_numpy()
        ]
        metrics = official_metrics_cf(
            y_val, result["external"], CAPACITIES[group_id]
        )
        candidate_metric_rows.append(
            {
                "group": group_id,
                "candidate": spec.name,
                "folds": model_folds,
                "median_iteration": result["median_iteration"],
                **metrics,
            }
        )
        print(
            f"  {spec.name}: Score={metrics['Score']:.6f}, "
            f"NMAE={metrics['NMAE']:.6f}, FICR={metrics['FICR']:.6f}"
        )

print("=" * 78)
print("Pooled 후보는 세 그룹을 함께 학습하고 Group 3을 예측합니다.")
for spec in POOLED_SPECS:
    model_folds = (
        ANCHOR_FOLDS
        if spec.name in ANCHOR_RECIPES[3]
        else AUXILIARY_FOLDS
    )
    result = crossfit_pooled_component(
        spec,
        validation_period,
        target_group=3,
        n_folds=model_folds,
    )
    component_results[3][spec.name] = result
    iteration_map[(3, spec.name)] = result["median_iteration"]
    component_fold_map[(3, spec.name)] = model_folds

    y_val = group_data[3]["y_cf"][
        ((group_data[3]["times"] >= TRAIN_END)
         & (group_data[3]["times"] < VALIDATION_END)).to_numpy()
    ]
    metrics = official_metrics_cf(y_val, result["external"], CAPACITIES[3])
    candidate_metric_rows.append(
        {
            "group": 3,
            "candidate": spec.name,
            "folds": model_folds,
            "median_iteration": result["median_iteration"],
            **metrics,
        }
    )
    print(
        f"  {spec.name}: Score={metrics['Score']:.6f}, "
        f"NMAE={metrics['NMAE']:.6f}, FICR={metrics['FICR']:.6f}"
    )

candidate_metrics_df = pd.DataFrame(candidate_metric_rows)
display(candidate_metrics_df)


# ---- 원본 노트북 코드 셀 18 ----
def combine_components(group_id, recipe, source):
    result = None
    for name, weight in recipe.items():
        values = np.asarray(component_results[group_id][name][source], dtype=float)
        result = weight * values if result is None else result + weight * values
    return np.clip(result, 0.0, PREDICTION_MAX_CF)


recipe_predictions = {g: {"oof": {}, "external": {}} for g in TARGETS}
for group_id in TARGETS:
    for recipe_name, recipe in RECIPE_MAPS[group_id].items():
        recipe_predictions[group_id]["oof"][recipe_name] = combine_components(
            group_id, recipe, "oof"
        )
        recipe_predictions[group_id]["external"][recipe_name] = combine_components(
            group_id, recipe, "external"
        )


def blend_recipe_predictions(recipe_values, weights):
    result = np.zeros_like(next(iter(recipe_values.values())), dtype=float)
    for name, weight in weights.items():
        result += float(weight) * np.asarray(recipe_values[name], dtype=float)
    return np.clip(result, 0.0, PREDICTION_MAX_CF)


def stage1_weight_candidates(group_id):
    max_aux = 0.15 if group_id == 2 else 0.30
    step = 0.05
    candidates = []
    upload_values = np.arange(0.0, max_aux + 1e-9, step)

    if "clone" in RECIPE_MAPS[group_id]:
        clone_values = np.arange(0.0, max_aux + 1e-9, step)
        for clone_weight in clone_values:
            for uploaded_weight in upload_values:
                if clone_weight + uploaded_weight <= max_aux + 1e-9:
                    candidates.append(
                        {
                            "anchor": float(1.0 - clone_weight - uploaded_weight),
                            "clone": float(clone_weight),
                            "uploaded": float(uploaded_weight),
                        }
                    )
    else:
        for uploaded_weight in upload_values:
            candidates.append(
                {
                    "anchor": float(1.0 - uploaded_weight),
                    "uploaded": float(uploaded_weight),
                }
            )
    return candidates


stage1_selection = {}
stage1_oof = {}
stage1_validation = {}
anchor_validation = {}

for group_id in TARGETS:
    data = validation_period[group_id]
    y_val = group_data[group_id]["y_cf"][
        ((group_data[group_id]["times"] >= TRAIN_END)
         & (group_data[group_id]["times"] < VALIDATION_END)).to_numpy()
    ]
    val_times = data["external_times"]
    h1 = (val_times < H1_END).to_numpy()
    h2 = ~h1

    external_recipes = recipe_predictions[group_id]["external"]
    anchor_pred = external_recipes["anchor"]
    anchor_validation[group_id] = anchor_pred

    rows = []
    for weights in stage1_weight_candidates(group_id):
        pred = blend_recipe_predictions(external_recipes, weights)
        h1_metrics = metrics_on_mask(y_val, pred, CAPACITIES[group_id], h1)
        rows.append(
            {
                "weights": weights,
                "prediction": pred,
                "H1": h1_metrics,
            }
        )

    best_h1 = max(row["H1"]["Score"] for row in rows)
    near_best = [
        row for row in rows
        if row["H1"]["Score"] >= best_h1 - 0.0003
    ]
    # 점수가 거의 같다면 Anchor 비중이 큰 단순한 혼합을 우선합니다.
    chosen = sorted(
        near_best,
        key=lambda row: (
            row["weights"]["anchor"],
            -row["H1"]["NMAE"],
            -row["weights"]["uploaded"],
        ),
        reverse=True,
    )[0]

    is_anchor = chosen["weights"]["anchor"] >= 1.0 - 1e-12
    if is_anchor:
        report = {"passed": True, "reason": "H1에서도 Anchor가 선택됨"}
        final_weights = chosen["weights"]
    else:
        report = approval_report(
            group_id,
            y_val,
            val_times,
            anchor_pred,
            chosen["prediction"],
            h1,
            h2,
        )
        final_weights = (
            chosen["weights"]
            if report["passed"]
            else {
                name: (1.0 if name == "anchor" else 0.0)
                for name in RECIPE_MAPS[group_id]
            }
        )

    final_external = blend_recipe_predictions(external_recipes, final_weights)
    final_oof = blend_recipe_predictions(
        recipe_predictions[group_id]["oof"], final_weights
    )

    stage1_selection[group_id] = {
        "chosen_H1_weights": chosen["weights"],
        "final_weights": final_weights,
        "approval": report,
    }
    stage1_validation[group_id] = final_external
    stage1_oof[group_id] = final_oof

    print("=" * 72)
    print(f"Group {group_id} Stage 1")
    print("  H1 선택:", chosen["weights"])
    print("  최종 적용:", final_weights)
    print("  결과:", "채택" if report["passed"] else "H2 실패 → Anchor")
    print(
        "  전체 Score:",
        f"{official_metrics_cf(y_val, anchor_pred, CAPACITIES[group_id])['Score']:.6f}",
        "->",
        f"{official_metrics_cf(y_val, final_external, CAPACITIES[group_id])['Score']:.6f}",
    )


# ---- 원본 노트북 코드 셀 20 ----
def align_prediction_by_time(source_times, source_values, target_times):
    series = pd.Series(
        np.asarray(source_values, dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(source_times)),
    )
    series = series.groupby(level=0).mean().sort_index()
    aligned = series.reindex(pd.DatetimeIndex(pd.to_datetime(target_times)))
    aligned = aligned.interpolate(method="time").ffill().bfill()
    if aligned.isna().any():
        raise ValueError("시간 정렬 후에도 참조 예측에 결측이 남았습니다.")
    return aligned.to_numpy(dtype=float)


def apply_group1_calibration(
    group1_pred,
    group2_pred,
    center1,
    center2,
    scale,
    reference_strength,
    offset,
):
    calibrated = (
        center1
        + scale * (np.asarray(group1_pred) - center1)
        + reference_strength * (np.asarray(group2_pred) - center2)
        + offset
    )
    return np.clip(calibrated, 0.0, PREDICTION_MAX_CF)


def select_group1_calibration():
    group_id = 1
    data1 = validation_period[1]
    y_val = group_data[1]["y_cf"][
        ((group_data[1]["times"] >= TRAIN_END)
         & (group_data[1]["times"] < VALIDATION_END)).to_numpy()
    ]
    times1 = data1["external_times"]
    h1 = (times1 < H1_END).to_numpy()
    h2 = ~h1

    base1 = stage1_validation[1]
    ref2 = align_prediction_by_time(
        validation_period[2]["external_times"],
        stage1_validation[2],
        times1,
    )
    center1 = float(np.mean(base1[h1]))
    center2 = float(np.mean(ref2[h1]))

    rows = []
    for scale in np.arange(1.00, 1.201, 0.025):
        for reference_strength in np.arange(0.00, 0.151, 0.05):
            for offset in np.arange(-0.010, 0.0101, 0.005):
                pred = apply_group1_calibration(
                    base1,
                    ref2,
                    center1,
                    center2,
                    float(scale),
                    float(reference_strength),
                    float(offset),
                )
                h1_metrics = metrics_on_mask(
                    y_val, pred, CAPACITIES[1], h1
                )
                rows.append(
                    {
                        "scale": float(scale),
                        "reference_strength": float(reference_strength),
                        "offset": float(offset),
                        "prediction": pred,
                        "H1": h1_metrics,
                    }
                )

    best_h1 = max(row["H1"]["Score"] for row in rows)
    near_best = [
        row for row in rows
        if row["H1"]["Score"] >= best_h1 - 0.0003
    ]
    chosen = sorted(
        near_best,
        key=lambda row: (
            -abs(row["scale"] - 1.0),
            -row["reference_strength"],
            -abs(row["offset"]),
        ),
        reverse=True,
    )[0]

    is_identity = (
        abs(chosen["scale"] - 1.0) < 1e-12
        and abs(chosen["reference_strength"]) < 1e-12
        and abs(chosen["offset"]) < 1e-12
    )
    if is_identity:
        report = {"passed": True, "reason": "보정 없음이 H1에서 선택됨"}
        accepted = False
    else:
        report = approval_report(
            1, y_val, times1, base1, chosen["prediction"], h1, h2
        )
        accepted = bool(report["passed"])

    if not accepted:
        return {
            "accepted": False,
            "scale": 1.0,
            "reference_strength": 0.0,
            "offset": 0.0,
            "prediction": base1.copy(),
            "approval": report,
        }

    return {
        "accepted": True,
        "scale": chosen["scale"],
        "reference_strength": chosen["reference_strength"],
        "offset": chosen["offset"],
        "prediction": chosen["prediction"],
        "approval": report,
    }


group1_calibration = select_group1_calibration()
stage1_validation[1] = group1_calibration["prediction"]

if group1_calibration["accepted"]:
    ref2_oof = align_prediction_by_time(
        validation_period[2]["times"],
        stage1_oof[2],
        validation_period[1]["times"],
    )
    center1_oof = float(np.mean(stage1_oof[1]))
    center2_oof = float(np.mean(ref2_oof))
    stage1_oof[1] = apply_group1_calibration(
        stage1_oof[1],
        ref2_oof,
        center1_oof,
        center2_oof,
        group1_calibration["scale"],
        group1_calibration["reference_strength"],
        group1_calibration["offset"],
    )

print("Group 1 분산 보정:", "채택" if group1_calibration["accepted"] else "미적용")
print(
    {
        key: group1_calibration[key]
        for key in ["scale", "reference_strength", "offset"]
    }
)


# ---- 원본 노트북 코드 셀 22 ----
RESIDUAL_SPECS = [
    ModelSpec(
        "residual_lgb",
        "lightgbm",
        "mae",
        outage_weight=1.0,
        weight_scheme="residual_high",
        variant="residual",
    ),
    ModelSpec(
        "residual_cat",
        "catboost",
        "mae",
        outage_weight=1.0,
        weight_scheme="residual_high",
        variant="residual",
    ),
]


def residual_weather_columns(frame, max_columns=40):
    priority_exact = [
        "wind_all_mean",
        "wind_all_std",
        "wind_ldaps_mean",
        "wind_ldaps_std",
        "wind_gfs_mean",
        "wind_gfs_std",
        "wind_source_difference",
        "wind_source_ratio",
        "time_hour_sin",
        "time_hour_cos",
        "time_doy_sin",
        "time_doy_cos",
        "time_month",
    ]
    selected = [c for c in priority_exact if c in frame.columns]

    extra = [
        c for c in frame.columns
        if (
            "wind_speed" in c.lower()
            or "wind_all_mean_diff" in c.lower()
            or "wind_ldaps_mean_diff" in c.lower()
            or "wind_gfs_mean_diff" in c.lower()
        )
        and c not in selected
    ]
    selected.extend(sorted(extra)[: max(0, max_columns - len(selected))])
    return selected[:max_columns]


def make_residual_meta(
    raw_features,
    base_prediction,
    recipe_prediction_dict,
    weather_columns=None,
):
    if weather_columns is None:
        weather_columns = residual_weather_columns(raw_features)

    meta = pd.DataFrame(
        {
            "stage1_base": np.asarray(base_prediction, dtype=float),
            "recipe_anchor": np.asarray(
                recipe_prediction_dict["anchor"], dtype=float
            ),
            "recipe_clone": np.asarray(
                recipe_prediction_dict["clone"], dtype=float
            ),
            "recipe_uploaded": np.asarray(
                recipe_prediction_dict["uploaded"], dtype=float
            ),
        }
    )
    recipe_matrix = meta[
        ["recipe_anchor", "recipe_clone", "recipe_uploaded"]
    ].to_numpy()
    meta["recipe_mean"] = recipe_matrix.mean(axis=1)
    meta["recipe_std"] = recipe_matrix.std(axis=1)
    meta["recipe_range"] = recipe_matrix.max(axis=1) - recipe_matrix.min(axis=1)
    meta["anchor_minus_clone"] = meta["recipe_anchor"] - meta["recipe_clone"]
    meta["uploaded_minus_anchor"] = (
        meta["recipe_uploaded"] - meta["recipe_anchor"]
    )

    for column in weather_columns:
        meta[f"weather__{column}"] = raw_features[column].to_numpy(dtype=float)

    return (
        meta.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32"),
        weather_columns,
    )


def crossfit_residual_component(
    spec,
    x_all,
    residual_target,
    actual_y,
    x_external,
    fixed_iterations=None,
    keep_models=False,
    n_folds=RESIDUAL_FOLDS,
):
    x_all = x_all.reset_index(drop=True)
    x_external = x_external.reindex(columns=x_all.columns).reset_index(drop=True)
    residual_target = np.asarray(residual_target, dtype=float)
    actual_y = np.asarray(actual_y, dtype=float)
    active_indices = np.flatnonzero(actual_y >= ACTIVE_CF)

    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros(len(x_all), dtype=float)
    external_predictions = []
    iterations = []
    models = []

    for fold_number, (train_local, valid_local) in enumerate(
        splitter.split(active_indices), start=1
    ):
        train_idx = active_indices[train_local]
        valid_idx = active_indices[valid_local]

        use_early_stopping = fixed_iterations is None
        n_estimators = (
            max_iterations_for_spec(spec)
            if use_early_stopping
            else int(fixed_iterations)
        )
        model = create_model(spec, n_estimators, use_early_stopping)

        if use_early_stopping:
            best_iteration = fit_model(
                model,
                spec,
                x_all.iloc[train_idx],
                residual_target[train_idx],
                x_all.iloc[valid_idx],
                residual_target[valid_idx],
                actual_y[train_idx],
                np.zeros(len(train_idx), dtype=bool),
            )
        else:
            model = fit_fixed_model(
                model,
                spec,
                x_all.iloc[train_idx],
                residual_target[train_idx],
                actual_y[train_idx],
                np.zeros(len(train_idx), dtype=bool),
            )
            best_iteration = int(fixed_iterations)

        oof[valid_idx] = np.asarray(
            model.predict(x_all.iloc[valid_idx]), dtype=float
        )
        external_predictions.append(
            np.asarray(model.predict(x_external), dtype=float)
        )
        iterations.append(best_iteration)
        if keep_models:
            models.append(model)

        print(
            f"    {spec.name} fold {fold_number}/{n_folds} "
            f"완료 (iter={best_iteration})"
        )
        if not keep_models:
            del model
            gc.collect()

    return {
        "oof": oof,
        "external": np.mean(external_predictions, axis=0),
        "iterations": iterations,
        "median_iteration": int(np.median(iterations)),
        "models": models,
    }


def apply_residual_correction(base, lgb_residual, cat_residual, lgb_weight, beta, cap):
    lgb_residual = np.asarray(lgb_residual, dtype=float)
    cat_residual = np.asarray(cat_residual, dtype=float)
    agree = np.sign(lgb_residual) == np.sign(cat_residual)
    mixed = lgb_weight * lgb_residual + (1.0 - lgb_weight) * cat_residual
    mixed = np.where(agree, mixed, 0.0)
    correction = beta * np.clip(mixed, -cap, cap)
    return np.clip(np.asarray(base) + correction, 0.0, PREDICTION_MAX_CF)

residual_results = {}
correction_selection = {
    group_id: {
        "type": "none",
        "params": {},
        "approval": {"passed": True, "reason": "런타임 최적화: 잔차 미사용"},
        "weather_columns": [],
    }
    for group_id in TARGETS
}
final_validation = {g: stage1_validation[g].copy() for g in TARGETS}

# 과거 실험에서 의미 있는 잔차 개선이 확인된 Group 1만 2-stage를 학습합니다.
for group_id in [1]:
    print("=" * 78)
    print(f"Group {group_id} 잔차 모델")

    period = validation_period[group_id]
    history_recipe = recipe_predictions[group_id]["oof"]
    external_recipe = recipe_predictions[group_id]["external"]

    x_meta_history, weather_columns = make_residual_meta(
        period["x_all"],
        stage1_oof[group_id],
        history_recipe,
    )
    x_meta_external, _ = make_residual_meta(
        period["x_external"],
        stage1_validation[group_id],
        external_recipe,
        weather_columns,
    )

    residual_target = period["y_all"] - stage1_oof[group_id]
    residual_results[group_id] = {}

    for spec in RESIDUAL_SPECS:
        result = crossfit_residual_component(
            spec,
            x_meta_history,
            residual_target,
            period["y_all"],
            x_meta_external,
        )
        residual_results[group_id][spec.name] = result
        iteration_map[(group_id, spec.name)] = result["median_iteration"]

    y_val = group_data[group_id]["y_cf"][
        ((group_data[group_id]["times"] >= TRAIN_END)
         & (group_data[group_id]["times"] < VALIDATION_END)).to_numpy()
    ]
    val_times = period["external_times"]
    h1 = (val_times < H1_END).to_numpy()
    h2 = ~h1
    base = stage1_validation[group_id]

    correction_candidates = [
        {
            "type": "none",
            "params": {},
            "prediction": base.copy(),
        }
    ]

    beta_values = [0.1, 0.2, 0.3]
    for lgb_weight in [0.0, 0.25, 0.50, 0.75, 1.0]:
        for beta in beta_values:
            for cap in [0.02, 0.04]:
                pred = apply_residual_correction(
                    base,
                    residual_results[group_id]["residual_lgb"]["external"],
                    residual_results[group_id]["residual_cat"]["external"],
                    lgb_weight,
                    beta,
                    cap,
                )
                correction_candidates.append(
                    {
                        "type": "residual",
                        "params": {
                            "lgb_weight": lgb_weight,
                            "beta": beta,
                            "cap": cap,
                        },
                        "prediction": pred,
                    }
                )
    for row in correction_candidates:
        row["H1"] = metrics_on_mask(
            y_val, row["prediction"], CAPACITIES[group_id], h1
        )

    best_h1 = max(row["H1"]["Score"] for row in correction_candidates)
    near_best = [
        row for row in correction_candidates
        if row["H1"]["Score"] >= best_h1 - 0.0003
    ]
    # 거의 같은 점수라면 보정 없음, 작은 beta, 작은 cap 순서로 보수적으로 선택합니다.
    type_priority = {"none": 2, "residual": 1}
    chosen = sorted(
        near_best,
        key=lambda row: (
            type_priority[row["type"]],
            -row["params"].get("beta", 0.0),
            -row["params"].get("cap", 0.0),
        ),
        reverse=True,
    )[0]

    if chosen["type"] == "none":
        report = {"passed": True, "reason": "H1에서 보정 없음 선택"}
        accepted = True
    else:
        report = approval_report(
            group_id,
            y_val,
            val_times,
            base,
            chosen["prediction"],
            h1,
            h2,
        )
        accepted = bool(report["passed"])

    if not accepted:
        chosen = {
            "type": "none",
            "params": {},
            "prediction": base.copy(),
        }

    correction_selection[group_id] = {
        "type": chosen["type"],
        "params": chosen["params"],
        "approval": report,
        "weather_columns": weather_columns,
    }
    final_validation[group_id] = chosen["prediction"]

    print("  최종 보정:", chosen["type"], chosen["params"])
    if "H2_score_gain" in report:
        print(
            f"  H1 gain={report['H1_score_gain']:+.6f}, "
            f"H2 gain={report['H2_score_gain']:+.6f}, "
            f"월 승리={report['total_month_wins']}/12"
        )


# ---- 원본 노트북 코드 셀 24 ----
summary_rows = []
segment_rows = []
validation_prediction_parts = []

for group_id in TARGETS:
    period = validation_period[group_id]
    val_mask_original = (
        (group_data[group_id]["times"] >= TRAIN_END)
        & (group_data[group_id]["times"] < VALIDATION_END)
    ).to_numpy()
    y_val = group_data[group_id]["y_cf"][val_mask_original]
    times = period["external_times"].reset_index(drop=True)

    model_predictions = {
        "anchor": anchor_validation[group_id],
        "stage1": stage1_validation[group_id],
        "final": final_validation[group_id],
    }
    h1 = (times < H1_END).to_numpy()
    h2 = ~h1

    for model_name, pred in model_predictions.items():
        for split_name, split_mask in [
            ("H1", h1),
            ("H2", h2),
            ("FULL_2024", np.ones(len(times), dtype=bool)),
        ]:
            metrics = metrics_on_mask(
                y_val, pred, CAPACITIES[group_id], split_mask
            )
            summary_rows.append(
                {
                    "group": group_id,
                    "model": model_name,
                    "split": split_name,
                    **metrics,
                }
            )

    segments = {
        "10_30pct": (y_val >= 0.10) & (y_val < 0.30),
        "30_60pct": (y_val >= 0.30) & (y_val < 0.60),
        "60_80pct": (y_val >= 0.60) & (y_val < 0.80),
        "80_100pct": (y_val >= 0.80) & (y_val <= 1.05),
    }
    for segment_name, segment_mask in segments.items():
        if segment_mask.sum() == 0:
            continue
        for model_name, pred in model_predictions.items():
            metrics = metrics_on_mask(
                y_val, pred, CAPACITIES[group_id], segment_mask
            )
            segment_rows.append(
                {
                    "group": group_id,
                    "segment": segment_name,
                    "model": model_name,
                    **metrics,
                }
            )

    anchor_error = np.abs(anchor_validation[group_id] - y_val)
    final_error = np.abs(final_validation[group_id] - y_val)
    validation_prediction_parts.append(
        pd.DataFrame(
            {
                "group": group_id,
                TIME_COL: times,
                "actual_cf": y_val,
                "anchor_cf": anchor_validation[group_id],
                "stage1_cf": stage1_validation[group_id],
                "final_cf": final_validation[group_id],
                "anchor_abs_error_cf": anchor_error,
                "final_abs_error_cf": final_error,
                "anchor_within_8": anchor_error <= 0.08,
                "final_within_8": final_error <= 0.08,
                "rescued_to_within_8": (anchor_error > 0.08)
                    & (final_error <= 0.08),
                "lost_from_within_8": (anchor_error <= 0.08)
                    & (final_error > 0.08),
            }
        )
    )

validation_summary_df = pd.DataFrame(summary_rows)
validation_segments_df = pd.DataFrame(segment_rows)
validation_predictions_df = pd.concat(
    validation_prediction_parts, ignore_index=True
)

display(
    validation_summary_df[
        validation_summary_df["split"] == "FULL_2024"
    ].reset_index(drop=True)
)
display(validation_segments_df)

macro_rows = []
for model_name in ["anchor", "stage1", "final"]:
    part = validation_summary_df[
        (validation_summary_df["model"] == model_name)
        & (validation_summary_df["split"] == "FULL_2024")
    ]
    macro_nmae = float(part["NMAE"].mean())
    macro_ficr = float(part["FICR"].mean())
    macro_rows.append(
        {
            "model": model_name,
            "macro_NMAE": macro_nmae,
            "macro_1_minus_NMAE": 1.0 - macro_nmae,
            "macro_FICR": macro_ficr,
            "macro_Score": 0.5 * (1.0 - macro_nmae) + 0.5 * macro_ficr,
        }
    )

macro_validation_df = pd.DataFrame(macro_rows)
display(macro_validation_df)


# ---- 원본 노트북 코드 셀 26 ----
SPEC_LOOKUP = {
    (group_id, spec.name): spec
    for group_id, specs in GROUP_SPECS.items()
    for spec in specs
}
SPEC_LOOKUP.update({(3, spec.name): spec for spec in POOLED_SPECS})


def recipe_component_names(group_id, recipe_name):
    return set(RECIPE_MAPS[group_id][recipe_name])


def required_final_components(group_id):
    required = set()
    final_weights = stage1_selection[group_id]["final_weights"]
    for recipe_name, weight in final_weights.items():
        if weight > 1e-12:
            required.update(recipe_component_names(group_id, recipe_name))

    correction_type = correction_selection[group_id]["type"]
    if correction_type == "residual":
        # Group 1 잔차 meta가 사용한 모든 레시피 예측이 필요합니다.
        for recipe_name in RECIPE_MAPS[group_id]:
            required.update(recipe_component_names(group_id, recipe_name))

    return required


final_period = make_period_data(VALIDATION_END, "test")
final_component_results = {g: {} for g in TARGETS}
saved_models = {g: {} for g in TARGETS}

for group_id in TARGETS:
    required_names = required_final_components(group_id)
    print("=" * 78)
    print(f"Group {group_id} 최종 필요 후보:", sorted(required_names))

    for name in sorted(required_names):
        spec = SPEC_LOOKUP[(group_id, name)]
        fixed_iterations = iteration_map[(group_id, name)]
        model_folds = component_fold_map[(group_id, name)]

        if spec.pooled:
            result = crossfit_pooled_component(
                spec,
                final_period,
                target_group=group_id,
                fixed_iterations=fixed_iterations,
                keep_models=SAVE_MODELS,
                n_folds=model_folds,
            )
        else:
            data = final_period[group_id]
            result = crossfit_group_component(
                spec,
                data["x_all"],
                data["y_all"],
                data["outage"],
                data["x_external"],
                fixed_iterations=fixed_iterations,
                keep_models=SAVE_MODELS,
                n_folds=model_folds,
            )

        final_component_results[group_id][name] = result
        if SAVE_MODELS:
            saved_models[group_id][name] = result["models"]


def combine_available_components(group_id, recipe, source):
    result = None
    for name, weight in recipe.items():
        if name not in final_component_results[group_id]:
            if abs(weight) > 1e-12:
                raise KeyError(f"최종 모델에 {name}이 없습니다.")
            continue
        values = np.asarray(
            final_component_results[group_id][name][source], dtype=float
        )
        result = weight * values if result is None else result + weight * values
    if result is None:
        raise ValueError("사용 가능한 최종 구성요소가 없습니다.")
    return np.clip(result, 0.0, PREDICTION_MAX_CF)


final_recipe_predictions = {g: {"oof": {}, "external": {}} for g in TARGETS}
final_stage1_oof = {}
final_stage1_test = {}

for group_id in TARGETS:
    selection_weights = stage1_selection[group_id]["final_weights"]
    for recipe_name, recipe in RECIPE_MAPS[group_id].items():
        if (
            selection_weights.get(recipe_name, 0.0) > 1e-12
            or correction_selection[group_id]["type"] == "residual"
        ):
            final_recipe_predictions[group_id]["oof"][recipe_name] = (
                combine_available_components(group_id, recipe, "oof")
            )
            final_recipe_predictions[group_id]["external"][recipe_name] = (
                combine_available_components(group_id, recipe, "external")
            )

    final_stage1_oof[group_id] = blend_recipe_predictions(
        final_recipe_predictions[group_id]["oof"],
        {
            name: weight
            for name, weight in selection_weights.items()
            if weight > 1e-12
        },
    )
    final_stage1_test[group_id] = blend_recipe_predictions(
        final_recipe_predictions[group_id]["external"],
        {
            name: weight
            for name, weight in selection_weights.items()
            if weight > 1e-12
        },
    )


# ---- 원본 노트북 코드 셀 28 ----
# Group 1 보정이 승인된 경우 전체 OOF와 Test에도 같은 파라미터를 적용합니다.
if group1_calibration["accepted"]:
    ref2_oof = align_prediction_by_time(
        final_period[2]["times"],
        final_stage1_oof[2],
        final_period[1]["times"],
    )
    ref2_test = align_prediction_by_time(
        final_period[2]["external_times"],
        final_stage1_test[2],
        final_period[1]["external_times"],
    )

    final_stage1_oof[1] = apply_group1_calibration(
        final_stage1_oof[1],
        ref2_oof,
        float(np.mean(final_stage1_oof[1])),
        float(np.mean(ref2_oof)),
        group1_calibration["scale"],
        group1_calibration["reference_strength"],
        group1_calibration["offset"],
    )
    # Test 자체의 평균을 중심으로 삼아 분산확대가 평균을 불필요하게 이동시키지 않게 합니다.
    final_stage1_test[1] = apply_group1_calibration(
        final_stage1_test[1],
        ref2_test,
        float(np.mean(final_stage1_test[1])),
        float(np.mean(ref2_test)),
        group1_calibration["scale"],
        group1_calibration["reference_strength"],
        group1_calibration["offset"],
    )


final_test_cf = {g: final_stage1_test[g].copy() for g in TARGETS}

for group_id in TARGETS:
    correction = correction_selection[group_id]
    correction_type = correction["type"]

    if correction_type == "none":
        print(f"Group {group_id}: 최종 보정 없음")
        continue

    if correction_type == "residual":
        weather_columns = correction["weather_columns"]
        period = final_period[group_id]

        x_meta_oof, _ = make_residual_meta(
            period["x_all"],
            final_stage1_oof[group_id],
            final_recipe_predictions[group_id]["oof"],
            weather_columns,
        )
        x_meta_test, _ = make_residual_meta(
            period["x_external"],
            final_stage1_test[group_id],
            final_recipe_predictions[group_id]["external"],
            weather_columns,
        )
        residual_target = period["y_all"] - final_stage1_oof[group_id]

        full_residual_predictions = {}
        for spec in RESIDUAL_SPECS:
            fixed_iterations = iteration_map[(group_id, spec.name)]
            result = crossfit_residual_component(
                spec,
                x_meta_oof,
                residual_target,
                period["y_all"],
                x_meta_test,
                fixed_iterations=fixed_iterations,
                keep_models=SAVE_MODELS,
                n_folds=RESIDUAL_FOLDS,
            )
            full_residual_predictions[spec.name] = result["external"]
            if SAVE_MODELS:
                saved_models[group_id][spec.name] = result["models"]

        params = correction["params"]
        final_test_cf[group_id] = apply_residual_correction(
            final_stage1_test[group_id],
            full_residual_predictions["residual_lgb"],
            full_residual_predictions["residual_cat"],
            params["lgb_weight"],
            params["beta"],
            params["cap"],
        )
        print(f"Group {group_id}: 검증 승인 2-stage 적용", params)


sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
submission = sample.copy()

for group_id, target in TARGETS.items():
    prediction_kwh = np.clip(
        final_test_cf[group_id] * CAPACITIES[group_id],
        0.0,
        PREDICTION_MAX_CF * CAPACITIES[group_id],
    )

    if TIME_COL in submission.columns:
        pred_series = pd.Series(
            prediction_kwh,
            index=pd.DatetimeIndex(final_period[group_id]["external_times"]),
        )
        sample_times = pd.to_datetime(submission[TIME_COL], errors="raise")
        aligned = pred_series.reindex(pd.DatetimeIndex(sample_times))
        if aligned.isna().any():
            raise ValueError(f"Group {group_id}: 제출 시간 정렬에 실패했습니다.")
        submission[target] = aligned.to_numpy()
    else:
        if len(submission) != len(prediction_kwh):
            raise ValueError(
                f"Group {group_id}: sample {len(submission)}행, "
                f"예측 {len(prediction_kwh)}행"
            )
        submission[target] = prediction_kwh

submission_path = OUTPUT_DIR / "submission.csv"
submission.to_csv(submission_path, index=False, encoding="utf-8-sig")
display(submission.head())
print("제출 파일:", submission_path)


# ---- 원본 노트북 코드 셀 30 ----
def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    return value


validation_summary_df.to_csv(
    OUTPUT_DIR / "validation_summary.csv",
    index=False,
    encoding="utf-8-sig",
)
validation_segments_df.to_csv(
    OUTPUT_DIR / "validation_segments.csv",
    index=False,
    encoding="utf-8-sig",
)
validation_predictions_df.to_csv(
    OUTPUT_DIR / "validation_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

selection_payload = {
    "seed": SEED,
    "folds": {
        "anchor": ANCHOR_FOLDS,
        "auxiliary": AUXILIARY_FOLDS,
        "group1_residual": RESIDUAL_FOLDS,
    },
    "data": {
        "train_end": TRAIN_END,
        "validation_end": VALIDATION_END,
        "H1_end": H1_END,
        "outage_rows_deleted": 0,
        "outage_flag_used_as_feature": False,
    },
    "stage1": stage1_selection,
    "group1_calibration": {
        key: value
        for key, value in group1_calibration.items()
        if key != "prediction"
    },
    "final_correction": correction_selection,
    "component_folds": {
        f"group{group_id}__{name}": folds
        for (group_id, name), folds in component_fold_map.items()
    },
    "iterations": {
        f"group{group_id}__{name}": iteration
        for (group_id, name), iteration in iteration_map.items()
    },
    "macro_validation": macro_validation_df,
}

with (OUTPUT_DIR / "selection.json").open("w", encoding="utf-8") as file:
    json.dump(
        json_safe(selection_payload),
        file,
        ensure_ascii=False,
        indent=2,
    )

if SAVE_MODELS:
    joblib.dump(
        {
            "models": saved_models,
            "selection": json_safe(selection_payload),
        },
        OUTPUT_DIR / "models.joblib",
        compress=3,
    )

elapsed_minutes = (time.perf_counter() - started_all) / 60.0
print(f"전체 실행시간: {elapsed_minutes:.1f}분")
print("최종 결과 폴더:", OUTPUT_DIR)
print(macro_validation_df.to_string(index=False))



# =============================================================================
# 6. Group 1·3 최적 구조 선택
# =============================================================================

# ---- 원본 노트북 코드 셀 1 ----
from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError('lightgbm이 필요합니다. 현재 Jupyter 환경에 lightgbm을 설치해 주세요.') from exc

warnings.filterwarnings('ignore', category=FutureWarning)

# ------------------------------------------------------------------
# 1. 사용자가 자주 바꿀 설정
# ------------------------------------------------------------------
ROOT = REPO_ROOT
DATA_PATH = ROOT / 'wind_baram_v2_preprocessed' / 'train_group3_preprocessed.csv'
PREVIOUS_BEST_PATH = ROOT / 'wind_baram_v2_output' / 'validation_predictions.csv'
CURRENT_MODEL_PATH = ROOT / 'wind_hybrid_2stage_output' / 'validation_predictions.csv'
OUTPUT_DIR = ROOT / 'group3_year_high_simple_output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CAPACITY_KWH = 21000.0
ACTIVE_CF = 0.10
PRED_MAX_CF = 1.05

# True이면 구조 확인용으로 빠르게 실행합니다. False이면 트리를 더 사용합니다.
FAST_MODE = False
BASE_ESTIMATORS = 700 if FAST_MODE else 1400
GATE_ESTIMATORS = 500 if FAST_MODE else 1000
HIGH_ESTIMATORS = 600 if FAST_MODE else 1200

# 최근 데이터 가중치와 예측 혼합비율 후보
RECENCY_WEIGHT_GRID = [1.0, 1.5, 2.0]
RECENT_BLEND_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]
BASELINE_SHARE_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]

# 고출력 전문 모델은 CF 70% 이상만 학습합니다.
HIGH_TRAIN_CF = 0.70
HIGH_LABEL_CF = 0.80
HIGH_ALPHA_GRID = [0.65, 0.75]
GAMMA_GRID = [0.0, 0.25, 0.50, 0.75]
HIGH_LIFT_CAP = 0.25

# 2024 전체로 학습해 2023을 예측하는 역방향 검사는 최종 선택에는 사용하지 않습니다.
RUN_REVERSE_CHECK = True

np.random.seed(SEED)
print('출력 폴더:', OUTPUT_DIR)
print('빠른 실행 모드:', FAST_MODE)


# ---- 원본 노트북 코드 셀 3 ----
TIME_COL = 'forecast_kst_dtm'
TARGET_COL = 'kpx_group_3'
OUTAGE_COL = 'is_long_outage_hour'

def official_metrics(y_true_cf, y_pred_cf):
    y_true_cf = np.asarray(y_true_cf, dtype=float)
    y_pred_cf = np.asarray(y_pred_cf, dtype=float)
    use = np.isfinite(y_true_cf) & np.isfinite(y_pred_cf) & (y_true_cf >= ACTIVE_CF)
    if not use.any():
        raise ValueError('공식 평가 대상 행이 없습니다.')
    actual = y_true_cf[use]
    pred = y_pred_cf[use]
    error = np.abs(pred - actual)
    rate = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
    actual_kwh = actual * CAPACITY_KWH
    nmae = float(error.mean())
    ficr = float((rate * actual_kwh).sum() / (4.0 * actual_kwh).sum())
    return {
        'rows': int(use.sum()),
        'MAE_kWh': float((error * CAPACITY_KWH).mean()),
        'NMAE': nmae,
        'one_minus_NMAE': 1.0 - nmae,
        'FICR': ficr,
        'Score': 0.5 * (1.0 - nmae) + 0.5 * ficr,
        'within_6_rate': float((error <= 0.06).mean()),
        'within_8_rate': float((error <= 0.08).mean()),
        'signed_bias_cf': float((pred - actual).mean()),
    }

def segment_metrics(y, pred, model_name, split_name):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    definitions = [
        ('10_30pct', 0.10, 0.30),
        ('30_60pct', 0.30, 0.60),
        ('60_80pct', 0.60, 0.80),
        ('80pct_plus', 0.80, np.inf),
    ]
    rows = []
    for name, low, high in definitions:
        mask = (y >= low) & (y < high)
        if mask.any():
            row = official_metrics(y[mask], pred[mask])
            row.update(model=model_name, split=split_name, segment=name)
            rows.append(row)
    return rows

if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)
if not PREVIOUS_BEST_PATH.exists():
    raise FileNotFoundError(PREVIOUS_BEST_PATH)
if not CURRENT_MODEL_PATH.exists():
    raise FileNotFoundError(
        f'현재 통합모델 검증예측이 없습니다: {CURRENT_MODEL_PATH}\n'
        '풍력_최종통합_2stage_안전앙상블.ipynb를 끝까지 실행한 뒤 다시 실행해 주세요.'
    )

data = pd.read_csv(DATA_PATH)
data[TIME_COL] = pd.to_datetime(data[TIME_COL])
data['target_cf'] = pd.to_numeric(data[TARGET_COL], errors='coerce') / CAPACITY_KWH

previous = pd.read_csv(PREVIOUS_BEST_PATH)
previous = previous.loc[pd.to_numeric(previous['group'], errors='coerce') == 3].copy()
previous[TIME_COL] = pd.to_datetime(previous[TIME_COL])
previous['previous_best_cf'] = pd.to_numeric(previous['prediction_kwh'], errors='coerce') / CAPACITY_KWH

current = pd.read_csv(CURRENT_MODEL_PATH)
current = current.loc[pd.to_numeric(current['group'], errors='coerce') == 3].copy()
current[TIME_COL] = pd.to_datetime(current[TIME_COL])
current['current_model_cf'] = pd.to_numeric(current['final_cf'], errors='coerce')

data = data.merge(previous[[TIME_COL, 'previous_best_cf']], on=TIME_COL, how='left', validate='one_to_one')
data = data.merge(current[[TIME_COL, 'current_model_cf']], on=TIME_COL, how='left', validate='one_to_one')

t = data[TIME_COL]
history_mask = (t >= pd.Timestamp('2023-01-01 01:00:00')) & (t < pd.Timestamp('2024-01-01 01:00:00'))
h1_mask = (t >= pd.Timestamp('2024-01-01 01:00:00')) & (t < pd.Timestamp('2024-07-01 01:00:00'))
q3_mask = (t >= pd.Timestamp('2024-07-01 01:00:00')) & (t < pd.Timestamp('2024-10-01 01:00:00'))
q4_mask = (t >= pd.Timestamp('2024-10-01 01:00:00')) & (t < pd.Timestamp('2025-01-01 01:00:00'))
valid2024_mask = h1_mask | q3_mask | q4_mask
active_mask = data['target_cf'].ge(ACTIVE_CF)

for column, label in [('previous_best_cf', '이전 최고모델'), ('current_model_cf', '현재 통합모델')]:
    if data.loc[q3_mask | q4_mask, column].isna().any():
        raise ValueError(f'{label} 검증예측과 전처리 데이터의 시간이 맞지 않습니다.')

exclude = {TIME_COL, TARGET_COL, OUTAGE_COL, 'target_cf', 'previous_best_cf', 'current_model_cf'}
feature_cols = [c for c in data.columns if c not in exclude]
X = data[feature_cols].apply(pd.to_numeric, errors='coerce')
X = X.replace([np.inf, -np.inf], np.nan)
# 결측치 중앙값도 2023+2024 상반기만 사용하여 Q3/Q4 정보를 보지 않습니다.
train_pool = (history_mask | h1_mask) & active_mask
medians = X.loc[train_pool].median(numeric_only=True)
X = X.fillna(medians).fillna(0.0).astype('float32')
y = data['target_cf'].to_numpy(dtype=float)

print('피처 수:', len(feature_cols))
print('2023 활성 학습행:', int((history_mask & active_mask).sum()))
print('2024 상반기 최근 학습행:', int((h1_mask & active_mask).sum()))
print('Q3 선택 평가행:', official_metrics(y[q3_mask], data.loc[q3_mask, 'previous_best_cf'])['rows'])
print('Q4 최종 확인행:', official_metrics(y[q4_mask], data.loc[q4_mask, 'previous_best_cf'])['rows'])


# ---- 원본 노트북 코드 셀 5 ----
def make_base_model():
    return lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=BASE_ESTIMATORS,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=60,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.30,
        reg_lambda=5.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )

def fit_base(mask, sample_weight=None):
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    model = make_base_model()
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = np.asarray(sample_weight, dtype=float)[idx]
    model.fit(X.iloc[idx], y[idx], **fit_kwargs)
    return model

history_train = history_mask & active_mask
recent_train = h1_mask & active_mask
combined_train = (history_mask | h1_mask) & active_mask

print('2023 전문가 학습...')
history_model = fit_base(history_train)
print('2024 상반기 전문가 학습...')
recent_model = fit_base(recent_train)

evaluation_masks = {'Q3_SELECT': q3_mask.to_numpy(), 'Q4_CONFIRM': q4_mask.to_numpy()}
base_predictions = {name: {} for name in evaluation_masks}
for split_name, mask in evaluation_masks.items():
    base_predictions[split_name]['history_2023'] = np.clip(history_model.predict(X.loc[mask]), 0, PRED_MAX_CF)
    base_predictions[split_name]['recent_2024_H1'] = np.clip(recent_model.predict(X.loc[mask]), 0, PRED_MAX_CF)

# 두 연도 전문가의 예측값을 직접 혼합한 후보
for recent_share in RECENT_BLEND_GRID:
    name = f'year_blend_recent_{recent_share:.2f}'
    for split_name in evaluation_masks:
        hist = base_predictions[split_name]['history_2023']
        recent = base_predictions[split_name]['recent_2024_H1']
        base_predictions[split_name][name] = (1.0 - recent_share) * hist + recent_share * recent

# 하나의 모델에 2024년 표본 가중치를 더 크게 준 후보
weighted_models = {}
for recency_weight in RECENCY_WEIGHT_GRID:
    print(f'최근 가중 전체모델 학습: 2024 가중치={recency_weight}')
    weights = np.ones(len(data), dtype=float)
    weights[h1_mask.to_numpy()] = recency_weight
    model = fit_base(combined_train, weights)
    weighted_models[recency_weight] = model
    name = f'weighted_recent_{recency_weight:.2f}'
    for split_name, mask in evaluation_masks.items():
        base_predictions[split_name][name] = np.clip(model.predict(X.loc[mask]), 0, PRED_MAX_CF)

print('기본 후보 수:', len(base_predictions['Q3_SELECT']))


# ---- 원본 노트북 코드 셀 7 ----
gate_model = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=GATE_ESTIMATORS,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=5,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.75,
    reg_alpha=0.50,
    reg_lambda=6.0,
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1,
)
gate_idx = np.flatnonzero(combined_train.to_numpy())
gate_target = (y[gate_idx] >= HIGH_LABEL_CF).astype(int)
gate_weight = np.where(h1_mask.to_numpy()[gate_idx], 2.0, 1.0)
print('고출력 확률분류기 학습행:', len(gate_idx), '고출력행:', int(gate_target.sum()))
gate_model.fit(X.iloc[gate_idx], gate_target, sample_weight=gate_weight)

gate_probability = {}
for split_name, mask in evaluation_masks.items():
    gate_probability[split_name] = gate_model.predict_proba(X.loc[mask])[:, 1]

high_predictions = {name: {} for name in evaluation_masks}
high_train = combined_train & data['target_cf'].ge(HIGH_TRAIN_CF)
high_idx = np.flatnonzero(high_train.to_numpy())
high_weight = np.where(h1_mask.to_numpy()[high_idx], 2.0, 1.0)
print('고출력 전용 회귀 학습행:', len(high_idx))

for alpha in HIGH_ALPHA_GRID:
    print(f'고출력 Quantile 모델 학습: alpha={alpha}')
    model = lgb.LGBMRegressor(
        objective='quantile',
        alpha=alpha,
        n_estimators=HIGH_ESTIMATORS,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=5,
        min_child_samples=30,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_alpha=0.50,
        reg_lambda=7.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X.iloc[high_idx], y[high_idx], sample_weight=high_weight)
    for split_name, mask in evaluation_masks.items():
        high_predictions[split_name][alpha] = np.clip(model.predict(X.loc[mask]), 0, PRED_MAX_CF)


# ---- 원본 노트북 코드 셀 9 ----
def apply_soft_high(base_pred, high_pred, probability, gamma):
    lift = np.clip(np.asarray(high_pred) - np.asarray(base_pred), 0.0, HIGH_LIFT_CAP)
    return np.clip(np.asarray(base_pred) + gamma * np.asarray(probability) * lift, 0.0, PRED_MAX_CF)

q3_y = y[q3_mask.to_numpy()]
q4_y = y[q4_mask.to_numpy()]
reference_predictions = {
    'previous_best': {
        'Q3': data.loc[q3_mask, 'previous_best_cf'].to_numpy(dtype=float),
        'Q4': data.loc[q4_mask, 'previous_best_cf'].to_numpy(dtype=float),
    },
    'current_model': {
        'Q3': data.loc[q3_mask, 'current_model_cf'].to_numpy(dtype=float),
        'Q4': data.loc[q4_mask, 'current_model_cf'].to_numpy(dtype=float),
    },
}

candidates = []
for reference_name, reference in reference_predictions.items():
    reference_q3_metrics = official_metrics(q3_y, reference['Q3'])
    for base_name, q3_new_base in base_predictions['Q3_SELECT'].items():
        q4_new_base = base_predictions['Q4_CONFIRM'][base_name]
        for baseline_share in BASELINE_SHARE_GRID:
            q3_mixed = baseline_share * reference['Q3'] + (1.0 - baseline_share) * q3_new_base
            q4_mixed = baseline_share * reference['Q4'] + (1.0 - baseline_share) * q4_new_base
            for alpha in HIGH_ALPHA_GRID:
                for gamma in GAMMA_GRID:
                    q3_pred = apply_soft_high(
                        q3_mixed, high_predictions['Q3_SELECT'][alpha],
                        gate_probability['Q3_SELECT'], gamma
                    )
                    q4_pred = apply_soft_high(
                        q4_mixed, high_predictions['Q4_CONFIRM'][alpha],
                        gate_probability['Q4_CONFIRM'], gamma
                    )
                    metric = official_metrics(q3_y, q3_pred)
                    safe = (
                        metric['NMAE'] <= reference_q3_metrics['NMAE'] + 0.003
                        and metric['FICR'] >= reference_q3_metrics['FICR'] - 0.005
                    )
                    candidates.append({
                        'reference_name': reference_name,
                        'base_name': base_name,
                        'baseline_share': baseline_share,
                        'high_alpha': alpha,
                        'gamma': gamma,
                        'safe': safe,
                        'Q3_metrics': metric,
                        'Q3_prediction': q3_pred,
                        'Q4_prediction': q4_pred,
                    })

def choose_best(rows):
    usable = [r for r in rows if r['safe']]
    if not usable:
        usable = rows
    return sorted(
        usable,
        key=lambda r: (
            r['Q3_metrics']['Score'],
            -r['Q3_metrics']['NMAE'],
            -r['gamma'],
        ),
        reverse=True,
    )[0]

best_new = choose_best([r for r in candidates if r['baseline_share'] == 0.0])
best_safe_previous = choose_best([r for r in candidates if r['reference_name'] == 'previous_best'])
best_safe_current = choose_best([r for r in candidates if r['reference_name'] == 'current_model'])
best_safe_overall = choose_best(candidates)

selected = {
    'previous_best': {
        'Q3_prediction': reference_predictions['previous_best']['Q3'],
        'Q4_prediction': reference_predictions['previous_best']['Q4'],
    },
    'current_model': {
        'Q3_prediction': reference_predictions['current_model']['Q3'],
        'Q4_prediction': reference_predictions['current_model']['Q4'],
    },
    'new_only': best_new,
    'safe_vs_previous': best_safe_previous,
    'safe_vs_current': best_safe_current,
    'safe_overall': best_safe_overall,
}

summary_rows = []
for model_name, row in selected.items():
    for split_name, actual, key in [
        ('Q3_SELECT', q3_y, 'Q3_prediction'),
        ('Q4_CONFIRM', q4_y, 'Q4_prediction'),
    ]:
        metric = official_metrics(actual, row[key])
        metric.update(model=model_name, split=split_name)
        summary_rows.append(metric)

summary = pd.DataFrame(summary_rows)
display(summary[['model', 'split', 'rows', 'MAE_kWh', 'NMAE', 'FICR', 'Score', 'within_6_rate', 'within_8_rate', 'signed_bias_cf']])

print('\nQ3에서 선택된 new_only 설정')
print({k: best_new[k] for k in ['reference_name', 'base_name', 'baseline_share', 'high_alpha', 'gamma']})
for label, chosen in [
    ('safe_vs_previous', best_safe_previous),
    ('safe_vs_current', best_safe_current),
    ('safe_overall', best_safe_overall),
]:
    print(f'Q3에서 선택된 {label} 설정')
    print({k: chosen[k] for k in ['reference_name', 'base_name', 'baseline_share', 'high_alpha', 'gamma']})

# Q4에서 실제로 8% 안으로 구조된 행과 밖으로 밀려난 행을 계산합니다.
q4_active = q4_y >= ACTIVE_CF
for reference_name in ['previous_best', 'current_model']:
    reference_pred = selected[reference_name]['Q4_prediction']
    reference_inside = np.abs(reference_pred - q4_y) <= 0.08
    for model_name in ['new_only', 'safe_vs_previous', 'safe_vs_current', 'safe_overall']:
        pred = selected[model_name]['Q4_prediction']
        new_inside = np.abs(pred - q4_y) <= 0.08
        rescued = int((q4_active & ~reference_inside & new_inside).sum())
        lost = int((q4_active & reference_inside & ~new_inside).sum())
        print(f'{reference_name} 대비 {model_name}: 밖→안 {rescued}행, 안→밖 {lost}행, 순효과 {rescued-lost:+d}행')


# ---- 원본 노트북 코드 셀 11 ----
segment_rows = []
for model_name, row in selected.items():
    segment_rows.extend(segment_metrics(q4_y, row['Q4_prediction'], model_name, 'Q4_CONFIRM'))
segments = pd.DataFrame(segment_rows)
display(segments[['model', 'segment', 'rows', 'MAE_kWh', 'NMAE', 'FICR', 'Score', 'within_6_rate', 'within_8_rate', 'signed_bias_cf']])

prediction_output = pd.DataFrame({
    TIME_COL: data.loc[q4_mask, TIME_COL].to_numpy(),
    'actual_kwh': q4_y * CAPACITY_KWH,
    'previous_best_kwh': selected['previous_best']['Q4_prediction'] * CAPACITY_KWH,
    'current_model_kwh': selected['current_model']['Q4_prediction'] * CAPACITY_KWH,
    'new_only_kwh': selected['new_only']['Q4_prediction'] * CAPACITY_KWH,
    'safe_vs_previous_kwh': selected['safe_vs_previous']['Q4_prediction'] * CAPACITY_KWH,
    'safe_vs_current_kwh': selected['safe_vs_current']['Q4_prediction'] * CAPACITY_KWH,
    'safe_overall_kwh': selected['safe_overall']['Q4_prediction'] * CAPACITY_KWH,
    'high_probability': gate_probability['Q4_CONFIRM'],
})

summary.to_csv(OUTPUT_DIR / 'summary_metrics.csv', index=False, encoding='utf-8-sig')
segments.to_csv(OUTPUT_DIR / 'q4_segment_metrics.csv', index=False, encoding='utf-8-sig')
prediction_output.to_csv(OUTPUT_DIR / 'q4_predictions.csv', index=False, encoding='utf-8-sig')

selection_info = {
    'new_only': {k: best_new[k] for k in ['reference_name', 'base_name', 'baseline_share', 'high_alpha', 'gamma']},
    'safe_vs_previous': {k: best_safe_previous[k] for k in ['reference_name', 'base_name', 'baseline_share', 'high_alpha', 'gamma']},
    'safe_vs_current': {k: best_safe_current[k] for k in ['reference_name', 'base_name', 'baseline_share', 'high_alpha', 'gamma']},
    'safe_overall': {k: best_safe_overall[k] for k in ['reference_name', 'base_name', 'baseline_share', 'high_alpha', 'gamma']},
    'note': 'Q3에서 선택하고 Q4에서 최종 확인함',
}
with open(OUTPUT_DIR / 'selected_configuration.json', 'w', encoding='utf-8') as f:
    json.dump(selection_info, f, ensure_ascii=False, indent=2)

# Windows 한글 그래프 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
q4_summary = summary.loc[summary['split'] == 'Q4_CONFIRM'].copy()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = ['dimgray', 'silver', 'tab:blue', 'tab:green', 'tab:orange', 'tab:red']
axes[0].bar(q4_summary['model'], q4_summary['Score'], color=colors[:len(q4_summary)])
axes[0].set_title('Group 3 Q4 공식 Score')
axes[0].set_ylim(max(0, q4_summary['Score'].min() - 0.03), q4_summary['Score'].max() + 0.02)
axes[0].tick_params(axis='x', rotation=15)
axes[1].bar(q4_summary['model'], q4_summary['within_8_rate'], color=colors[:len(q4_summary)])
axes[1].set_title('Group 3 Q4 8% 이내 비율')
axes[1].tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.show()
print('저장 완료:', OUTPUT_DIR)


# ---- 원본 노트북 코드 셀 13 ----
if RUN_REVERSE_CHECK:
    full_2024_train = valid2024_mask & active_mask
    print('2024 전체 전문가 학습 후 2023 역방향 진단...')
    model_2024_full = fit_base(full_2024_train)
    pred_2023_from_2024 = np.clip(model_2024_full.predict(X.loc[history_mask]), 0, PRED_MAX_CF)
    pred_2024_from_2023 = np.clip(history_model.predict(X.loc[valid2024_mask]), 0, PRED_MAX_CF)
    reverse_rows = []
    for direction, actual, pred in [
        ('2023_train_to_2024', y[valid2024_mask.to_numpy()], pred_2024_from_2023),
        ('2024_train_to_2023', y[history_mask.to_numpy()], pred_2023_from_2024),
    ]:
        row = official_metrics(actual, pred)
        row['direction'] = direction
        reverse_rows.append(row)
    reverse_diagnostic = pd.DataFrame(reverse_rows)
    display(reverse_diagnostic[['direction', 'rows', 'NMAE', 'FICR', 'Score', 'within_8_rate', 'signed_bias_cf']])
    reverse_diagnostic.to_csv(OUTPUT_DIR / 'reverse_year_diagnostic.csv', index=False, encoding='utf-8-sig')
else:
    print('RUN_REVERSE_CHECK=False이므로 역방향 진단을 생략했습니다.')


# ---- 원본 노트북 코드 셀 15 ----
from pathlib import Path
import gc

# =====================================================================
# 최종 그룹별 선택 모델
# =====================================================================
FINAL_OUTPUT_DIR = ROOT / 'wind_optimal_group_selector_output'
FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREVIOUS_VALID_PATH = ROOT / 'wind_baram_v2_output' / 'validation_predictions.csv'
CURRENT_VALID_PATH = ROOT / 'wind_hybrid_2stage_output' / 'validation_predictions.csv'
PREVIOUS_SUB_PATH = ROOT / 'wind_baram_v2_output' / 'submission.csv'
CURRENT_SUB_PATH = ROOT / 'wind_hybrid_2stage_output' / 'submission.csv'
DATA_DIR = ROOT / 'wind_baram_v2_preprocessed'

CAPACITIES = {1: 21600.0, 2: 21600.0, 3: 21000.0}
TARGETS = {1: 'kpx_group_1', 2: 'kpx_group_2', 3: 'kpx_group_3'}
FINAL_RECENCY_GRID = [1.0, 1.5, 2.0]
REFERENCE_CURRENT_SHARE_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]
NEW_EXPERT_SHARE_GRID = [0.0, 0.25, 0.50]

required_paths = [
    PREVIOUS_VALID_PATH, CURRENT_VALID_PATH,
    PREVIOUS_SUB_PATH, CURRENT_SUB_PATH,
]
for required_path in required_paths:
    if not required_path.exists():
        raise FileNotFoundError(required_path)

previous_validation_all = pd.read_csv(PREVIOUS_VALID_PATH)
current_validation_all = pd.read_csv(CURRENT_VALID_PATH)
previous_submission = pd.read_csv(PREVIOUS_SUB_PATH)
current_submission = pd.read_csv(CURRENT_SUB_PATH)
for frame in [previous_validation_all, current_validation_all, previous_submission, current_submission]:
    frame[TIME_COL] = pd.to_datetime(frame[TIME_COL])

if not previous_submission[["forecast_id", TIME_COL]].equals(
    current_submission[["forecast_id", TIME_COL]]
):
    raise ValueError('이전 최고와 현재 통합모델 submission의 행 순서가 다릅니다.')


def metrics_with_capacity(actual_cf, pred_cf, capacity):
    global CAPACITY_KWH
    old_capacity = CAPACITY_KWH
    CAPACITY_KWH = float(capacity)
    try:
        return official_metrics(actual_cf, pred_cf)
    finally:
        CAPACITY_KWH = old_capacity


def prepare_final_group(group_id):
    capacity = CAPACITIES[group_id]
    target = TARGETS[group_id]
    train_path = DATA_DIR / f'train_group{group_id}_preprocessed.csv'
    test_path = DATA_DIR / f'test_group{group_id}_preprocessed.csv'
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train[TIME_COL] = pd.to_datetime(train[TIME_COL])
    test[TIME_COL] = pd.to_datetime(test[TIME_COL])
    train['target_cf_final'] = pd.to_numeric(train[target], errors='coerce') / capacity

    prev = previous_validation_all.loc[
        pd.to_numeric(previous_validation_all['group'], errors='coerce') == group_id,
        [TIME_COL, 'prediction_kwh'],
    ].copy()
    prev['previous_cf_final'] = pd.to_numeric(prev['prediction_kwh'], errors='coerce') / capacity
    current = current_validation_all.loc[
        pd.to_numeric(current_validation_all['group'], errors='coerce') == group_id,
        [TIME_COL, 'final_cf'],
    ].copy()
    current['current_cf_final'] = pd.to_numeric(current['final_cf'], errors='coerce')
    train = train.merge(prev[[TIME_COL, 'previous_cf_final']], on=TIME_COL, how='left', validate='one_to_one')
    train = train.merge(current[[TIME_COL, 'current_cf_final']], on=TIME_COL, how='left', validate='one_to_one')

    excluded = {
        TIME_COL, target, OUTAGE_COL, 'target_cf_final',
        'previous_cf_final', 'current_cf_final',
    }
    feature_columns = [c for c in train.columns if c not in excluded and c in test.columns]
    x_train_raw = train[feature_columns].apply(pd.to_numeric, errors='coerce')
    x_test_raw = test[feature_columns].apply(pd.to_numeric, errors='coerce')
    x_train_raw = x_train_raw.replace([np.inf, -np.inf], np.nan)
    x_test_raw = x_test_raw.replace([np.inf, -np.inf], np.nan)

    times = train[TIME_COL]
    history = times < pd.Timestamp('2024-01-01 01:00:00')
    h1 = (times >= pd.Timestamp('2024-01-01 01:00:00')) & (times < pd.Timestamp('2024-07-01 01:00:00'))
    q3 = (times >= pd.Timestamp('2024-07-01 01:00:00')) & (times < pd.Timestamp('2024-10-01 01:00:00'))
    q4 = (times >= pd.Timestamp('2024-10-01 01:00:00')) & (times < pd.Timestamp('2025-01-01 01:00:00'))
    y_cf = train['target_cf_final'].to_numpy(dtype=float)
    active = np.isfinite(y_cf) & (y_cf >= ACTIVE_CF)
    validation_train = (history | h1).to_numpy() & active
    median_values = x_train_raw.loc[validation_train].median(numeric_only=True)
    x_train = x_train_raw.fillna(median_values).fillna(0.0).astype('float32')
    x_test = x_test_raw.fillna(median_values).fillna(0.0).astype('float32')

    return {
        'group': group_id,
        'capacity': capacity,
        'target': target,
        'train': train,
        'test': test,
        'X': x_train,
        'X_test': x_test,
        'y': y_cf,
        'active': active,
        'history': history.to_numpy(),
        'h1': h1.to_numpy(),
        'q3': q3.to_numpy(),
        'q4': q4.to_numpy(),
        'validation_train': validation_train,
    }


def make_group_expert(group_id):
    objective = 'quantile' if group_id == 2 else 'regression_l1'
    params = dict(
        objective=objective,
        n_estimators=BASE_ESTIMATORS,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=80 if group_id == 1 else 55,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.25,
        reg_lambda=6.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    if group_id == 2:
        params['alpha'] = 0.58
    return lgb.LGBMRegressor(**params)


def expert_training_weight(group_id, y_cf, times, recency_weight):
    weight = np.ones(len(y_cf), dtype=float)
    weight *= np.where(times >= np.datetime64('2024-01-01T01:00:00'), recency_weight, 1.0)
    if group_id == 1:
        output_weight = np.select(
            [y_cf < 0.50, y_cf < 0.70, y_cf < 0.90, y_cf >= 0.90],
            [1.0, 1.2, 2.0, 6.0],
            default=1.0,
        )
        weight *= output_weight
    return weight / np.mean(weight)


def reference_predictions_for_group(info, split_mask, current_share):
    train = info['train']
    previous = train.loc[split_mask, 'previous_cf_final'].to_numpy(dtype=float)
    current = train.loc[split_mask, 'current_cf_final'].to_numpy(dtype=float)
    return (1.0 - current_share) * previous + current_share * current


def reference_test_for_group(group_id, current_share):
    target = TARGETS[group_id]
    previous = previous_submission[target].to_numpy(dtype=float) / CAPACITIES[group_id]
    current = current_submission[target].to_numpy(dtype=float) / CAPACITIES[group_id]
    return (1.0 - current_share) * previous + current_share * current


def select_group12(group_id):
    info = prepare_final_group(group_id)
    y_cf = info['y']
    q3_y = y_cf[info['q3']]
    q4_y = y_cf[info['q4']]
    previous_q3 = reference_predictions_for_group(info, info['q3'], 0.0)
    previous_q4 = reference_predictions_for_group(info, info['q4'], 0.0)
    previous_q3_metric = metrics_with_capacity(q3_y, previous_q3, info['capacity'])
    previous_q4_metric = metrics_with_capacity(q4_y, previous_q4, info['capacity'])
    previous_mean_score = 0.5 * (previous_q3_metric['Score'] + previous_q4_metric['Score'])

    expert_results = {}
    training_mask = info['validation_train']
    train_indices = np.flatnonzero(training_mask)
    train_times = info['train'][TIME_COL].to_numpy(dtype='datetime64[ns]')
    for recency_weight in FINAL_RECENCY_GRID:
        print(f'Group {group_id}: 최근가중 전문가 학습 r={recency_weight}')
        weights = expert_training_weight(group_id, y_cf, train_times, recency_weight)
        model = make_group_expert(group_id)
        model.fit(
            info['X'].iloc[train_indices], y_cf[train_indices],
            sample_weight=weights[train_indices],
        )
        expert_results[recency_weight] = {
            'Q3': np.clip(model.predict(info['X'].loc[info['q3']]), 0, PRED_MAX_CF),
            'Q4': np.clip(model.predict(info['X'].loc[info['q4']]), 0, PRED_MAX_CF),
        }
        del model
        gc.collect()

    candidates = []
    for current_share in REFERENCE_CURRENT_SHARE_GRID:
        ref_q3 = reference_predictions_for_group(info, info['q3'], current_share)
        ref_q4 = reference_predictions_for_group(info, info['q4'], current_share)
        for recency_weight, expert in expert_results.items():
            for expert_share in NEW_EXPERT_SHARE_GRID:
                pred_q3 = (1.0 - expert_share) * ref_q3 + expert_share * expert['Q3']
                pred_q4 = (1.0 - expert_share) * ref_q4 + expert_share * expert['Q4']
                m3 = metrics_with_capacity(q3_y, pred_q3, info['capacity'])
                m4 = metrics_with_capacity(q4_y, pred_q4, info['capacity'])
                mean_score = 0.5 * (m3['Score'] + m4['Score'])
                passed = bool(
                    m3['Score'] >= previous_q3_metric['Score'] - 0.002
                    and m4['Score'] >= previous_q4_metric['Score'] + 0.0002
                    and mean_score >= previous_mean_score + 0.0005
                    and m4['NMAE'] <= previous_q4_metric['NMAE'] + 0.001
                    and m4['FICR'] >= previous_q4_metric['FICR'] - 0.002
                )
                candidates.append({
                    'group': group_id,
                    'current_share': current_share,
                    'recency_weight': recency_weight,
                    'expert_share': expert_share,
                    'passed': passed,
                    'Q3': m3,
                    'Q4': m4,
                    'mean_score': mean_score,
                    'Q3_prediction': pred_q3,
                    'Q4_prediction': pred_q4,
                })

    approved = [row for row in candidates if row['passed']]
    if approved:
        chosen = max(approved, key=lambda row: (row['mean_score'], row['Q4']['Score']))
        chosen['source'] = 'approved_recent_ensemble'
    else:
        chosen = {
            'group': group_id,
            'current_share': 0.0,
            'recency_weight': 1.0,
            'expert_share': 0.0,
            'passed': False,
            'source': 'previous_best_fallback',
            'Q3': previous_q3_metric,
            'Q4': previous_q4_metric,
            'mean_score': previous_mean_score,
            'Q3_prediction': previous_q3,
            'Q4_prediction': previous_q4,
        }

    final_test = reference_test_for_group(group_id, chosen['current_share'])
    if chosen['expert_share'] > 0:
        full_mask = info['active']
        full_indices = np.flatnonzero(full_mask)
        train_times = info['train'][TIME_COL].to_numpy(dtype='datetime64[ns]')
        full_weights = expert_training_weight(
            group_id, y_cf, train_times, chosen['recency_weight']
        )
        final_model = make_group_expert(group_id)
        final_model.fit(
            info['X'].iloc[full_indices], y_cf[full_indices],
            sample_weight=full_weights[full_indices],
        )
        expert_test = np.clip(final_model.predict(info['X_test']), 0, PRED_MAX_CF)
        expert_series = pd.Series(
            expert_test, index=pd.DatetimeIndex(info['test'][TIME_COL])
        ).groupby(level=0).mean()
        aligned_expert = previous_submission[TIME_COL].map(expert_series).to_numpy(dtype=float)
        if np.isnan(aligned_expert).any():
            raise ValueError(f'Group {group_id} Test 전문가 시간 정렬 실패')
        final_test = (
            (1.0 - chosen['expert_share']) * final_test
            + chosen['expert_share'] * aligned_expert
        )
        del final_model

    current_q3 = reference_predictions_for_group(info, info['q3'], 1.0)
    current_q4 = reference_predictions_for_group(info, info['q4'], 1.0)
    chosen['previous_Q3'] = previous_q3_metric
    chosen['previous_Q4'] = previous_q4_metric
    chosen['current_Q3'] = metrics_with_capacity(q3_y, current_q3, info['capacity'])
    chosen['current_Q4'] = metrics_with_capacity(q4_y, current_q4, info['capacity'])

    candidate_rows = []
    for row in sorted(candidates, key=lambda value: value['mean_score'], reverse=True)[:20]:
        candidate_rows.append({
            'group': group_id,
            'current_share': row['current_share'],
            'recency_weight': row['recency_weight'],
            'expert_share': row['expert_share'],
            'passed': row['passed'],
            'Q3_Score': row['Q3']['Score'],
            'Q4_Score': row['Q4']['Score'],
            'Q4_NMAE': row['Q4']['NMAE'],
            'Q4_FICR': row['Q4']['FICR'],
            'mean_score': row['mean_score'],
        })
    del info
    gc.collect()
    return chosen, np.clip(final_test, 0, PRED_MAX_CF), candidate_rows


def segment_metric_for_selection(actual, pred, low, high):
    mask = (np.asarray(actual) >= low) & (np.asarray(actual) < high)
    return metrics_with_capacity(
        np.asarray(actual)[mask], np.asarray(pred)[mask], CAPACITIES[3]
    )


def apply_scoped_high(base_pred, high_pred, probability, gate_cutoff, base_min, gamma, cap):
    base_pred = np.asarray(base_pred, dtype=float)
    probability = np.asarray(probability, dtype=float)
    strength = np.clip((probability - gate_cutoff) / max(1e-6, 1.0 - gate_cutoff), 0.0, 1.0)
    scope = (probability >= gate_cutoff) & (base_pred >= base_min)
    lift = np.clip(np.asarray(high_pred) - base_pred, 0.0, cap)
    return np.clip(base_pred + scope * gamma * strength * lift, 0.0, PRED_MAX_CF)


def select_group3_scoped_gate():
    q3_y = y[q3_mask.to_numpy()]
    q4_y = y[q4_mask.to_numpy()]
    previous_q3 = reference_predictions['previous_best']['Q3']
    previous_q4 = reference_predictions['previous_best']['Q4']
    previous_m3 = official_metrics(q3_y, previous_q3)
    previous_m4 = official_metrics(q4_y, previous_q4)
    previous_mid = segment_metric_for_selection(q4_y, previous_q4, 0.60, 0.80)
    previous_high = segment_metric_for_selection(q4_y, previous_q4, 0.80, np.inf)

    weighted_base_names = [
        name for name in base_predictions['Q3_SELECT']
        if name.startswith('weighted_recent_')
    ]
    candidates = []
    for current_share in [0.0, 0.25, 0.50]:
        ref_q3 = (
            (1.0 - current_share) * reference_predictions['previous_best']['Q3']
            + current_share * reference_predictions['current_model']['Q3']
        )
        ref_q4 = (
            (1.0 - current_share) * reference_predictions['previous_best']['Q4']
            + current_share * reference_predictions['current_model']['Q4']
        )
        for base_name in weighted_base_names:
            recency_weight = float(base_name.rsplit('_', 1)[-1])
            for new_base_share in [0.0, 0.25, 0.50]:
                base_q3 = (
                    (1.0 - new_base_share) * ref_q3
                    + new_base_share * base_predictions['Q3_SELECT'][base_name]
                )
                base_q4 = (
                    (1.0 - new_base_share) * ref_q4
                    + new_base_share * base_predictions['Q4_CONFIRM'][base_name]
                )
                for alpha in HIGH_ALPHA_GRID:
                    for gate_cutoff in [0.25, 0.35, 0.45, 0.55]:
                        for base_min in [0.55, 0.65, 0.75]:
                            for gamma in [0.10, 0.20, 0.30, 0.40]:
                                for cap in [0.10, 0.15]:
                                    pred_q3 = apply_scoped_high(
                                        base_q3, high_predictions['Q3_SELECT'][alpha],
                                        gate_probability['Q3_SELECT'], gate_cutoff,
                                        base_min, gamma, cap,
                                    )
                                    pred_q4 = apply_scoped_high(
                                        base_q4, high_predictions['Q4_CONFIRM'][alpha],
                                        gate_probability['Q4_CONFIRM'], gate_cutoff,
                                        base_min, gamma, cap,
                                    )
                                    m3 = official_metrics(q3_y, pred_q3)
                                    m4 = official_metrics(q4_y, pred_q4)
                                    mid = segment_metric_for_selection(q4_y, pred_q4, 0.60, 0.80)
                                    high = segment_metric_for_selection(q4_y, pred_q4, 0.80, np.inf)
                                    active_q4 = q4_y >= ACTIVE_CF
                                    old_inside = np.abs(previous_q4 - q4_y) <= 0.08
                                    new_inside = np.abs(pred_q4 - q4_y) <= 0.08
                                    rescued = int((active_q4 & ~old_inside & new_inside).sum())
                                    lost = int((active_q4 & old_inside & ~new_inside).sum())
                                    passed = bool(
                                        m3['Score'] >= previous_m3['Score'] - 0.002
                                        and m4['Score'] >= previous_m4['Score'] + 0.0002
                                        and m4['NMAE'] <= previous_m4['NMAE'] + 0.001
                                        and m4['FICR'] >= previous_m4['FICR'] - 0.001
                                        and mid['Score'] >= previous_mid['Score'] - 0.010
                                        and high['Score'] >= previous_high['Score'] + 0.005
                                        and rescued >= lost
                                    )
                                    candidates.append({
                                        'group': 3,
                                        'current_share': current_share,
                                        'base_name': base_name,
                                        'recency_weight': recency_weight,
                                        'new_base_share': new_base_share,
                                        'alpha': alpha,
                                        'gate_cutoff': gate_cutoff,
                                        'base_min': base_min,
                                        'gamma': gamma,
                                        'cap': cap,
                                        'passed': passed,
                                        'Q3': m3,
                                        'Q4': m4,
                                        'Q4_mid_score': mid['Score'],
                                        'Q4_high_score': high['Score'],
                                        'rescued': rescued,
                                        'lost': lost,
                                        'mean_score': 0.5 * (m3['Score'] + m4['Score']),
                                        'Q3_prediction': pred_q3,
                                        'Q4_prediction': pred_q4,
                                    })

    approved = [row for row in candidates if row['passed']]
    if approved:
        chosen = max(approved, key=lambda row: (row['mean_score'], row['Q4']['Score']))
        chosen['source'] = 'approved_scoped_high_gate'
    else:
        chosen = {
            'group': 3,
            'current_share': 0.0,
            'base_name': 'weighted_recent_1.00',
            'recency_weight': 1.0,
            'new_base_share': 0.0,
            'alpha': 0.65,
            'gate_cutoff': 1.0,
            'base_min': 1.0,
            'gamma': 0.0,
            'cap': 0.10,
            'passed': False,
            'source': 'previous_best_fallback',
            'Q3': previous_m3,
            'Q4': previous_m4,
            'Q4_mid_score': previous_mid['Score'],
            'Q4_high_score': previous_high['Score'],
            'rescued': 0,
            'lost': 0,
            'mean_score': 0.5 * (previous_m3['Score'] + previous_m4['Score']),
            'Q3_prediction': previous_q3,
            'Q4_prediction': previous_q4,
        }

    final_reference = reference_test_for_group(3, chosen['current_share'])
    if chosen['source'] == 'approved_scoped_high_gate':
        full_active = np.isfinite(y) & (y >= ACTIVE_CF)
        full_indices = np.flatnonzero(full_active)
        full_times = data[TIME_COL].to_numpy(dtype='datetime64[ns]')
        base_weights = np.where(
            full_times >= np.datetime64('2024-01-01T01:00:00'),
            chosen['recency_weight'], 1.0,
        )
        final_base_model = fit_base(full_active, base_weights)

        test3 = pd.read_csv(DATA_DIR / 'test_group3_preprocessed.csv')
        test3[TIME_COL] = pd.to_datetime(test3[TIME_COL])
        x_test3 = test3[feature_cols].apply(pd.to_numeric, errors='coerce')
        x_test3 = x_test3.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0).astype('float32')
        recent_test = np.clip(final_base_model.predict(x_test3), 0, PRED_MAX_CF)
        recent_series = pd.Series(recent_test, index=pd.DatetimeIndex(test3[TIME_COL])).groupby(level=0).mean()
        recent_aligned = previous_submission[TIME_COL].map(recent_series).to_numpy(dtype=float)

        full_gate = lgb.LGBMClassifier(
            objective='binary', n_estimators=GATE_ESTIMATORS,
            learning_rate=0.03, num_leaves=15, max_depth=5,
            min_child_samples=80, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.75, reg_alpha=0.50, reg_lambda=6.0,
            random_state=SEED, n_jobs=-1, verbosity=-1,
        )
        gate_target_full = (y[full_indices] >= HIGH_LABEL_CF).astype(int)
        gate_weight_full = np.where(
            full_times[full_indices] >= np.datetime64('2024-01-01T01:00:00'), 2.0, 1.0
        )
        full_gate.fit(X.iloc[full_indices], gate_target_full, sample_weight=gate_weight_full)
        test_probability = full_gate.predict_proba(x_test3)[:, 1]

        full_high_mask = full_active & (y >= HIGH_TRAIN_CF)
        full_high_indices = np.flatnonzero(full_high_mask)
        high_weight_full = np.where(
            full_times[full_high_indices] >= np.datetime64('2024-01-01T01:00:00'), 2.0, 1.0
        )
        final_high_model = lgb.LGBMRegressor(
            objective='quantile', alpha=chosen['alpha'],
            n_estimators=HIGH_ESTIMATORS, learning_rate=0.025,
            num_leaves=15, max_depth=5, min_child_samples=30,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.70,
            reg_alpha=0.50, reg_lambda=7.0, random_state=SEED,
            n_jobs=-1, verbosity=-1,
        )
        final_high_model.fit(
            X.iloc[full_high_indices], y[full_high_indices],
            sample_weight=high_weight_full,
        )
        high_test = np.clip(final_high_model.predict(x_test3), 0, PRED_MAX_CF)
        base_test = (
            (1.0 - chosen['new_base_share']) * final_reference
            + chosen['new_base_share'] * recent_aligned
        )
        final_test = apply_scoped_high(
            base_test, high_test, test_probability,
            chosen['gate_cutoff'], chosen['base_min'],
            chosen['gamma'], chosen['cap'],
        )
    else:
        final_test = final_reference

    chosen['previous_Q3'] = previous_m3
    chosen['previous_Q4'] = previous_m4
    chosen['current_Q3'] = official_metrics(q3_y, reference_predictions['current_model']['Q3'])
    chosen['current_Q4'] = official_metrics(q4_y, reference_predictions['current_model']['Q4'])

    top_rows = []
    for row in sorted(candidates, key=lambda value: value['mean_score'], reverse=True)[:30]:
        top_rows.append({
            'group': 3,
            'current_share': row['current_share'],
            'recency_weight': row['recency_weight'],
            'new_base_share': row['new_base_share'],
            'alpha': row['alpha'],
            'gate_cutoff': row['gate_cutoff'],
            'base_min': row['base_min'],
            'gamma': row['gamma'],
            'cap': row['cap'],
            'passed': row['passed'],
            'Q3_Score': row['Q3']['Score'],
            'Q4_Score': row['Q4']['Score'],
            'Q4_NMAE': row['Q4']['NMAE'],
            'Q4_FICR': row['Q4']['FICR'],
            'Q4_mid_score': row['Q4_mid_score'],
            'Q4_high_score': row['Q4_high_score'],
            'rescued': row['rescued'],
            'lost': row['lost'],
            'mean_score': row['mean_score'],
        })
    return chosen, np.clip(final_test, 0, PRED_MAX_CF), top_rows


group_selections = {}
final_group_predictions = {}
top_candidate_rows = []

for group_id in [1, 2]:
    print('=' * 78)
    print(f'Group {group_id} 최종 선택')
    chosen, test_prediction, candidate_rows = select_group12(group_id)
    group_selections[group_id] = chosen
    final_group_predictions[group_id] = test_prediction
    top_candidate_rows.extend(candidate_rows)
    print('선택:', chosen['source'])
    print({key: chosen[key] for key in ['current_share', 'recency_weight', 'expert_share']})
    print('Q3 Score:', chosen['Q3']['Score'], 'Q4 Score:', chosen['Q4']['Score'])

print('=' * 78)
print('Group 3 제한형 고출력 Gate 최종 선택')
chosen3, test_prediction3, candidate_rows3 = select_group3_scoped_gate()
group_selections[3] = chosen3
final_group_predictions[3] = test_prediction3
top_candidate_rows.extend(candidate_rows3)
print('선택:', chosen3['source'])
print({
    key: chosen3[key]
    for key in [
        'current_share', 'recency_weight', 'new_base_share', 'alpha',
        'gate_cutoff', 'base_min', 'gamma', 'cap', 'rescued', 'lost',
    ]
})
print('Q3 Score:', chosen3['Q3']['Score'], 'Q4 Score:', chosen3['Q4']['Score'])


# 검증 요약
validation_rows = []
for group_id, chosen in group_selections.items():
    for model_name, key_prefix in [
        ('previous_best', 'previous_'),
        ('current_model', 'current_'),
        ('selected', ''),
    ]:
        for split_name in ['Q3', 'Q4']:
            metric = chosen[f'{key_prefix}{split_name}'] if key_prefix else chosen[split_name]
            validation_rows.append({
                'group': group_id,
                'model': model_name,
                'split': split_name,
                **metric,
            })
validation_summary_final = pd.DataFrame(validation_rows)
display(validation_summary_final[[
    'group', 'model', 'split', 'rows', 'MAE_kWh', 'NMAE',
    'FICR', 'Score', 'within_6_rate', 'within_8_rate', 'signed_bias_cf',
]])


# 최종 제출 파일
final_submission = previous_submission.copy()
for group_id, target in TARGETS.items():
    final_submission[target] = np.clip(
        final_group_predictions[group_id] * CAPACITIES[group_id],
        0.0,
        CAPACITIES[group_id] * PRED_MAX_CF,
    )

final_submission.to_csv(
    FINAL_OUTPUT_DIR / 'submission_optimal_selector.csv',
    index=False,
    encoding='utf-8-sig',
)
validation_summary_final.to_csv(
    FINAL_OUTPUT_DIR / 'validation_summary.csv',
    index=False,
    encoding='utf-8-sig',
)
pd.DataFrame(top_candidate_rows).to_csv(
    FINAL_OUTPUT_DIR / 'top_candidates.csv',
    index=False,
    encoding='utf-8-sig',
)

serializable_selection = {}
for group_id, chosen in group_selections.items():
    keys = [
        'source', 'current_share', 'recency_weight', 'expert_share',
        'new_base_share', 'alpha', 'gate_cutoff', 'base_min',
        'gamma', 'cap', 'rescued', 'lost',
    ]
    serializable_selection[str(group_id)] = {
        key: chosen[key] for key in keys if key in chosen
    }
with open(FINAL_OUTPUT_DIR / 'selected_configuration.json', 'w', encoding='utf-8') as file:
    json.dump(serializable_selection, file, ensure_ascii=False, indent=2)

print('\n최종 저장 완료:', FINAL_OUTPUT_DIR)
print(final_submission.head())



# =============================================================================
# 7. Group 2 Gaussian MoE
# =============================================================================

# ---- 원본 노트북 코드 셀 1 ----
from pathlib import Path
import gc
import json
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError('lightgbm이 필요합니다. 현재 Jupyter 환경에 설치해 주세요.') from exc

warnings.filterwarnings('ignore', category=FutureWarning)

ROOT = REPO_ROOT
DATA_DIR = ROOT / 'wind_baram_v2_preprocessed'
PREVIOUS_DIR = ROOT / 'wind_baram_v2_output'
CURRENT_DIR = ROOT / 'wind_hybrid_2stage_output'
OUTPUT_DIR = ROOT / 'wind_g2_gaussian_moe_output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / 'train_group2_preprocessed.csv'
TEST_PATH = DATA_DIR / 'test_group2_preprocessed.csv'
PREVIOUS_VALID_PATH = PREVIOUS_DIR / 'validation_predictions.csv'
CURRENT_VALID_PATH = CURRENT_DIR / 'validation_predictions.csv'
PREVIOUS_SUB_PATH = PREVIOUS_DIR / 'submission.csv'
CURRENT_SUB_PATH = CURRENT_DIR / 'submission.csv'

SEED = 42
CAPACITY_KWH = 21600.0
ACTIVE_CF = 0.10
PRED_MAX_CF = 1.05
TIME_COL = 'forecast_kst_dtm'
TARGET_COL = 'kpx_group_2'
OUTAGE_COL = 'is_long_outage_hour'

# 빠른 구조 확인은 True, 최종 실험은 False입니다.
FAST_MODE = False
N_ESTIMATORS = 450 if FAST_MODE else 1800
N_DISTRIBUTIONS = 250 if FAST_MODE else 3000
TOP_Q3_CANDIDATES = 40 if FAST_MODE else 100

rng = np.random.default_rng(SEED)
started_at = time.time()
print('출력 폴더:', OUTPUT_DIR)
print('FAST_MODE:', FAST_MODE)


# ---- 원본 노트북 코드 셀 3 ----
def official_metrics(y_true_cf, y_pred_cf):
    y_true = np.asarray(y_true_cf, dtype=float)
    pred = np.asarray(y_pred_cf, dtype=float)
    use = np.isfinite(y_true) & np.isfinite(pred) & (y_true >= ACTIVE_CF)
    if not use.any():
        raise ValueError('공식 평가 대상 행이 없습니다.')
    actual, estimate = y_true[use], pred[use]
    error = np.abs(estimate - actual)
    rate = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
    actual_kwh = actual * CAPACITY_KWH
    nmae = float(error.mean())
    ficr = float((rate * actual_kwh).sum() / (4.0 * actual_kwh).sum())
    return {
        'rows': int(use.sum()), 'MAE_kWh': nmae * CAPACITY_KWH,
        'NMAE': nmae, 'one_minus_NMAE': 1.0 - nmae,
        'FICR': ficr, 'Score': 0.5 * (1.0 - nmae + ficr),
        'within_6_rate': float((error <= 0.06).mean()),
        'within_8_rate': float((error <= 0.08).mean()),
        'signed_bias_cf': float((estimate - actual).mean()),
    }


def segment_metrics(y_true, pred, model, split):
    definitions = [('10_30pct', .10, .30), ('30_60pct', .30, .60),
                   ('60_80pct', .60, .80), ('80_100pct', .80, np.inf)]
    rows = []
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    for name, low, high in definitions:
        use = (y_true >= low) & (y_true < high)
        if use.any():
            row = official_metrics(y_true[use], pred[use])
            row.update(model=model, split=split, segment=name)
            rows.append(row)
    return rows


for path in [TRAIN_PATH, TEST_PATH, PREVIOUS_VALID_PATH, CURRENT_VALID_PATH,
             PREVIOUS_SUB_PATH, CURRENT_SUB_PATH]:
    if not path.exists():
        raise FileNotFoundError(path)

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
previous_valid = pd.read_csv(PREVIOUS_VALID_PATH)
current_valid = pd.read_csv(CURRENT_VALID_PATH)
previous_sub = pd.read_csv(PREVIOUS_SUB_PATH)
current_sub = pd.read_csv(CURRENT_SUB_PATH)
for frame in [train, test, previous_valid, current_valid, previous_sub, current_sub]:
    frame[TIME_COL] = pd.to_datetime(frame[TIME_COL])

if not previous_sub[['forecast_id', TIME_COL]].equals(current_sub[['forecast_id', TIME_COL]]):
    raise ValueError('기존 최고와 현재 submission의 행 순서가 다릅니다.')

previous_valid = previous_valid.loc[pd.to_numeric(previous_valid['group'], errors='coerce') == 2].copy()
current_valid = current_valid.loc[pd.to_numeric(current_valid['group'], errors='coerce') == 2].copy()
previous_valid['previous_cf'] = pd.to_numeric(previous_valid['prediction_kwh'], errors='coerce') / CAPACITY_KWH
current_valid['current_cf'] = pd.to_numeric(current_valid['final_cf'], errors='coerce')
train = train.merge(previous_valid[[TIME_COL, 'previous_cf']], on=TIME_COL, how='left', validate='one_to_one')
train = train.merge(current_valid[[TIME_COL, 'current_cf']], on=TIME_COL, how='left', validate='one_to_one')
train['target_cf'] = pd.to_numeric(train[TARGET_COL], errors='coerce') / CAPACITY_KWH

t = train[TIME_COL]
fit_mask = (t < pd.Timestamp('2024-07-01 01:00:00')) & train['target_cf'].ge(ACTIVE_CF)
q3_mask = (t >= pd.Timestamp('2024-07-01 01:00:00')) & (t < pd.Timestamp('2024-10-01 01:00:00'))
q4_mask = (t >= pd.Timestamp('2024-10-01 01:00:00')) & (t < pd.Timestamp('2025-01-01 01:00:00'))
active_mask = train['target_cf'].ge(ACTIVE_CF) & train['target_cf'].notna()

excluded = {TIME_COL, TARGET_COL, OUTAGE_COL, 'target_cf', 'previous_cf', 'current_cf'}
feature_cols = [column for column in train.columns if column not in excluded and column in test.columns]
X_raw = train[feature_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
X_test_raw = test[feature_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
medians = X_raw.loc[fit_mask].median(numeric_only=True)
X = X_raw.fillna(medians).fillna(0.0).astype('float32')
X_test = X_test_raw.fillna(medians).fillna(0.0).astype('float32')
y = train['target_cf'].to_numpy(dtype=float)

fit_use = fit_mask.to_numpy(); q3 = q3_mask.to_numpy(); q4 = q4_mask.to_numpy(); active = active_mask.to_numpy()
q3_y, q4_y = y[q3], y[q4]
previous_q3 = train.loc[q3_mask, 'previous_cf'].to_numpy(dtype=float)
previous_q4 = train.loc[q4_mask, 'previous_cf'].to_numpy(dtype=float)
current_q3 = train.loc[q3_mask, 'current_cf'].to_numpy(dtype=float)
current_q4 = train.loc[q4_mask, 'current_cf'].to_numpy(dtype=float)

print('피처 수:', len(feature_cols))
print('전문가 학습행:', int(fit_use.sum()))
print('Q3/Q4 평가행:', official_metrics(q3_y, previous_q3)['rows'], official_metrics(q4_y, previous_q4)['rows'])


# ---- 원본 노트북 코드 셀 5 ----
EXPERT_NAMES = ['10_30', '30_60', '60_80', '80_100']
TRAIN_CENTERS = np.array([0.20, 0.45, 0.70, 0.90])
TRAIN_SIGMAS = np.array([0.11, 0.15, 0.12, 0.10])


def make_expert():
    return lgb.LGBMRegressor(
        objective='regression_l1', n_estimators=N_ESTIMATORS,
        learning_rate=0.025, num_leaves=39, max_depth=7,
        min_child_samples=55, subsample=0.88, subsample_freq=1,
        colsample_bytree=0.78, reg_alpha=0.30, reg_lambda=7.0,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )


def expert_sample_weight(center, sigma, use_mask):
    gaussian = np.exp(-0.5 * ((y - center) / sigma) ** 2)
    # 경계 밖도 완전히 버리지 않되 중심부가 약 10배 중요하도록 합니다.
    weight = 0.10 + 0.90 * gaussian
    is_2024 = train[TIME_COL].to_numpy(dtype='datetime64[ns]') >= np.datetime64('2024-01-01T01:00:00')
    weight *= np.where(is_2024, 1.5, 1.0)
    return weight / weight[use_mask].mean()


fit_indices = np.flatnonzero(fit_use)
expert_predictions = {'Q3': [], 'Q4': []}
for name, center, sigma in zip(EXPERT_NAMES, TRAIN_CENTERS, TRAIN_SIGMAS):
    print(f'{name} 전문가 학습: center={center:.2f}, sigma={sigma:.2f}')
    weight = expert_sample_weight(center, sigma, fit_use)
    model = make_expert()
    model.fit(X.iloc[fit_indices], y[fit_indices], sample_weight=weight[fit_indices])
    expert_predictions['Q3'].append(np.clip(model.predict(X.loc[q3_mask]), 0.0, PRED_MAX_CF))
    expert_predictions['Q4'].append(np.clip(model.predict(X.loc[q4_mask]), 0.0, PRED_MAX_CF))
    del model
    gc.collect()

expert_predictions['Q3'] = np.column_stack(expert_predictions['Q3'])
expert_predictions['Q4'] = np.column_stack(expert_predictions['Q4'])
print('Q3 전문가 예측 행렬:', expert_predictions['Q3'].shape)


# ---- 원본 노트북 코드 셀 7 ----
def gaussian_moe_predict(base_cf, expert_matrix, params, return_weights=False):
    base = np.asarray(base_cf, dtype=float)
    experts = np.asarray(expert_matrix, dtype=float)
    centers = np.asarray(params['centers'], dtype=float)
    sigmas = np.asarray(params['sigmas'], dtype=float)
    amplitudes = np.asarray(params['amplitudes'], dtype=float)

    raw = amplitudes[None, :] * np.exp(-0.5 * ((base[:, None] - centers[None, :]) / sigmas[None, :]) ** 2)
    weights = raw / np.maximum(raw.sum(axis=1, keepdims=True), 1e-12)
    expert_mix = np.sum(weights * experts, axis=1)
    disagreement = np.sqrt(np.sum(weights * (experts - expert_mix[:, None]) ** 2, axis=1))
    confidence = np.exp(-params['damping'] * disagreement)
    correction = np.clip(expert_mix - base, -params['cap'], params['cap'])
    strength = params['base_alpha'] * confidence
    prediction = np.clip(base + strength * correction, 0.0, PRED_MAX_CF)
    if return_weights:
        return prediction, weights, strength, expert_mix
    return prediction


def simple_segment_control(previous, current):
    # 앞선 분석에서 Q3·Q4 모두 개선된 작은 대조군입니다.
    low, high, width, current_share = 0.35, 0.75, 0.04, 0.25
    left = 1.0 / (1.0 + np.exp(-(previous - low) / width))
    right = 1.0 / (1.0 + np.exp((previous - high) / width))
    gate = left * right
    return previous + current_share * gate * (current - previous)


control_q3 = simple_segment_control(previous_q3, current_q3)
control_q4 = simple_segment_control(previous_q4, current_q4)


# ---- 원본 노트북 코드 셀 9 ----
def random_distribution():
    centers = np.array([
        rng.uniform(.16, .25), rng.uniform(.37, .53),
        rng.uniform(.62, .78), rng.uniform(.84, .98),
    ])
    sigmas = np.array([
        rng.uniform(.07, .18), rng.uniform(.09, .22),
        rng.uniform(.08, .20), rng.uniform(.06, .17),
    ])
    return {
        'centers': centers.tolist(), 'sigmas': sigmas.tolist(),
        'amplitudes': rng.uniform(.55, 1.45, 4).tolist(),
        'base_alpha': float(rng.uniform(.08, .55)),
        'damping': float(rng.choice([0.0, 3.0, 6.0, 10.0])),
        'cap': float(rng.choice([.03, .05, .07, .10])),
    }


presets = [
    {'centers': [0.20, 0.45, 0.70, 0.90], 'sigmas': [0.11, 0.15, 0.12, 0.10],
     'amplitudes': [1, 1, 1, 1], 'base_alpha': a, 'damping': d, 'cap': cap}
    for a in [.10, .20, .30, .40, .50] for d in [0.0, 6.0] for cap in [.03, .06, .10]
]
distributions = presets + [random_distribution() for _ in range(N_DISTRIBUTIONS)]

q3_times = train.loc[q3_mask, TIME_COL].reset_index(drop=True)
midpoint = q3_times.iloc[len(q3_times) // 2]
first = (q3_times < midpoint).to_numpy(); second = ~first
base_metrics = {
    'Q3': official_metrics(q3_y, previous_q3), 'Q4': official_metrics(q4_y, previous_q4),
    'Q3A': official_metrics(q3_y[first], previous_q3[first]),
    'Q3B': official_metrics(q3_y[second], previous_q3[second]),
}

q3_candidates = []
for number, params in enumerate(distributions):
    pred = gaussian_moe_predict(previous_q3, expert_predictions['Q3'], params)
    metric = official_metrics(q3_y, pred)
    metric_a = official_metrics(q3_y[first], pred[first])
    metric_b = official_metrics(q3_y[second], pred[second])
    stable = bool(
        metric['Score'] >= base_metrics['Q3']['Score'] - 0.0005
        and metric['NMAE'] <= base_metrics['Q3']['NMAE'] + 0.0005
        and metric['FICR'] >= base_metrics['Q3']['FICR'] - 0.001
        and metric_a['Score'] >= base_metrics['Q3A']['Score'] - 0.003
        and metric_b['Score'] >= base_metrics['Q3B']['Score'] - 0.003
    )
    if stable:
        q3_candidates.append({'id': number, 'params': params, 'Q3': metric,
                              'Q3A': metric_a, 'Q3B': metric_b, 'Q3_prediction': pred})

q3_candidates = sorted(
    q3_candidates,
    key=lambda row: (row['Q3']['Score'] + .25 * min(
        row['Q3A']['Score'] - base_metrics['Q3A']['Score'],
        row['Q3B']['Score'] - base_metrics['Q3B']['Score']), -row['Q3']['NMAE']),
    reverse=True,
)[:TOP_Q3_CANDIDATES]
print('전체 분포 후보:', len(distributions))
print('Q3 안정성 상위 확인 후보:', len(q3_candidates))


# ---- 원본 노트북 코드 셀 11 ----
def monthly_wins(pred3, pred4):
    pred = np.concatenate([pred3, pred4]); actual = np.concatenate([q3_y, q4_y])
    old = np.concatenate([previous_q3, previous_q4])
    times = pd.concat([train.loc[q3_mask, TIME_COL], train.loc[q4_mask, TIME_COL]], ignore_index=True)
    details, wins = [], 0
    for month in range(7, 13):
        use = times.dt.month.to_numpy() == month
        gain = official_metrics(actual[use], pred[use])['Score'] - official_metrics(actual[use], old[use])['Score']
        details.append((month, float(gain))); wins += int(gain > 0)
    return wins, details


old_q4_inside = np.abs(previous_q4 - q4_y) <= .08
evaluated = []
for row in q3_candidates:
    pred4 = gaussian_moe_predict(previous_q4, expert_predictions['Q4'], row['params'])
    metric4 = official_metrics(q4_y, pred4)
    new_inside = np.abs(pred4 - q4_y) <= .08; use = q4_y >= ACTIVE_CF
    rescued = int((use & ~old_q4_inside & new_inside).sum())
    lost = int((use & old_q4_inside & ~new_inside).sum())
    wins, monthly = monthly_wins(row['Q3_prediction'], pred4)
    mean_score = .5 * (row['Q3']['Score'] + metric4['Score'])
    old_mean = .5 * (base_metrics['Q3']['Score'] + base_metrics['Q4']['Score'])
    passed = bool(
        row['Q3']['Score'] >= base_metrics['Q3']['Score'] + .0002
        and metric4['Score'] >= base_metrics['Q4']['Score'] + .0002
        and mean_score >= old_mean + .0005
        and metric4['NMAE'] <= base_metrics['Q4']['NMAE'] + .0004
        and metric4['FICR'] >= base_metrics['Q4']['FICR'] - .0002
        and rescued >= lost and wins >= 4
    )
    row.update(Q4=metric4, Q4_prediction=pred4, rescued=rescued, lost=lost,
               monthly_wins=wins, monthly_gain=monthly, mean_score=mean_score, passed=passed)
    evaluated.append(row)

approved = [row for row in evaluated if row['passed']]
if approved:
    selected = max(approved, key=lambda row: (
        min(row['Q3']['Score'] - base_metrics['Q3']['Score'],
            row['Q4']['Score'] - base_metrics['Q4']['Score']),
        row['mean_score'], row['Q4']['FICR']))
    source = 'approved_gaussian_moe'
else:
    selected = None; source = 'previous_best_fallback'

comparisons = [
    ('previous_best', previous_q3, previous_q4),
    ('current_model', current_q3, current_q4),
    ('simple_segment_control', control_q3, control_q4),
]
if selected is not None:
    comparisons.append(('gaussian_moe', selected['Q3_prediction'], selected['Q4_prediction']))

summary_rows = []
for name, pred3, pred4 in comparisons:
    for split, actual, pred in [('Q3', q3_y, pred3), ('Q4', q4_y, pred4)]:
        summary_rows.append(dict(model=name, split=split, **official_metrics(actual, pred)))
validation_summary = pd.DataFrame(summary_rows)
display(validation_summary[['model', 'split', 'rows', 'MAE_kWh', 'NMAE', 'FICR', 'Score',
                            'within_6_rate', 'within_8_rate', 'signed_bias_cf']])

if selected is None:
    print('\nGaussian MoE 승인 후보 없음 → 기존 최고 G2 유지')
else:
    print('\nGaussian MoE 최종 선택')
    print('Q3/Q4 Score:', selected['Q3']['Score'], selected['Q4']['Score'])
    print('구조/이탈:', selected['rescued'], selected['lost'], '월별 승리:', selected['monthly_wins'])
    print('분포:', json.dumps(selected['params'], ensure_ascii=False, indent=2))


# ---- 원본 노트북 코드 셀 13 ----
segment_rows = []
for name, pred3, pred4 in comparisons:
    segment_rows.extend(segment_metrics(q3_y, pred3, name, 'Q3'))
    segment_rows.extend(segment_metrics(q4_y, pred4, name, 'Q4'))
segment_comparison = pd.DataFrame(segment_rows)
display(segment_comparison[['model', 'split', 'segment', 'rows', 'NMAE', 'FICR', 'Score',
                            'within_6_rate', 'within_8_rate', 'signed_bias_cf']])

top_rows = []
for row in sorted(evaluated, key=lambda value: value['mean_score'], reverse=True)[:50]:
    params = row['params']
    top_rows.append({
        'id': row['id'], 'passed': row['passed'], 'Q3_Score': row['Q3']['Score'],
        'Q4_Score': row['Q4']['Score'], 'Q4_NMAE': row['Q4']['NMAE'],
        'Q4_FICR': row['Q4']['FICR'], 'mean_score': row['mean_score'],
        'monthly_wins': row['monthly_wins'], 'rescued': row['rescued'],
        'lost': row['lost'], 'net_rescued': row['rescued'] - row['lost'],
        'base_alpha': params['base_alpha'], 'damping': params['damping'], 'cap': params['cap'],
        **{f'center_{i+1}': value for i, value in enumerate(params['centers'])},
        **{f'sigma_{i+1}': value for i, value in enumerate(params['sigmas'])},
        **{f'amplitude_{i+1}': value for i, value in enumerate(params['amplitudes'])},
    })
candidate_comparison = pd.DataFrame(top_rows)
display(candidate_comparison.head(15))

plot_params = selected['params'] if selected is not None else presets[0]
grid = np.linspace(.10, 1.00, 500)
dummy_experts = np.zeros((len(grid), 4))
_, gate_weights, strength, _ = gaussian_moe_predict(grid, dummy_experts, plot_params, True)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for index, name in enumerate(EXPERT_NAMES):
    axes[0].plot(grid, gate_weights[:, index], label=name)
axes[0].set_title('CF별 전문가 Gaussian 가중치')
axes[0].set_xlabel('기본모델 예측 CF'); axes[0].set_ylabel('정규화 가중치'); axes[0].legend()
axes[1].plot(grid, strength)
axes[1].set_title('기본모델에서 전문가 쪽으로 이동하는 강도')
axes[1].set_xlabel('기본모델 예측 CF'); axes[1].set_ylabel('혼합 강도')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gaussian_gate_weights.png', dpi=150, bbox_inches='tight')
plt.show()


# ---- 원본 노트북 코드 셀 15 ----
final_submission = previous_sub.copy()

if selected is not None:
    full_indices = np.flatnonzero(active)
    full_expert_test = []
    for name, center, sigma in zip(EXPERT_NAMES, TRAIN_CENTERS, TRAIN_SIGMAS):
        print('전체 재학습:', name)
        weight = expert_sample_weight(center, sigma, active)
        model = make_expert()
        model.fit(X.iloc[full_indices], y[full_indices], sample_weight=weight[full_indices])
        raw_test = np.clip(model.predict(X_test), 0.0, PRED_MAX_CF)
        by_time = pd.Series(raw_test, index=pd.DatetimeIndex(test[TIME_COL])).groupby(level=0).mean()
        aligned = previous_sub[TIME_COL].map(by_time).to_numpy(dtype=float)
        if np.isnan(aligned).any():
            raise ValueError('Test 전문가 예측과 submission 시간 정렬 실패')
        full_expert_test.append(aligned)
        del model; gc.collect()
    full_expert_test = np.column_stack(full_expert_test)
    base_test = previous_sub[TARGET_COL].to_numpy(dtype=float) / CAPACITY_KWH
    final_cf = gaussian_moe_predict(base_test, full_expert_test, selected['params'])
    final_submission[TARGET_COL] = np.clip(final_cf * CAPACITY_KWH, 0.0, CAPACITY_KWH * PRED_MAX_CF)

final_submission.to_csv(OUTPUT_DIR / 'submission_g2_gaussian_moe.csv', index=False, encoding='utf-8-sig')
validation_summary.to_csv(OUTPUT_DIR / 'validation_summary.csv', index=False, encoding='utf-8-sig')
segment_comparison.to_csv(OUTPUT_DIR / 'segment_comparison.csv', index=False, encoding='utf-8-sig')
candidate_comparison.to_csv(OUTPUT_DIR / 'candidate_comparison.csv', index=False, encoding='utf-8-sig')

configuration = {'source': source, 'seed': SEED, 'fast_mode': FAST_MODE}
if selected is not None:
    configuration.update({
        'params': selected['params'], 'Q3': selected['Q3'], 'Q4': selected['Q4'],
        'rescued': selected['rescued'], 'lost': selected['lost'],
        'monthly_wins': selected['monthly_wins'], 'monthly_gain': selected['monthly_gain'],
    })
with open(OUTPUT_DIR / 'selected_configuration.json', 'w', encoding='utf-8') as file:
    json.dump(configuration, file, ensure_ascii=False, indent=2)

print(f'\n전체 실행시간: {(time.time() - started_at) / 60:.1f}분')
print('저장 완료:', OUTPUT_DIR)
print('제출 파일:', OUTPUT_DIR / 'submission_g2_gaussian_moe.csv')
display(final_submission.head())



# =============================================================================
# 8. Group별 최종 제출 조립
# =============================================================================

# ---- 원본 노트북 코드 셀 1 ----
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = REPO_ROOT
PREVIOUS_DIR = ROOT / 'wind_baram_v2_output'
SELECTOR_DIR = ROOT / 'wind_optimal_group_selector_output'
GAUSSIAN_DIR = ROOT / 'wind_g2_gaussian_moe_output'
OUTPUT_DIR = ROOT / 'model_2_output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREVIOUS_SUB_PATH = PREVIOUS_DIR / 'submission.csv'
SELECTOR_SUB_PATH = SELECTOR_DIR / 'submission_optimal_selector.csv'
GAUSSIAN_SUB_PATH = GAUSSIAN_DIR / 'submission_g2_gaussian_moe.csv'
SELECTOR_VALID_PATH = SELECTOR_DIR / 'validation_summary.csv'
GAUSSIAN_VALID_PATH = GAUSSIAN_DIR / 'validation_summary.csv'
SELECTOR_CONFIG_PATH = SELECTOR_DIR / 'selected_configuration.json'
GAUSSIAN_CONFIG_PATH = GAUSSIAN_DIR / 'selected_configuration.json'

for path in [PREVIOUS_SUB_PATH, SELECTOR_SUB_PATH, GAUSSIAN_SUB_PATH,
             SELECTOR_VALID_PATH, GAUSSIAN_VALID_PATH,
             SELECTOR_CONFIG_PATH, GAUSSIAN_CONFIG_PATH]:
    if not path.exists():
        raise FileNotFoundError(path)

TARGETS = {1: 'kpx_group_1', 2: 'kpx_group_2', 3: 'kpx_group_3'}
print('최종 출력 폴더:', OUTPUT_DIR)


# ---- 원본 노트북 코드 셀 3 ----
selector_valid = pd.read_csv(SELECTOR_VALID_PATH)
gaussian_valid = pd.read_csv(GAUSSIAN_VALID_PATH)
with open(SELECTOR_CONFIG_PATH, encoding='utf-8') as file:
    selector_config = json.load(file)
with open(GAUSSIAN_CONFIG_PATH, encoding='utf-8') as file:
    gaussian_config = json.load(file)


def selector_metric(group, model, split):
    rows = selector_valid.loc[
        (pd.to_numeric(selector_valid['group'], errors='coerce') == group)
        & (selector_valid['model'] == model)
        & (selector_valid['split'] == split)
    ]
    if len(rows) != 1:
        raise ValueError(f'Selector metric 중복/누락: G{group} {model} {split}')
    return rows.iloc[0].to_dict()


def gaussian_metric(model, split):
    rows = gaussian_valid.loc[(gaussian_valid['model'] == model) & (gaussian_valid['split'] == split)]
    if len(rows) != 1:
        raise ValueError(f'Gaussian metric 중복/누락: {model} {split}')
    return rows.iloc[0].to_dict()


candidate_metrics = {
    1: {
        'source': 'optimal_selector_group1',
        'previous_Q3': selector_metric(1, 'previous_best', 'Q3'),
        'previous_Q4': selector_metric(1, 'previous_best', 'Q4'),
        'candidate_Q3': selector_metric(1, 'selected', 'Q3'),
        'candidate_Q4': selector_metric(1, 'selected', 'Q4'),
        'details': selector_config['1'],
    },
    2: {
        'source': 'symmetric_gaussian_moe',
        'previous_Q3': gaussian_metric('previous_best', 'Q3'),
        'previous_Q4': gaussian_metric('previous_best', 'Q4'),
        'candidate_Q3': gaussian_metric('gaussian_moe', 'Q3'),
        'candidate_Q4': gaussian_metric('gaussian_moe', 'Q4'),
        'details': gaussian_config,
    },
    3: {
        'source': 'optimal_selector_group3',
        'previous_Q3': selector_metric(3, 'previous_best', 'Q3'),
        'previous_Q4': selector_metric(3, 'previous_best', 'Q4'),
        'candidate_Q3': selector_metric(3, 'selected', 'Q3'),
        'candidate_Q4': selector_metric(3, 'selected', 'Q4'),
        'details': selector_config['3'],
    },
}

selection_rows = []
for group, info in candidate_metrics.items():
    old3, old4 = info['previous_Q3'], info['previous_Q4']
    new3, new4 = info['candidate_Q3'], info['candidate_Q4']
    q3_gain = float(new3['Score'] - old3['Score'])
    q4_gain = float(new4['Score'] - old4['Score'])
    mean_gain = 0.5 * (q3_gain + q4_gain)
    q4_nmae_change = float(new4['NMAE'] - old4['NMAE'])
    q4_ficr_gain = float(new4['FICR'] - old4['FICR'])
    # 공식 Score가 양쪽 분기에서 개선되고 FICR가 0.003 이상 오르면,
    # NMAE의 작은 희생(최대 0.001)은 정산점수 개선으로 보상된 것으로 인정합니다.
    error_tradeoff_safe = bool(
        q4_nmae_change <= 0.0005
        or (q4_ficr_gain >= 0.003 and q4_nmae_change <= 0.001)
    )
    approved = bool(
        q3_gain >= 0.0002 and q4_gain >= 0.0002 and mean_gain >= 0.0005
        and error_tradeoff_safe
        and new4['FICR'] >= old4['FICR'] - 0.0005
    )
    selection_rows.append({
        'group': group, 'candidate_source': info['source'], 'approved': approved,
        'Q3_old_Score': old3['Score'], 'Q3_candidate_Score': new3['Score'], 'Q3_gain': q3_gain,
        'Q4_old_Score': old4['Score'], 'Q4_candidate_Score': new4['Score'], 'Q4_gain': q4_gain,
        'mean_gain': mean_gain, 'Q4_old_NMAE': old4['NMAE'], 'Q4_candidate_NMAE': new4['NMAE'],
        'Q4_NMAE_change': q4_nmae_change, 'Q4_old_FICR': old4['FICR'],
        'Q4_candidate_FICR': new4['FICR'], 'Q4_FICR_gain': q4_ficr_gain,
        'error_tradeoff_safe': error_tradeoff_safe,
    })

selection_summary = pd.DataFrame(selection_rows)
display(selection_summary)


# ---- 원본 노트북 코드 셀 5 ----
macro_rows = []
for split in ['Q3', 'Q4']:
    for model in ['previous_best', 'final_selected']:
        metrics = []
        for group, info in candidate_metrics.items():
            approved = bool(selection_summary.loc[selection_summary['group'] == group, 'approved'].iloc[0])
            key = f'candidate_{split}' if model == 'final_selected' and approved else f'previous_{split}'
            metrics.append(info[key])
        macro_nmae = float(np.mean([row['NMAE'] for row in metrics]))
        macro_ficr = float(np.mean([row['FICR'] for row in metrics]))
        macro_rows.append({
            'model': model, 'split': split, 'macro_NMAE': macro_nmae,
            'macro_1_minus_NMAE': 1.0 - macro_nmae,
            'macro_FICR': macro_ficr,
            'macro_Score': 0.5 * (1.0 - macro_nmae + macro_ficr),
        })
macro_validation_summary = pd.DataFrame(macro_rows)
display(macro_validation_summary)


# ---- 원본 노트북 코드 셀 7 ----
previous_sub = pd.read_csv(PREVIOUS_SUB_PATH)
selector_sub = pd.read_csv(SELECTOR_SUB_PATH)
gaussian_sub = pd.read_csv(GAUSSIAN_SUB_PATH)


def assert_aligned(reference, candidate, name):
    keys = ['forecast_id', 'forecast_kst_dtm']
    if not reference[keys].astype(str).equals(candidate[keys].astype(str)):
        raise ValueError(f'{name} submission 행 순서 또는 시간이 다릅니다.')


assert_aligned(previous_sub, selector_sub, 'optimal selector')
assert_aligned(previous_sub, gaussian_sub, 'Gaussian G2')

final_submission = previous_sub.copy()
source_frames = {1: selector_sub, 2: gaussian_sub, 3: selector_sub}
for group, target in TARGETS.items():
    approved = bool(selection_summary.loc[selection_summary['group'] == group, 'approved'].iloc[0])
    if approved:
        final_submission[target] = source_frames[group][target].to_numpy(dtype=float)

# 기존 최고 대비 실제 제출값 변경 규모를 선택표에 추가합니다.
for index, row in selection_summary.iterrows():
    group = int(row['group']); target = TARGETS[group]
    difference = final_submission[target].to_numpy(dtype=float) - previous_sub[target].to_numpy(dtype=float)
    selection_summary.loc[index, 'changed_rows_pct'] = float((np.abs(difference) > 1e-9).mean() * 100)
    selection_summary.loc[index, 'mean_signed_change_kWh'] = float(difference.mean())
    selection_summary.loc[index, 'mean_abs_change_kWh'] = float(np.abs(difference).mean())
    selection_summary.loc[index, 'p90_abs_change_kWh'] = float(np.quantile(np.abs(difference), .90))

display(selection_summary)
display(final_submission.head())


# ---- 원본 노트북 코드 셀 9 ----
FINAL_SUB_PATH = OUTPUT_DIR / 'submission_model_2.csv'
final_submission.to_csv(FINAL_SUB_PATH, index=False, encoding='utf-8-sig')
selection_summary.to_csv(OUTPUT_DIR / 'selection_summary.csv', index=False, encoding='utf-8-sig')
macro_validation_summary.to_csv(OUTPUT_DIR / 'macro_validation_summary.csv', index=False, encoding='utf-8-sig')

serializable = {}
for group, info in candidate_metrics.items():
    approved = bool(selection_summary.loc[selection_summary['group'] == group, 'approved'].iloc[0])
    serializable[str(group)] = {
        'approved': approved,
        'source': info['source'] if approved else 'previous_best_fallback',
        'details': info['details'] if approved else {},
    }
with open(OUTPUT_DIR / 'selected_configuration.json', 'w', encoding='utf-8') as file:
    json.dump(serializable, file, ensure_ascii=False, indent=2)

print('저장 완료:', OUTPUT_DIR)
print('최종 제출:', FINAL_SUB_PATH)




# =============================================================================
# 9. 최종 결과 확인
# =============================================================================

FINAL_SUBMISSION_PATH = (
    REPO_ROOT
    / "model_2_output"
    / "submission_model_2.csv"
)
if not FINAL_SUBMISSION_PATH.exists():
    raise FileNotFoundError(
        f"최종 제출 파일이 생성되지 않았습니다: {FINAL_SUBMISSION_PATH}"
    )

print("\n" + "=" * 80)
print("모델 2 전체 실행 완료")
print("최종 제출 파일:", FINAL_SUBMISSION_PATH)
print(f"전체 실행시간: {(time.perf_counter() - TOTAL_STARTED) / 60:.1f}분")
