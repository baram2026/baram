import gc
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# SCADA 0·저출력 원인 분류 및 장기 고장·정비 결측 처리
#
# Colab /content에 필요한 파일
# 1. scada_vestas_train.csv
# 2. scada_unison_train.csv
# 3. ldaps_train.csv
#
# 생성 파일
# - /content/preprocessed/scada_vestas_train_cleaned.csv
# - /content/preprocessed/scada_unison_train_cleaned.csv
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
    vestas_diagnostics
):
    """
    VESTAS 터빈별 10분 장기연속정지 결과를
    Group 1·2의 시간별 장기정지 터빈 수로 변환합니다.

    시간 대응:
    - 01:10, 01:20, ..., 02:00
      → 02:00 발전량 Label

    반환 컬럼:
    - group1_outage_turbine_count
    - group1_available_turbine_count
    - group1_fault_maintenance_flag
    - group2_outage_turbine_count
    - group2_available_turbine_count
    - group2_fault_maintenance_flag

    이 요약값은 Group 1·2의 장기정지 행 제외에만 사용하며,
    최종 모델 입력 피처로는 남기지 않습니다.
    """

    required_cols = [
        "kst_dtm",
        "turbine_id",
        "kpx_group",
        "is_long_outage"
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in vestas_diagnostics.columns
    ]

    if missing_cols:
        raise KeyError(
            "시간별 장기정지 요약 생성에 "
            "필요한 컬럼이 없습니다: "
            f"{missing_cols}"
        )

    temp = vestas_diagnostics[
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

    # VESTAS Group 1·2만 장기정지 행 제외 판단에 사용
    temp = temp[
        temp["kpx_group"].isin(
            [1, 2]
        )
    ].copy()

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
            1: "group1_outage_turbine_count",
            2: "group2_outage_turbine_count"
        }
    )

    total_turbines_by_group = {
        1: 6,
        2: 6
    }

    for group_id, total_turbines in (
        total_turbines_by_group.items()
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

    output_cols = [
        "forecast_kst_dtm",
        "group1_outage_turbine_count",
        "group1_available_turbine_count",
        "group1_fault_maintenance_flag",
        "group2_outage_turbine_count",
        "group2_available_turbine_count",
        "group2_fault_maintenance_flag"
    ]

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

    for group_id in [1, 2]:
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
    Group 1·2에서 해당 그룹의 장기연속정지 터빈이
    한 대라도 존재하는 시간의 학습 행을 제외합니다.

    중요:
    - Group 1 장기정지는 Group 1 행에만 적용
    - Group 2 장기정지는 Group 2 행에만 적용
    - 동일 시각의 다른 그룹 행은 유지
    - Group 3에는 VESTAS 장기정지 정보를 적용하지 않음
    - SCADA 요약 열은 필터링 직후 삭제
    """

    result = group_df.copy()

    if group_id == 3:
        return result, 0

    if group_id not in {1, 2}:
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
        | (result[outage_col] > 6)
    )

    if invalid_mask.any():
        raise ValueError(
            f"Group {group_id} 장기정지 터빈 수가 "
            "0~6 범위를 벗어났습니다."
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
    - Group 1·2는 장기연속정지 터빈이 한 대라도 있는
      해당 그룹·시간 행을 삭제
    - Group 3는 VESTAS 장기정지 정보를 적용하지 않음
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
    # Group 1·2 장기연속정지 행을 그룹별로 제외
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

    if group_id in {1, 2}:
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
    - Group 1·2의 장기연속정지 행만 그룹별로 제외
    - Group 3에는 VESTAS 장기정지 정보를 적용하지 않음
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
                "Train Group 1·2 장기정지 행 제외에 "
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
    input_dir="/content",
    save_dir="/content/preprocessed",
    process_test_if_available=True
):
    """
    하나의 함수에서 다음을 순서대로 수행합니다.

    1. SCADA 저출력 원인 분류
    2. 24시간 이상 장기연속정지 판정
    3. VESTAS Group 1·2 시간별 장기정지 터빈 수 계산
    4. SCADA cleaned CSV와 장기정지 요약을 별도 저장
    5. Train 기상변수 전처리
    6. Group 1·2에서 장기정지 터빈이 한 대라도 있는
       해당 그룹·시간 학습 행 삭제
    7. train_labels 값은 비례 보정하지 않고 원값 사용
    8. Group 3 원래 Target 결측 제외
    9. Test 기상변수 전처리
    10. SCADA 열이 없는 그룹별 최종 CSV 저장

    주의:
    - Group 1 장기정지는 Group 1 Train에서만 삭제합니다.
    - Group 2 장기정지는 Group 2 Train에서만 삭제합니다.
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

    required_train_paths = [
        input_dir / "ldaps_train.csv",
        input_dir / "gfs_train.csv",
        input_dir / "train_labels.csv",
        input_dir / "scada_vestas_train.csv",
        input_dir / "scada_unison_train.csv"
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
            input_dir
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
        input_dir=input_dir,
        save_dir=save_dir
    )

    hourly_outage_summary = (
        build_hourly_outage_summary(
            vestas_diagnostics=(
                vestas_diagnostics
            )
        )
    )

    # UNISON은 cleaned SCADA 파일만 별도 생성하며,
    # Group 1·2 행 삭제에는 사용하지 않음
    (
        unison_cleaned,
        unison_diagnostics
    ) = process_manufacturer(
        manufacturer="unison",
        ldaps_weather=ldaps_icing_weather,
        input_dir=input_dir,
        save_dir=save_dir
    )

    print(
        "\n[시간별 장기정지 요약]"
    )

    print(
        hourly_outage_summary[
            [
                "group1_outage_turbine_count",
                "group2_outage_turbine_count"
            ]
        ].describe()
    )

    outage_summary_path = (
        save_dir
        / "scada_vestas_hourly_outage_summary.csv"
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
        unison_cleaned,
        unison_diagnostics
    )

    gc.collect()

    # =====================================================
    # 2단계: 기상변수 fallback 계산
    # =====================================================
    print("\n" + "=" * 70)
    print("2단계: Train 기상 fallback 계산")
    print("=" * 70)

    ldaps_train_raw = read_csv_utf8(
        input_dir
        / "ldaps_train.csv"
    )

    gfs_train_raw = read_csv_utf8(
        input_dir
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
            input_dir=input_dir,
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
        input_dir
        / "ldaps_test.csv"
    )

    gfs_test_path = (
        input_dir
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
                    input_dir=input_dir,
                    save_dir=save_dir,
                    fallback_values=(
                        fallback_values
                    ),
                    hourly_outage_summary=None
                )
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
# 7. 최종 실행부
# =========================================================
if __name__ == "__main__":
    results = run_preprocessing(
        input_dir="/content",
        save_dir="/content/preprocessed",
        process_test_if_available=True
    )

    # 최종 Train: 기상 피처 + 해당 그룹 Target만 포함
    train_group1 = (
        results["train"]["group1"]
    )

    train_group2 = (
        results["train"]["group2"]
    )

    train_group3 = (
        results["train"]["group3"]
    )

    # 진단용 시간별 장기정지 요약
    # 최종 그룹별 모델 파일에는 포함되지 않음
    hourly_outage_summary = (
        results[
            "hourly_outage_summary"
        ]
    )

    if "test" in results:
        test_group1 = (
            results["test"]["group1"]
        )

        test_group2 = (
            results["test"]["group2"]
        )

        test_group3 = (
            results["test"]["group3"]
        )
