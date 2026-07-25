# =========================================================
# SCADA 0·저출력 원인 분류 및 고장/정비 후보 결측 처리
#
# Colab /content에 필요한 파일
# 1. scada_vestas_train.csv
# 2. scada_unison_train.csv
# 3. ldaps_train.csv
# =========================================================

from pathlib import Path
import gc

import numpy as np
import pandas as pd


# =========================================================
# 0. 실행 설정
# =========================================================
INPUT_DIR = Path("/content")
SAVE_DIR = Path("/content/preprocessed")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

EFFICIENCY_THRESHOLD = 0.02       # 정격 대비 2% 미만
ICING_TEMP_K = 273.15             # 0°C 미만
ICING_HUMIDITY = 90.0             # 상대습도 90% 이상
LOW_WIND_MS = 4.0                 # 터빈 SCADA 풍속 4m/s 미만
PEER_ACTIVE_RATIO = 0.50          # 타 터빈 절반 이상 정상발전
MIN_VALID_PEERS = 2               # 비교 가능한 타 터빈 최소 수
LONG_OUTAGE_HOURS = 72            # 72시간 이상 연속 저출력
SCADA_INTERVAL_MINUTES = 10
SAVE_DIAGNOSTICS = True


# =========================================================
# 1. 제조사·터빈 설정
# =========================================================
SCADA_CONFIG = {
    "vestas": {
        "file_name": "scada_vestas_train.csv",
        "n_turbines": 12,
        "rated_power_kw": 3600.0,
        "group_map": {
            **{i: 1 for i in range(1, 7)},
            **{i: 2 for i in range(7, 13)}
        }
    },
    "unison": {
        "file_name": "scada_unison_train.csv",
        "n_turbines": 5,
        "rated_power_kw": 4200.0,
        "group_map": {
            i: 3 for i in range(1, 6)
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

FAULT_STATUSES = {
    "개별정지_고장정비후보",
    "장기연속정지"
}


# =========================================================
# 2. 공통 파일 검증
# =========================================================
required_files = [
    INPUT_DIR / "scada_vestas_train.csv",
    INPUT_DIR / "scada_unison_train.csv",
    INPUT_DIR / "ldaps_train.csv"
]

missing_files = [
    path.name
    for path in required_files
    if not path.exists()
]

if missing_files:
    raise FileNotFoundError(
        "필수 파일이 없습니다: "
        + ", ".join(missing_files)
    )


# =========================================================
# 3. 착빙 판정용 LDAPS 기상 데이터 준비
# =========================================================
def prepare_ldaps_icing_weather(ldaps_path):
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
            sorted(set(TURBINE_GRID_MAP.values()))
        )
    ].copy()

    weather["temp_kelvin"] = pd.to_numeric(
        weather["heightAboveGround_2_t"],
        errors="coerce"
    )

    weather["humidity"] = pd.to_numeric(
        weather["heightAboveGround_2_r"],
        errors="coerce"
    ).clip(
        lower=0,
        upper=100
    )

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

    for col in ["temp_kelvin", "humidity"]:
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
        # Train의 동일 격자 중앙값으로 보완
        grid_median = (
            weather
            .groupby("grid_id")[col]
            .transform("median")
        )

        weather[col] = weather[col].fillna(
            grid_median
        )

        weather[col] = weather[col].fillna(
            weather[col].median()
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
            "LDAPS에 중복된 forecast_kst_dtm × grid_id가 있습니다."
        )

    return weather.sort_values(
        [
            "forecast_kst_dtm",
            "grid_id"
        ]
    ).reset_index(drop=True)


# =========================================================
# 4. SCADA Wide → Long
# =========================================================
def scada_wide_to_long(scada_df, manufacturer):
    """
    원본 터빈별 Wide SCADA를 시각 × 터빈 Long 형식으로 변환합니다.
    """

    config = SCADA_CONFIG[manufacturer]
    frames = []

    for turbine_no in range(
        1,
        config["n_turbines"] + 1
    ):
        turbine_id = (
            f"{manufacturer}_wtg{turbine_no:02d}"
        )

        power_col = (
            f"{turbine_id}_power_kw10m"
        )
        ws_col = f"{turbine_id}_ws"
        wd_col = f"{turbine_id}_wd"

        missing = [
            col
            for col in [power_col, ws_col, wd_col]
            if col not in scada_df.columns
        ]

        if missing:
            raise KeyError(
                f"{turbine_id} 컬럼 누락: {missing}"
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

        part["manufacturer"] = manufacturer
        part["turbine_no"] = turbine_no
        part["turbine_id"] = turbine_id
        part["kpx_group"] = (
            config["group_map"][turbine_no]
        )
        part["grid_id"] = np.int64(
            TURBINE_GRID_MAP[turbine_id]
        )
        part["source_power_col"] = power_col

        # 10분 단위 정격 발전량:
        # VESTAS 3600 × 10/60 = 600
        # UNISON 4200 × 10/60 = 700
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

    for col in ["power", "ws", "wd"]:
        long_df[col] = pd.to_numeric(
            long_df[col],
            errors="coerce"
        )

    return long_df


# =========================================================
# 5. 10분 SCADA에 시간별 LDAPS 기상값 결합
# =========================================================
def attach_weather(scada_long, ldaps_weather):
    """
    터빈별 최근접 격자의 가장 가까운 시간별 LDAPS 값을 결합합니다.

    SCADA가 10분, LDAPS가 1시간이므로 ±31분 범위에서
    가장 가까운 예보 시각을 연결합니다.

    merge_asof의 by 컬럼은 양쪽 dtype이 완전히 같아야 하므로
    grid_id를 모두 int64로 통일합니다.
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

    # MergeError 방지: 양쪽 병합 키 dtype 통일
    left["grid_id"] = pd.to_numeric(
        left["grid_id"],
        errors="raise"
    ).astype("int64")

    right["grid_id"] = pd.to_numeric(
        right["grid_id"],
        errors="raise"
    ).astype("int64")

    # merge_asof는 on 키가 전체적으로 오름차순이어야 함
    left = left.sort_values(
        [
            "kst_dtm",
            "grid_id"
        ]
    ).reset_index(drop=True)

    right = right.sort_values(
        [
            "forecast_kst_dtm",
            "grid_id"
        ]
    ).reset_index(drop=True)

    return pd.merge_asof(
        left=left,
        right=right,
        left_on="kst_dtm",
        right_on="forecast_kst_dtm",
        by="grid_id",
        direction="nearest",
        tolerance=pd.Timedelta("31min")
    )


# =========================================================
# 6. 터빈별 장기 연속 저출력 구간 탐지
# =========================================================
def add_long_outage_flag(df):
    """
    발전 효율 2% 미만이 72시간 이상 연속되는 구간의
    전체 행을 장기연속정지로 표시합니다.
    """

    result = df.sort_values(
        [
            "turbine_id",
            "kst_dtm"
        ]
    ).reset_index(drop=True)

    expected_gap = pd.Timedelta(
        minutes=SCADA_INTERVAL_MINUTES
    )

    previous_state = (
        result
        .groupby(
            "turbine_id",
            sort=False
        )["is_low_output"]
        .shift()
    )

    state_changed = (
        previous_state.isna()
        | result["is_low_output"].ne(
            previous_state
        )
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

    new_streak = state_changed | time_broken

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
        )["is_low_output"]
        .transform("size")
    )

    result["low_output_streak_hours"] = np.where(
        result["is_low_output"],
        (
            streak_size
            * SCADA_INTERVAL_MINUTES
            / 60.0
        ),
        0.0
    )

    result["is_long_outage"] = (
        result["is_low_output"]
        & (
            result["low_output_streak_hours"]
            >= LONG_OUTAGE_HOURS
        )
    )

    return result.drop(
        columns="_streak_id"
    )


# =========================================================
# 7. 저출력 원인 분류 및 발전량 정제
# =========================================================
def classify_and_clean(scada_weather):
    """
    판단 순서

    1. 정격 대비 발전 효율 2% 이상 → 정상발전
    2. 2% 미만 + 0°C 미만·습도 90% 이상 → 착빙
    3. 착빙 아님 + SCADA 풍속 4m/s 미만 → 저풍속
    4. 나머지에서 같은 KPX 그룹 타 터빈 절반 이상 정상발전
       → 개별정지_고장정비후보
    5. 72시간 이상 연속 저출력 → 장기연속정지

    결측 처리:
    - 개별정지_고장정비후보 → NaN
    - 장기연속정지 → NaN
    - 착빙·저풍속·판정불가 → 원래 값 유지
    """

    result = scada_weather.copy()

    result["power_efficiency"] = (
        result["power"]
        / result["rated_10min"]
    )

    result["is_low_output"] = (
        result["power"].notna()
        & (
            result["power_efficiency"]
            < EFFICIENCY_THRESHOLD
        )
    )

    result["is_active"] = (
        result["power"].notna()
        & (
            result["power_efficiency"]
            >= EFFICIENCY_THRESHOLD
        )
    )

    # 같은 시각·같은 KPX 그룹의 다른 터빈 비교
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
        - result["is_active"].astype("int8")
    )

    result["peer_valid_count"] = (
        all_valid
        - result["power"].notna().astype("int8")
    )

    result["peer_active_ratio"] = (
        result["peer_active_count"]
        / result["peer_valid_count"].replace(
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

    # 저풍속은 터빈별 동시점 SCADA 풍속으로 판정
    result["is_low_wind"] = (
        result["ws"] < LOW_WIND_MS
    )

    result["status"] = "정상발전"

    original_missing = result["power"].isna()
    candidate = result["is_low_output"]

    result.loc[
        original_missing,
        "status"
    ] = "원본결측"

    result.loc[
        candidate,
        "status"
    ] = "판정불가"

    # 1순위: 착빙
    icing_mask = (
        candidate
        & result["is_icing"]
    )

    result.loc[
        icing_mask,
        "status"
    ] = "착빙"

    # 2순위: 저풍속
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

    # 풍속 또는 기상값이 없으면 보수적으로 유지
    remaining = (
        remaining
        & ~result["is_low_wind"]
    )

    result.loc[
        remaining & result["ws"].isna(),
        "status"
    ] = "판정불가_풍속결측"

    weather_missing = (
        result["temp_kelvin"].isna()
        | result["humidity"].isna()
    )

    result.loc[
        remaining
        & result["ws"].notna()
        & weather_missing,
        "status"
    ] = "판정불가_기상결측"

    # 3순위: 타 터빈 비교
    fault_mask = (
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
        fault_mask,
        "status"
    ] = "개별정지_고장정비후보"

    # 4순위: 장기 연속 저출력
    result = add_long_outage_flag(result)

    result.loc[
        result["is_long_outage"],
        "status"
    ] = "장기연속정지"

    # 고장·정비 후보만 결측 처리
    result["power_clean"] = result["power"]

    result.loc[
        result["status"].isin(
            FAULT_STATUSES
        ),
        "power_clean"
    ] = np.nan

    return result


# =========================================================
# 8. 정제된 발전량을 원본 Wide 구조로 복원
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

    # 원본과 같은 컬럼 순서
    return cleaned[
        original_df.columns
    ]


# =========================================================
# 9. 제조사별 실행 함수
# =========================================================
def process_manufacturer(
    manufacturer,
    ldaps_weather
):
    config = SCADA_CONFIG[manufacturer]

    input_path = (
        INPUT_DIR
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
            f"{config['file_name']}에 kst_dtm이 없습니다."
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
        SAVE_DIR
        / f"scada_{manufacturer}_train_cleaned.csv"
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
        "power",
        "power_clean",
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
        "is_long_outage",
        "low_output_streak_hours",
        "status"
    ]

    if SAVE_DIAGNOSTICS:
        diagnostics_path = (
            SAVE_DIR
            / f"scada_{manufacturer}_zero_diagnostics.csv"
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
        diagnostics["power"].isna().sum()
    )

    cleaned_nan = int(
        diagnostics["power_clean"].isna().sum()
    )

    print(
        diagnostics["status"].value_counts(
            dropna=False
        )
    )

    print(
        "새로 NaN 처리된 발전량:",
        cleaned_nan - original_nan
    )

    del raw, long_df
    gc.collect()

    return cleaned, diagnostics


# =========================================================
# 10. 전체 실행
# =========================================================
print("=" * 70)
print("LDAPS 착빙 기상 데이터 준비")
print("=" * 70)

ldaps_weather = prepare_ldaps_icing_weather(
    INPUT_DIR / "ldaps_train.csv"
)

vestas_cleaned, vestas_diagnostics = (
    process_manufacturer(
        manufacturer="vestas",
        ldaps_weather=ldaps_weather
    )
)

unison_cleaned, unison_diagnostics = (
    process_manufacturer(
        manufacturer="unison",
        ldaps_weather=ldaps_weather
    )
)

print("\n" + "=" * 70)
print("SCADA 저출력 원인 분류 및 전처리 완료")
print("저장 위치:", SAVE_DIR)
print("=" * 70)

for path in sorted(
    SAVE_DIR.glob("scada_*")
):
    print("-", path.name)
