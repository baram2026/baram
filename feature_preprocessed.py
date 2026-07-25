import gc
from pathlib import Path

import numpy as np
import pandas as pd


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
                "heightAboveGround_50_50MUmax",
                "heightAboveGround_50_50MVmax",
                "ldaps_50m_max"
            ),
            (
                "heightAboveGround_50_50MUmin",
                "heightAboveGround_50_50MVmin",
                "ldaps_50m_min"
            )
        ],
        "wind_feature_prefixes": [
            "ldaps_10m",
            "ldaps_5m_blws",
            "ldaps_50m_max",
            "ldaps_50m_min",
            "ldaps_50m_mean"
        ],
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
            "data_available_kst_dtm"
        ]
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
            "upper_speed_col": "gfs_80m_wind_speed",
            "lower_height": 10.0,
            "upper_height": 80.0,
            "output_col": "gfs_shear_alpha_10m_80m"
        },
        "veer": {
            "lower_direction_col": "gfs_10m_wind_direction_deg",
            "upper_direction_col": "gfs_100m_wind_direction_deg",
            "lower_speed_col": "gfs_10m_wind_speed",
            "upper_speed_col": "gfs_100m_wind_speed",
            "output_col": "gfs_veer_100m_10m_deg"
        },
        "drop_cols": [
            "surface_0_lsm",
            "surface_0_h",
            "data_available_kst_dtm"
        ]
    }
}


GROUP_CAPACITY_KWH = {
    "kpx_group_1": 21600.0,
    "kpx_group_2": 21600.0,
    "kpx_group_3": 21000.0
}

TARGET_COLS = list(GROUP_CAPACITY_KWH)

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
    calm_threshold=1e-12
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
    epsilon=1e-6
):
    """
    alpha =
        ln((v_upper + epsilon) / (v_lower + epsilon))
        / ln(z_upper / z_lower)
    """

    upper_speed = pd.to_numeric(
        upper_speed,
        errors="coerce"
    )

    lower_speed = pd.to_numeric(
        lower_speed,
        errors="coerce"
    )

    return (
        np.log(
            (upper_speed + epsilon)
            / (lower_speed + epsilon)
        )
        / np.log(upper_height / lower_height)
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


def add_speed_direction_interactions(
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
# 3. 보간 및 Train fallback 통계치 함수
# =========================================================
def get_weather_value_columns(df, source):
    """
    시간·격자 키를 제외한 기상값 컬럼을 반환합니다.
    """

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


def calculate_train_fallbacks(
    train_df,
    source
):
    """
    보간 후에도 남는 결측치에 사용할 중앙값을
    Train에서만 계산합니다.
    """

    source = source.lower()
    config = WEATHER_CONFIG[source]
    temp = train_df.copy()

    for col in config["humidity_cols"]:
        if col in temp.columns:
            temp[col] = pd.to_numeric(
                temp[col],
                errors="coerce"
            ).clip(
                lower=0,
                upper=100
            )

    for col in config["nonnegative_cols"]:
        if col in temp.columns:
            temp[col] = pd.to_numeric(
                temp[col],
                errors="coerce"
            ).clip(lower=0)

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
    동일한 grid_id 및 data_available_kst_dtm 안에서
    forecast_kst_dtm 순으로 선형 보간합니다.

    보간 후에도 남는 결측치는 Train에서 계산한
    fallback_values로 채웁니다.
    """

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

    result["__original_order"] = np.arange(
        len(result)
    )

    value_cols = get_weather_value_columns(
        result,
        source
    )

    for col in value_cols:
        result[col] = pd.to_numeric(
            result[col],
            errors="coerce"
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

        result = result.sort_values(
            group_cols + ["forecast_kst_dtm"]
        )

        grouped = result.groupby(
            group_cols,
            sort=False,
            observed=True
        )

        for col in cols_with_missing:
            result[col] = grouped[col].transform(
                lambda series: series.interpolate(
                    method="linear",
                    limit_direction="both"
                )
            )

        if fallback_values is not None:
            for col in cols_with_missing:
                if col in fallback_values:
                    result[col] = result[col].fillna(
                        fallback_values[col]
                    )

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
    # 2단계: 물리적 범위 보정
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
    for u_col, v_col, prefix in config["wind_pairs"]:
        result = add_uv_wind_features(
            df=result,
            u_col=u_col,
            v_col=v_col,
            prefix=prefix
        )

    if source == "ldaps":
        result["ldaps_50m_mean_wind_speed"] = (
            result["ldaps_50m_max_wind_speed"]
            + result["ldaps_50m_min_wind_speed"]
        ) / 2.0

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
    # 3.5-3단계: LDAPS 50m 평균 풍향
    # -----------------------------------------------------
    if source == "ldaps":
        u50_mean = (
            pd.to_numeric(
                result[
                    "heightAboveGround_50_50MUmax"
                ],
                errors="coerce"
            )
            + pd.to_numeric(
                result[
                    "heightAboveGround_50_50MUmin"
                ],
                errors="coerce"
            )
        ) / 2.0

        v50_mean = (
            pd.to_numeric(
                result[
                    "heightAboveGround_50_50MVmax"
                ],
                errors="coerce"
            )
            + pd.to_numeric(
                result[
                    "heightAboveGround_50_50MVmin"
                ],
                errors="coerce"
            )
        ) / 2.0

        (
            _,
            mean_direction_deg,
            mean_dir_sin,
            mean_dir_cos
        ) = calculate_uv_direction(
            u=u50_mean.to_numpy(
                dtype="float64"
            ),
            v=v50_mean.to_numpy(
                dtype="float64"
            )
        )

        result[
            "ldaps_50m_mean_wind_direction_deg"
        ] = mean_direction_deg

        result[
            "ldaps_50m_mean_dir_sin"
        ] = mean_dir_sin

        result[
            "ldaps_50m_mean_dir_cos"
        ] = mean_dir_cos

    # -----------------------------------------------------
    # 3.5-4단계: Wind Shear Alpha
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
    # 3.5-5단계: 야간 플래그
    # -----------------------------------------------------
    shortwave_radiation = pd.to_numeric(
        result[config["shortwave_col"]],
        errors="coerce"
    )

    result[config["night_flag_col"]] = (
        shortwave_radiation <= 0
    ).astype("int8")

    # -----------------------------------------------------
    # 3.5-6단계: 풍속×풍향 상호작용
    # -----------------------------------------------------
    result = add_speed_direction_interactions(
        df=result,
        wind_prefixes=config[
            "wind_feature_prefixes"
        ]
    )

    # -----------------------------------------------------
    # 3.5-7단계: 야간 풍속 상호작용
    # -----------------------------------------------------
    result = add_night_speed_interactions(
        df=result,
        night_flag_col=config[
            "night_flag_col"
        ],
        wind_prefixes=config[
            "wind_feature_prefixes"
        ]
    )

    # -----------------------------------------------------
    # 3.5-8단계: Veer
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

    # 어느 한쪽이라도 무풍이면 풍향 변화는 0으로 처리
    calm_mask = (
        (
            result[
                veer_config["lower_speed_col"]
            ] <= 1e-12
        )
        | (
            result[
                veer_config["upper_speed_col"]
            ] <= 1e-12
        )
    )

    result[
        veer_config["output_col"]
    ] = veer.mask(calm_mask, 0.0)

    # -----------------------------------------------------
    # 4단계: 불필요 컬럼 삭제
    # -----------------------------------------------------
    result = result.drop(
        columns=config["drop_cols"]
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
}


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
        *GRID_METADATA_COLS,
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


def build_group_dataset(
    ldaps_processed,
    gfs_grid5,
    group_id,
    train_labels=None
):
    """
    그룹별 최종 데이터프레임을 만듭니다.

    Train:
        그룹 관련 LDAPS Wide + GFS grid 5 + 해당 그룹 Target

    Test:
        그룹 관련 LDAPS Wide + GFS grid 5
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

    # Test처럼 Label이 없는 경우 기상 피처만 반환
    if train_labels is None:
        return group_df

    target_col = GROUP_TARGET_COLS[group_id]

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

    # Target을 마지막 컬럼에 배치
    feature_cols = [
        col
        for col in final_group_df.columns
        if col != target_col
    ]

    return final_group_df[
        feature_cols + [target_col]
    ]


def process_grouped_dataset(
    split,
    input_dir,
    save_dir,
    fallback_values,
    label_upper_ratio=1.05
):
    """
    Train/Test를 동일한 구조로 처리하여
    Group 1, 2, 3 데이터프레임을 각각 생성하고 저장합니다.
    """

    split = split.lower()

    ldaps_processed, gfs_processed = preprocess_weather_split(
        split=split,
        input_dir=input_dir,
        fallback_values=fallback_values
    )

    # GFS grid 5는 모든 그룹에서 공통 사용
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

    elif split != "test":
        raise ValueError(
            "split은 'train' 또는 'test'여야 합니다."
        )

    group_results = {}

    for group_id in [1, 2, 3]:
        group_name = f"group{group_id}"

        group_df = build_group_dataset(
            ldaps_processed=ldaps_processed,
            gfs_grid5=gfs_grid5,
            group_id=group_id,
            train_labels=train_labels
        )

        output_path = (
            save_dir
            / f"{split}_{group_name}_preprocessed.csv"
        )

        group_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
            date_format="%Y-%m-%d %H:%M:%S"
        )

        group_results[group_name] = group_df

    del ldaps_processed, gfs_processed, gfs_grid5
    gc.collect()

    return group_results


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
# 8. 전체 실행 함수
# =========================================================
def run_preprocessing(
    input_dir="/content",
    save_dir="/content/preprocessed",
    process_test_if_available=True
):
    """
    필수 Train 파일:
    - ldaps_train.csv
    - gfs_train.csv
    - train_labels.csv

    생성 파일:
    - train_group1_preprocessed.csv
    - train_group2_preprocessed.csv
    - train_group3_preprocessed.csv

    Test 파일 두 개가 모두 존재하면 동일한 함수로:
    - test_group1_preprocessed.csv
    - test_group2_preprocessed.csv
    - test_group3_preprocessed.csv

    도 함께 생성합니다.
    """

    input_dir = Path(input_dir)
    save_dir = Path(save_dir)

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_paths = [
        input_dir / "ldaps_train.csv",
        input_dir / "gfs_train.csv",
        input_dir / "train_labels.csv"
    ]

    check_required_files(
        train_paths,
        "필수 TRAIN"
    )

    # Train에서만 fallback 중앙값 계산
    ldaps_train_raw = read_csv_utf8(
        input_dir / "ldaps_train.csv"
    )

    gfs_train_raw = read_csv_utf8(
        input_dir / "gfs_train.csv"
    )

    fallback_values = {
        "ldaps": calculate_train_fallbacks(
            train_df=ldaps_train_raw,
            source="ldaps"
        ),
        "gfs": calculate_train_fallbacks(
            train_df=gfs_train_raw,
            source="gfs"
        )
    }

    del ldaps_train_raw, gfs_train_raw
    gc.collect()

    results = {}

    results["train"] = process_grouped_dataset(
        split="train",
        input_dir=input_dir,
        save_dir=save_dir,
        fallback_values=fallback_values
    )

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
            results["test"] = process_grouped_dataset(
                split="test",
                input_dir=input_dir,
                save_dir=save_dir,
                fallback_values=fallback_values
            )

        elif any(test_file_exists):
            raise FileNotFoundError(
                "Test 파일은 LDAPS와 GFS 두 개가 "
                "모두 필요합니다. 현재 하나만 존재합니다."
            )

    print("=" * 70)
    print("그룹별 전처리 및 저장 완료")
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
        print("-", file_path.name)

    return results


# =========================================================
# 9. 실행
# =========================================================
if __name__ == "__main__":
    results = run_preprocessing(
        input_dir="/content",
        save_dir="/content/preprocessed",
        process_test_if_available=True
    )

    train_group1 = results["train"]["group1"]
    train_group2 = results["train"]["group2"]
    train_group3 = results["train"]["group3"]

    if "test" in results:
        test_group1 = results["test"]["group1"]
        test_group2 = results["test"]["group2"]
        test_group3 = results["test"]["group3"]
