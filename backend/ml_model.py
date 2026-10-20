import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_FOLDER = "models"

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "peakdemand_model.joblib"
)

FEATURE_COLUMNS = [
    "Hour",
    "Day",
    "Month",
    "Season",
    "Holiday",
    "Temperature"
]

TARGET_COLUMN = "Demand_MW"


# ============================================================
# PEAK LIMIT
# ============================================================

PEAK_LIMIT = 500.0


# ============================================================
# SEASON
# ============================================================

def get_season(month):

    month = int(month)

    if month in [12, 1, 2]:
        return "Winter"

    if month in [3, 4, 5]:
        return "Summer"

    if month in [6, 7, 8, 9]:
        return "Monsoon"

    return "Autumn"


# ============================================================
# CLEAN DATASET
# ============================================================

def clean_dataset(filepath):

    df = pd.read_csv(filepath)

    required_columns = [
        "Date_Time",
        "Demand_MW",
        "Temperature"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column '{column}' is missing from the CSV file."
            )

    df["Date_Time"] = pd.to_datetime(
        df["Date_Time"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date_Time"]
    )

    df["Hour"] = df["Date_Time"].dt.hour
    df["Day"] = df["Date_Time"].dt.day
    df["Month"] = df["Date_Time"].dt.month

    if "Season" not in df.columns:

        df["Season"] = df["Month"].apply(
            get_season
        )

    else:

        df["Season"] = df["Season"].fillna(
            df["Month"].apply(get_season)
        )

        df["Season"] = df["Season"].astype(str)

    if "Holiday" not in df.columns:
        df["Holiday"] = 0

    df["Holiday"] = pd.to_numeric(
        df["Holiday"],
        errors="coerce"
    ).fillna(0)

    df["Temperature"] = pd.to_numeric(
        df["Temperature"],
        errors="coerce"
    )

    df["Demand_MW"] = pd.to_numeric(
        df["Demand_MW"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Temperature",
            "Demand_MW"
        ]
    )

    df = df.drop_duplicates(
        subset=["Date_Time"]
    )

    df = df.sort_values(
        "Date_Time"
    ).reset_index(drop=True)

    return df


# ============================================================
# ONE-HOT ENCODER
# ============================================================

def create_one_hot_encoder():

    try:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    except TypeError:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


# ============================================================
# REGRESSION MODEL
# Predicts demand for a particular hour.
# ============================================================

def create_demand_model():

    categorical_features = [
        "Season"
    ]

    numeric_features = [
        "Hour",
        "Day",
        "Month",
        "Holiday",
        "Temperature"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "season",
                create_one_hot_encoder(),
                categorical_features
            )
        ],
        remainder="passthrough"
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=None,
        min_samples_split=2,
        n_jobs=-1
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )


# ============================================================
# PEAK-HOUR MODEL
#
# IMPORTANT:
# Peak time is NOT calculated with np.argmax(predicted demand).
#
# Instead, the historical CSV is converted into one record per
# day. The actual highest-demand hour of each historical day is
# used as the target of a separate ML classifier.
#
# Features:
#   Day
#   Month
#   Season
#   Holiday
#   Temperature
#
# Target:
#   Actual historical peak hour
#
# Therefore:
#   input features -> peak-hour model -> predicted peak time
# ============================================================

PEAK_FEATURE_COLUMNS = [
    "Month",
    "Season",
    "Holiday",
    "Temperature"
]


def build_daily_peak_dataset(cleaned_df):

    work = cleaned_df.copy()

    work["Prediction_Date"] = (
        work["Date_Time"].dt.date
    )

    # Find the actual highest-demand record for each day.
    peak_rows = (
        work.loc[
            work.groupby("Prediction_Date")["Demand_MW"].idxmax()
        ]
        .copy()
    )

    # Daily feature values are taken from the same historical day.
    # Temperature is represented by that day's average temperature.
    daily_temperature = (
        work.groupby("Prediction_Date")["Temperature"]
        .mean()
        .rename("Daily_Temperature")
    )

    daily_holiday = (
        work.groupby("Prediction_Date")["Holiday"]
        .max()
        .rename("Daily_Holiday")
    )

    daily = peak_rows[
        [
            "Prediction_Date",
            "Day",
            "Month",
            "Season"
        ]
    ].copy()

    daily = daily.join(
        daily_temperature,
        on="Prediction_Date"
    )

    daily = daily.join(
        daily_holiday,
        on="Prediction_Date"
    )

    daily["Peak_Hour"] = (
        peak_rows["Hour"].astype(int).values
    )

    daily["Holiday"] = (
        daily["Daily_Holiday"]
        .fillna(0)
        .astype(int)
    )

    daily["Temperature"] = (
        daily["Daily_Temperature"]
        .astype(float)
    )

    daily = daily[
        [
            "Month",
            "Season",
            "Holiday",
            "Temperature",
            "Peak_Hour"
        ]
    ].reset_index(drop=True)

    return daily


def create_peak_hour_model():

    categorical_features = [
        "Season"
    ]

    numeric_features = [
        "Month",
        "Holiday",
        "Temperature"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "season",
                create_one_hot_encoder(),
                categorical_features
            )
        ],
        remainder="passthrough"
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        max_depth=None,
        min_samples_split=2,
        n_jobs=-1,
        class_weight="balanced"
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(filepath):

    cleaned_df = clean_dataset(
        filepath
    )

    if len(cleaned_df) < 2:

        raise ValueError(
            "At least two valid records are required to train the model."
        )

    # --------------------------------------------------------
    # 1. DEMAND REGRESSION MODEL
    # --------------------------------------------------------

    X = cleaned_df[
        FEATURE_COLUMNS
    ].copy()

    y = cleaned_df[
        TARGET_COLUMN
    ].copy()

    demand_model = create_demand_model()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    demand_model.fit(
        X_train,
        y_train
    )

    y_pred_train = demand_model.predict(
        X_train
    )

    y_pred_test = demand_model.predict(
        X_test
    )

    mae = float(
        mean_absolute_error(
            y_test,
            y_pred_test
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_test,
                y_pred_test
            )
        )
    )

    r2 = float(
        r2_score(
            y_test,
            y_pred_test
        )
    )

    # --------------------------------------------------------
    # 2. SEPARATE PEAK-HOUR CLASSIFIER
    # --------------------------------------------------------

    daily_peak_df = build_daily_peak_dataset(
        cleaned_df
    )

    if len(daily_peak_df) < 2:

        raise ValueError(
            "The CSV must contain at least two different dates so "
            "the peak-hour model can learn peak time from the features."
        )

    peak_X = daily_peak_df[
        PEAK_FEATURE_COLUMNS
    ].copy()

    peak_y = (
        daily_peak_df["Peak_Hour"]
        .astype(int)
        .copy()
    )

    # A classifier cannot learn different peak times when the entire
    # historical dataset contains only one peak-hour class.
    unique_peak_hours = sorted(
        peak_y.unique().tolist()
    )

    if len(unique_peak_hours) < 2:

        raise ValueError(
            "The uploaded CSV contains only one historical peak hour "
            f"({unique_peak_hours[0]:02d}:00). "
            "The peak-time model needs historical days with at least "
            "two different peak hours so the peak time can change "
            "according to the input features."
        )

    peak_hour_model = create_peak_hour_model()

    # Keep a validation split only when the dataset is large enough.
    # The final classifier is then fitted on all daily records.
    peak_validation_accuracy = None

    if len(peak_X) >= 10 and peak_y.value_counts().min() >= 2:

        peak_X_train, peak_X_test, peak_y_train, peak_y_test = train_test_split(
            peak_X,
            peak_y,
            test_size=0.20,
            random_state=42,
            stratify=peak_y
        )

        validation_model = create_peak_hour_model()

        validation_model.fit(
            peak_X_train,
            peak_y_train
        )

        peak_test_pred = validation_model.predict(
            peak_X_test
        )

        peak_validation_accuracy = float(
            accuracy_score(
                peak_y_test,
                peak_test_pred
            )
        )

    # Fit final peak model using ALL historical daily records.
    peak_hour_model.fit(
        peak_X,
        peak_y
    )

    # --------------------------------------------------------
    # Save BOTH models together.
    # --------------------------------------------------------

    os.makedirs(
        MODEL_FOLDER,
        exist_ok=True
    )

    model_bundle = {
        "demand_model": demand_model,
        "peak_hour_model": peak_hour_model,
        "peak_feature_columns": PEAK_FEATURE_COLUMNS,
        "peak_training_days": int(len(daily_peak_df)),
        "peak_training_hours": unique_peak_hours,
        "peak_validation_accuracy": peak_validation_accuracy
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH
    )

    training_records = []

    for actual, predicted in zip(
        y_train.tolist(),
        y_pred_train.tolist()
    ):

        training_records.append(
            {
                "actual": float(actual),
                "predicted": float(predicted)
            }
        )

    testing_records = []

    for actual, predicted in zip(
        y_test.tolist(),
        y_pred_test.tolist()
    ):

        testing_records.append(
            {
                "actual": float(actual),
                "predicted": float(predicted)
            }
        )

    metrics = {
        "records": int(len(cleaned_df)),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "features": FEATURE_COLUMNS,
        "peak_model": "RandomForestClassifier",
        "peak_training_days": int(len(daily_peak_df)),
        "historical_peak_hours": [
            f"{h:02d}:00"
            for h in unique_peak_hours
        ],
        "peak_validation_accuracy": peak_validation_accuracy,
        "training_records_data": training_records,
        "testing_records_data": testing_records
    }

    return cleaned_df, metrics


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "Trained model not found. Please upload a CSV and train the model first."
        )

    saved = joblib.load(
        MODEL_PATH
    )

    # New model format.
    if isinstance(saved, dict):

        if (
            "demand_model" not in saved
            or "peak_hour_model" not in saved
        ):

            raise ValueError(
                "The saved model is incomplete. Please upload the CSV again "
                "to retrain the new demand and peak-hour models."
            )

        return saved

    # Old model format is deliberately rejected because it does not
    # contain the separate peak-hour classifier.
    raise ValueError(
        "An older PeakDemand model was found. Please upload the CSV again "
        "so the new feature-based peak-time model can be trained."
    )


# ============================================================
# DEMAND LEVEL
# ============================================================

def get_demand_level(demand):

    demand = float(demand)

    if demand < 300:
        return "LOW"

    if demand < 400:
        return "MEDIUM"

    if demand < 500:
        return "HIGH"

    return "CRITICAL"


# ============================================================
# SUGGESTION
# ============================================================

def get_suggestion(demand_level, peak_demand):

    peak_demand = float(peak_demand)

    if demand_level == "LOW":

        return (
            "Demand is currently low. This is a suitable period "
            "for planned maintenance and flexible energy usage."
        )

    if demand_level == "MEDIUM":

        return (
            "Demand is moderate. Monitor consumption and avoid "
            "unnecessary high-power usage during the predicted peak period."
        )

    if demand_level == "HIGH":

        return (
            "Demand is high. Reduce non-essential loads and shift "
            "flexible activities away from the predicted peak hour."
        )

    return (
        f"Demand is critical at approximately {peak_demand:.2f} MW. "
        "Immediate demand-management measures are recommended, "
        "including reducing non-essential loads and shifting flexible consumption."
    )


# ============================================================
# PEAK LIMIT STATUS
# ============================================================

def check_peak_limit(peak_demand):

    peak_demand = float(peak_demand)

    reached = peak_demand >= PEAK_LIMIT

    excess = max(
        0.0,
        peak_demand - PEAK_LIMIT
    )

    return {
        "limit": float(PEAK_LIMIT),
        "reached": bool(reached),
        "peak_demand": peak_demand,
        "excess": float(excess)
    }


# ============================================================
# PEAK-HOUR ALERT MESSAGE
#
# This only describes the predicted peak. The actual live alert
# is activated by the dashboard only when the current hour equals
# this backend-predicted peak hour.
# ============================================================

def get_peak_alert(peak_time, peak_demand):

    status = check_peak_limit(
        peak_demand
    )

    if status["reached"]:

        message = (
            f"Peak demand is predicted at {peak_time} with "
            f"{status['peak_demand']:.2f} MW, which is at or above "
            f"the configured limit of {PEAK_LIMIT:.2f} MW. "
            "The live alert activates only during this peak hour."
        )

    else:

        message = (
            f"Predicted peak time is {peak_time} with "
            f"{status['peak_demand']:.2f} MW. "
            "The live alert activates only during this peak hour."
        )

    return {
        **status,
        "peak_time": peak_time,
        "message": message
    }


# ============================================================
# 24-HOUR PREDICTION
#
# IMPORTANT:
# peak_time is obtained from the separate peak-hour classifier.
# It is NOT obtained using np.argmax(predictions).
# ============================================================

def predict_hourly(
    prediction_date,
    temperature,
    holiday
):

    model_bundle = load_model()

    demand_model = model_bundle[
        "demand_model"
    ]

    peak_hour_model = model_bundle[
        "peak_hour_model"
    ]

    prediction_date = pd.to_datetime(
        prediction_date
    )

    temperature = float(
        temperature
    )

    holiday = int(
        holiday
    )

    season = get_season(
        prediction_date.month
    )

    # --------------------------------------------------------
    # Hourly demand prediction for the graph.
    # --------------------------------------------------------

    rows = []

    for hour in range(24):

        rows.append(
            {
                "Hour": hour,
                "Day": prediction_date.day,
                "Month": prediction_date.month,
                "Season": season,
                "Holiday": holiday,
                "Temperature": temperature
            }
        )

    input_df = pd.DataFrame(
        rows
    )

    predictions = demand_model.predict(
        input_df[
            FEATURE_COLUMNS
        ]
    )

    hourly_predictions = []

    for hour, demand in zip(
        range(24),
        predictions
    ):

        hourly_predictions.append(
            {
                "hour": int(hour),
                "time": f"{hour:02d}:00",
                "demand": float(demand)
            }
        )

    # --------------------------------------------------------
    # FEATURE-BASED PEAK TIME
    #
    # This is the important change:
    # the classifier predicts the peak hour directly from
    # date/season/holiday/temperature features.
    # --------------------------------------------------------

    peak_input = pd.DataFrame(
        [
            {
                "Month": prediction_date.month,
                "Season": season,
                "Holiday": holiday,
                "Temperature": temperature
            }
        ]
    )

    predicted_peak_hour = int(
        peak_hour_model.predict(
            peak_input[
                PEAK_FEATURE_COLUMNS
            ]
        )[0]
    )

    # Make sure the classifier output is a valid 24-hour value.
    predicted_peak_hour = max(
        0,
        min(
            23,
            predicted_peak_hour
        )
    )

    peak_time = (
        f"{predicted_peak_hour:02d}:00"
    )

    # Demand shown for the predicted peak hour comes from the
    # demand regression model, but it does NOT determine the hour.
    peak_demand = float(
        predictions[predicted_peak_hour]
    )

    demand_level = get_demand_level(
        peak_demand
    )

    suggestion = get_suggestion(
        demand_level,
        peak_demand
    )

    peak_alert = get_peak_alert(
        peak_time,
        peak_demand
    )

    return {
        "prediction_date": prediction_date.strftime(
            "%Y-%m-%d"
        ),
        "temperature": temperature,
        "holiday": holiday,
        "season": season,
        "hourly_predictions": hourly_predictions,

        # Feature-based peak-time result.
        "peak_time": peak_time,
        "peak_hour": predicted_peak_hour,

        # This is retained for compatibility with the dashboard.
        "peak_index": predicted_peak_hour,

        # Demand at the classifier-selected peak hour.
        "peak_demand": peak_demand,

        "demand_level": demand_level,
        "suggestion": suggestion,
        "peak_limit": peak_alert["limit"],
        "peak_limit_reached": peak_alert["reached"],
        "peak_limit_excess": peak_alert["excess"],
        "peak_alert_message": peak_alert["message"],

        "peak_time_source": "feature_based_peak_hour_classifier"
    }
