import os
import json
import re
from datetime import datetime
from functools import wraps

import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from werkzeug.utils import secure_filename

from dotenv import load_dotenv
from sqlalchemy.engine import URL

from .database import db

from .ml_model import (
    train_model,
    predict_hourly,
    PEAK_LIMIT
)


BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

# Load the .env file from the PeakDemand project folder.
# override=True prevents an old Windows environment variable from
# overriding the values in this project's .env file.
ENV_FILE = os.path.join(BASE_DIR, ".env")
load_dotenv(
    dotenv_path=ENV_FILE,
    override=True
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


app = Flask(
    __name__,
    template_folder=os.path.join(
        BASE_DIR,
        "templates"
    ),
    static_folder=os.path.join(
        BASE_DIR,
        "static"
    )
)


app.config["SECRET_KEY"] = os.getenv(
    "SECRET_KEY",
    "peakdemand_secret"
)

MYSQL_HOST = os.getenv(
    "MYSQL_HOST",
    "localhost"
)

MYSQL_PORT = os.getenv(
    "MYSQL_PORT",
    "3306"
)

MYSQL_USER = os.getenv(
    "MYSQL_USER",
    "root"
)

MYSQL_PASSWORD = os.getenv(
    "MYSQL_PASSWORD",
    ""
)

MYSQL_DATABASE = os.getenv(
    "MYSQL_DATABASE",
    "peakdemand"
)

# Prevent a malformed old environment value such as
# 1904@localhost from being used as the MySQL hostname.
if not MYSQL_HOST or "@" in MYSQL_HOST:
    MYSQL_HOST = "localhost"

# SQLAlchemy URL.create() safely handles special characters
# such as @ in the MySQL password.
mysql_url = URL.create(
    drivername="mysql+pymysql",
    username=MYSQL_USER,
    password=MYSQL_PASSWORD,
    host=MYSQL_HOST,
    port=int(MYSQL_PORT),
    database=MYSQL_DATABASE
)

app.config["SQLALCHEMY_DATABASE_URI"] = mysql_url.render_as_string(
    hide_password=False
)

app.config[
    "SQLALCHEMY_TRACK_MODIFICATIONS"
] = False

app.config[
    "MAX_CONTENT_LENGTH"
] = 50 * 1024 * 1024

app.config[
    "UPLOAD_FOLDER"
] = UPLOAD_FOLDER


db.init_app(app)


# ============================================================
# DATABASE MODELS
# ============================================================

class User(db.Model):

    __tablename__ = "users"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(120),
        nullable=False
    )

    email = db.Column(
        db.String(150),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(255),
        nullable=False
    )

    role = db.Column(
        db.String(20),
        nullable=False,
        default="user"
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


class Dataset(db.Model):

    __tablename__ = "datasets"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False
    )

    filename = db.Column(
        db.String(255),
        nullable=False
    )

    records = db.Column(
        db.Integer,
        default=0
    )

    columns = db.Column(
        db.Text
    )

    uploaded_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


class DatasetRow(db.Model):

    __tablename__ = "dataset_rows"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    dataset_id = db.Column(
        db.Integer,
        db.ForeignKey("datasets.id"),
        nullable=False
    )

    date_time = db.Column(
        db.DateTime,
        nullable=False
    )

    hour = db.Column(
        db.Integer
    )

    day = db.Column(
        db.Integer
    )

    month = db.Column(
        db.Integer
    )

    season = db.Column(
        db.String(50)
    )

    holiday = db.Column(
        db.Integer
    )

    temperature = db.Column(
        db.Float
    )

    demand_mw = db.Column(
        db.Float
    )


class ModelRun(db.Model):

    __tablename__ = "model_runs"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False
    )

    dataset_id = db.Column(
        db.Integer,
        db.ForeignKey("datasets.id"),
        nullable=False
    )

    records = db.Column(
        db.Integer
    )

    training_records = db.Column(
        db.Integer
    )

    testing_records = db.Column(
        db.Integer
    )

    mae = db.Column(
        db.Float
    )

    rmse = db.Column(
        db.Float
    )

    r2 = db.Column(
        db.Float
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


class Prediction(db.Model):

    __tablename__ = "predictions"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False
    )

    prediction_date = db.Column(
        db.Date,
        nullable=False
    )

    temperature = db.Column(
        db.Float,
        nullable=False
    )

    season = db.Column(
        db.String(50)
    )

    holiday = db.Column(
        db.Integer
    )

    peak_time = db.Column(
        db.String(20)
    )

    peak_demand = db.Column(
        db.Float
    )

    demand_level = db.Column(
        db.String(30)
    )

    suggestion = db.Column(
        db.Text
    )

    hourly_predictions = db.Column(
        db.Text
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


# ============================================================
# LOGIN HELPERS
# ============================================================

def login_required(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if "user_id" not in session:
            flash(
                "Please login first.",
                "warning"
            )

            return redirect(
                url_for("login")
            )

        return function(
            *args,
            **kwargs
        )

    return wrapper


def role_required(role):

    def decorator(function):

        @wraps(function)
        def wrapper(*args, **kwargs):

            if "user_id" not in session:
                return redirect(
                    url_for("login")
                )

            if session.get("role") != role:
                flash(
                    "You do not have permission to access this page.",
                    "danger"
                )

                return redirect(
                    url_for("dashboard")
                )

            return function(
                *args,
                **kwargs
            )

        return wrapper

    return decorator


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        role = request.form.get(
            "role",
            "user"
        )

        admin_key = request.form.get(
            "admin_key",
            ""
        )

        if not name or not email or not password:

            flash(
                "All required fields must be filled.",
                "danger"
            )

            return redirect(
                url_for("register")
            )

        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "danger"
            )

            return redirect(
                url_for("register")
            )

        if role not in ["user", "admin"]:

            role = "user"

        if role == "admin":

            correct_admin_key = os.getenv(
                "ADMIN_REGISTRATION_KEY",
                "PEAKADMIN2026"
            )

            if admin_key != correct_admin_key:

                flash(
                    "Invalid admin registration key.",
                    "danger"
                )

                return redirect(
                    url_for("register")
                )

        existing_user = User.query.filter_by(
            email=email
        ).first()

        if existing_user:

            flash(
                "Email already registered.",
                "danger"
            )

            return redirect(
                url_for("login")
            )

        user = User(
            name=name,
            email=email,
            password=generate_password_hash(
                password
            ),
            role=role
        )

        db.session.add(
            user
        )

        db.session.commit()

        flash(
            "Registration successful. Please login.",
            "success"
        )

        return redirect(
            url_for("login")
        )

    return render_template(
        "register.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        selected_role = request.form.get(
            "role",
            "user"
        )

        user = User.query.filter_by(
            email=email
        ).first()

        if (
            user
            and check_password_hash(
                user.password,
                password
            )
            and user.role == selected_role
        ):

            session.clear()

            session["user_id"] = user.id
            session["name"] = user.name
            session["role"] = user.role
            session["email"] = user.email

            flash(
                f"Welcome back, {user.name}!",
                "success"
            )

            return redirect(
                url_for("dashboard")
            )

        flash(
            "Invalid email, password, or role.",
            "danger"
        )

    return render_template(
        "login.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out.",
        "success"
    )

    return redirect(
        url_for("home")
    )


# ============================================================
# DASHBOARD ROUTER
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():

    if session.get("role") == "admin":

        return redirect(
            url_for("admin_dashboard")
        )

    return redirect(
        url_for("user_dashboard")
    )


# ============================================================
# USER DASHBOARD
# ============================================================

@app.route("/user/dashboard")
@role_required("user")
def user_dashboard():

    user_id = session["user_id"]

    predictions = Prediction.query.filter_by(
        user_id=user_id
    ).order_by(
        Prediction.created_at.desc()
    ).all()

    datasets = Dataset.query.filter_by(
        user_id=user_id
    ).order_by(
        Dataset.uploaded_at.desc()
    ).all()

    latest_prediction = (
        predictions[0]
        if predictions
        else None
    )

    hourly_predictions = []

    if latest_prediction:

        try:

            hourly_predictions = json.loads(
                latest_prediction.hourly_predictions
            )

        except Exception:

            hourly_predictions = []

    # ========================================================
    # PEAK-TIME-ONLY ALERT
    #
    # The peak time stored in the Prediction table is produced by
    # the separate feature-based peak-hour ML model.
    #
    # The red/live alert is TRUE only when:
    #   1. current hour == predicted peak hour
    #   2. predicted peak demand >= PEAK_LIMIT
    #
    # Therefore a limit breach at another hour does not activate
    # the peak alert.
    # ========================================================

    peak_limit = PEAK_LIMIT
    peak_limit_reached = False
    peak_excess = 0.0
    peak_alert_message = ""
    is_peak_time_now = False

    if latest_prediction:

        try:

            current_peak_demand = float(
                latest_prediction.peak_demand
            )

            stored_peak_time = str(
                latest_prediction.peak_time
            ).strip()

            peak_hour_match = re.match(
                r"^(\d{1,2})",
                stored_peak_time
            )

            predicted_peak_hour = (
                int(peak_hour_match.group(1))
                if peak_hour_match
                else -1
            )

            current_hour = datetime.now().hour

            is_peak_time_now = (
                current_hour == predicted_peak_hour
            )

            peak_excess = max(
                0.0,
                current_peak_demand - peak_limit
            )

            # IMPORTANT: only true during the predicted peak hour.
            peak_limit_reached = (
                is_peak_time_now
                and current_peak_demand >= peak_limit
            )

            if peak_limit_reached:

                peak_alert_message = (
                    f"Peak-time alert is active now at "
                    f"{latest_prediction.peak_time}. Predicted demand is "
                    f"{current_peak_demand:.2f} MW, which is at or above "
                    f"the configured limit of {peak_limit:.2f} MW."
                )

        except (TypeError, ValueError):

            peak_limit_reached = False
            peak_excess = 0.0
            peak_alert_message = ""
            is_peak_time_now = False

    return render_template(
        "user_dashboard.html",
        predictions=predictions,
        datasets=datasets,
        latest_prediction=latest_prediction,
        hourly_predictions=hourly_predictions,
        peak_limit=peak_limit,
        peak_limit_reached=peak_limit_reached,
        peak_excess=peak_excess,
        peak_alert_message=peak_alert_message,
        is_peak_time_now=is_peak_time_now
    )


# ============================================================
# ADMIN DASHBOARD
# ============================================================

@app.route("/admin/dashboard")
@role_required("admin")
def admin_dashboard():

    users = User.query.order_by(
        User.created_at.desc()
    ).all()

    datasets = Dataset.query.order_by(
        Dataset.uploaded_at.desc()
    ).all()

    predictions = Prediction.query.order_by(
        Prediction.created_at.desc()
    ).limit(20).all()

    model_runs = ModelRun.query.order_by(
        ModelRun.created_at.desc()
    ).all()

    return render_template(
        "admin_dashboard.html",
        users=users,
        datasets=datasets,
        predictions=predictions,
        model_runs=model_runs
    )


# ============================================================
# UPLOAD CSV
# ============================================================

@app.route(
    "/upload",
    methods=["GET", "POST"]
)
@login_required
def upload():

    if request.method == "POST":

        if "csv_file" not in request.files:

            flash(
                "Please select a CSV file.",
                "danger"
            )

            return redirect(
                url_for("upload")
            )

        file = request.files[
            "csv_file"
        ]

        if not file.filename:

            flash(
                "Please select a file.",
                "danger"
            )

            return redirect(
                url_for("upload")
            )

        if not file.filename.lower().endswith(
            ".csv"
        ):

            flash(
                "Only CSV files are allowed.",
                "danger"
            )

            return redirect(
                url_for("upload")
            )

        filename = secure_filename(
            file.filename
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d%H%M%S"
        )

        saved_filename = (
            f"{timestamp}_{filename}"
        )

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            saved_filename
        )

        file.save(
            filepath
        )

        try:

            df = pd.read_csv(
                filepath
            )

            # Train model
            cleaned_df, metrics = train_model(
                filepath
            )

            dataset = Dataset(
                user_id=session["user_id"],
                filename=saved_filename,
                records=len(cleaned_df),
                columns=json.dumps(
                    list(cleaned_df.columns)
                )
            )

            db.session.add(
                dataset
            )

            db.session.flush()

            # Store every cleaned CSV row in MySQL
            for _, row in cleaned_df.iterrows():

                dataset_row = DatasetRow(
                    dataset_id=dataset.id,
                    date_time=row["Date_Time"].to_pydatetime(),
                    hour=int(row["Hour"]),
                    day=int(row["Day"]),
                    month=int(row["Month"]),
                    season=str(row["Season"]),
                    holiday=int(row["Holiday"]),
                    temperature=float(row["Temperature"]),
                    demand_mw=float(row["Demand_MW"])
                )

                db.session.add(
                    dataset_row
                )

            # Store model metrics
            model_run = ModelRun(
                user_id=session["user_id"],
                dataset_id=dataset.id,
                records=metrics["records"],
                training_records=metrics["training_records"],
                testing_records=metrics["testing_records"],
                mae=metrics["mae"],
                rmse=metrics["rmse"],
                r2=metrics["r2"]
            )

            db.session.add(
                model_run
            )

            db.session.commit()

            flash(
                "CSV uploaded, data extracted, database records created, and ML model trained successfully.",
                "success"
            )

            return redirect(
                url_for("dashboard")
            )

        except Exception as error:

            db.session.rollback()

            if os.path.exists(filepath):

                os.remove(filepath)

            flash(
                f"Processing failed: {str(error)}",
                "danger"
            )

            return redirect(
                url_for("upload")
            )

    return render_template(
        "upload.html"
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
@login_required
def predict():

    if request.method == "POST":

        prediction_date = request.form.get(
            "prediction_date"
        )

        temperature = request.form.get(
            "temperature"
        )

        holiday = request.form.get(
            "holiday"
        )

        if not prediction_date:
            flash(
                "Please select a date.",
                "danger"
            )

            return redirect(
                url_for("predict")
            )

        try:

            result = predict_hourly(
                prediction_date,
                temperature,
                holiday
            )

            prediction = Prediction(
                user_id=session["user_id"],
                prediction_date=pd.to_datetime(
                    prediction_date
                ).date(),
                temperature=float(
                    temperature
                ),
                season=result["season"],
                holiday=int(
                    holiday
                ),
                peak_time=result["peak_time"],
                peak_demand=result["peak_demand"],
                demand_level=result["demand_level"],
                suggestion=result["suggestion"],
                hourly_predictions=json.dumps(
                    result["hourly_predictions"]
                )
            )

            db.session.add(
                prediction
            )

            db.session.commit()

            session["latest_prediction_id"] = (
                prediction.id
            )

            flash(
                "24-hour demand prediction completed successfully.",
                "success"
            )

            return redirect(
                url_for(
                    "dashboard"
                )
            )

        except Exception as error:

            db.session.rollback()

            flash(
                f"Prediction failed: {str(error)}",
                "danger"
            )

    return render_template(
        "predict.html"
    )


# ============================================================
# API FOR LATEST PREDICTION
# ============================================================

@app.route(
    "/api/latest-prediction"
)
@login_required
def latest_prediction_api():

    prediction = Prediction.query.filter_by(
        user_id=session["user_id"]
    ).order_by(
        Prediction.created_at.desc()
    ).first()

    if not prediction:

        return jsonify(
            {
                "success": False
            }
        )

    return jsonify(
        {
            "success": True,
            "peak_time": prediction.peak_time,
            "peak_demand": prediction.peak_demand,
            "peak_limit": PEAK_LIMIT,
            "peak_limit_reached": (
                datetime.now().hour == int(
                    str(prediction.peak_time).split(":")[0]
                )
                and float(prediction.peak_demand) >= PEAK_LIMIT
            ),
            "is_peak_time_now": (
                datetime.now().hour == int(
                    str(prediction.peak_time).split(":")[0]
                )
            ),
            "peak_excess": max(
                0.0,
                float(prediction.peak_demand) - PEAK_LIMIT
            ),
            "peak_time_source": "feature_based_peak_hour_classifier",
            "demand_level": prediction.demand_level,
            "suggestion": prediction.suggestion,
            "hourly_predictions": json.loads(
                prediction.hourly_predictions
            )
        }
    )


# ============================================================
# CREATE TABLES
# ============================================================

with app.app_context():

    db.create_all()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )