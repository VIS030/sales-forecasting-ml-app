import json
import os
import sqlite3
from datetime import datetime
from functools import wraps

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from flask import (
    Flask,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from werkzeug.security import check_password_hash, generate_password_hash

matplotlib.use("Agg")

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "ecommerce_dataset.csv")
DB_PATH = os.path.join(BASE_DIR, "sales.db")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
EDA_DIR = os.path.join(BASE_DIR, "static", "eda")

REQUIRED_COLUMNS = [
    "UserID",
    "UserName",
    "Age",
    "Gender",
    "Country",
    "SignUpDate",
    "ProductID",
    "ProductName",
    "Category",
    "Price",
    "PurchaseDate",
    "Quantity",
    "TotalAmount",
    "HasDiscountApplied",
    "DiscountRate",
    "ReviewScore",
]

NUMERIC_FEATURES = ["Price", "Quantity", "DiscountRate"]
CATEGORICAL_FEATURES = ["Category", "Country"]
TARGET = "TotalAmount"
MODEL_ALGO = os.getenv("MODEL_ALGO", "rf").strip().lower()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-in-production")


def ensure_folders():
    os.makedirs(os.path.join(BASE_DIR, "templates"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "static", "css"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "static", "js"), exist_ok=True)
    os.makedirs(EDA_DIR, exist_ok=True)


def load_and_clean_dataset():
    df = pd.read_csv(DATASET_PATH)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[[col for col in REQUIRED_COLUMNS if col in df.columns]].copy()

    for date_col in ["PurchaseDate", "SignUpDate"]:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Normalize boolean values from mixed source formats.
    if "HasDiscountApplied" in df.columns:
        df["HasDiscountApplied"] = (
            df["HasDiscountApplied"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0})
            .fillna(0)
            .astype(int)
        )

    numeric_cols = [
        "Age",
        "Price",
        "Quantity",
        "TotalAmount",
        "DiscountRate",
        "ReviewScore",
        "HasDiscountApplied",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = ["UserName", "Gender", "Country", "ProductName", "Category"]
    for col in categorical_cols:
        if col in df.columns:
            mode = df[col].mode(dropna=True)
            mode_value = mode.iloc[0] if not mode.empty else "Unknown"
            df[col] = df[col].fillna(mode_value).astype(str)

    for col in ["PurchaseDate", "SignUpDate"]:
        if col in df.columns:
            fallback = df[col].dropna().median() if not df[col].dropna().empty else pd.Timestamp("2021-01-01")
            df[col] = df[col].fillna(fallback)

    return df


def populate_database(df):
    conn = sqlite3.connect(DB_PATH)
    db_df = df.copy()
    db_df["PurchaseDate"] = db_df["PurchaseDate"].dt.strftime("%Y-%m-%d")
    db_df["SignUpDate"] = db_df["SignUpDate"].dt.strftime("%Y-%m-%d")
    db_df.to_sql("sales_data", conn, if_exists="replace", index=False)
    conn.close()


def ensure_users_table():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def run_eda(df):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5))
    trend = df.groupby(df["PurchaseDate"].dt.to_period("M"))["TotalAmount"].sum()
    trend.index = trend.index.astype(str)
    trend.plot(marker="o", color="#2563eb")
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "sales_trend.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    df["Category"].value_counts().plot(kind="bar", color="#0ea5e9")
    plt.title("Category Distribution")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "category_distribution.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(df["Age"], bins=24, color="#6366f1", kde=True)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "age_distribution.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    gender = df["Gender"].value_counts()
    plt.pie(gender.values, labels=gender.index, autopct="%1.1f%%", startangle=135)
    plt.title("Gender Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "gender_distribution.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df.sample(min(len(df), 6000), random_state=42), x="DiscountRate", y="TotalAmount", alpha=0.6, color="#f97316")
    plt.title("Discount vs Sales")
    plt.xlabel("Discount Rate")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "discount_vs_sales.png"), dpi=180)
    plt.close()


def build_model_pipeline(model_name):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    if model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
        )
    else:
        model = XGBRegressor(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_models(df):
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    train_df = df[features + [TARGET]].dropna()
    X = train_df[features]
    y = train_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if MODEL_ALGO == "xgb" and XGBOOST_AVAILABLE:
        model_names = ["XGBoost"]
    else:
        model_names = ["Random Forest"]

    results = []
    best = None

    for name in model_names:
        pipeline = build_model_pipeline(name)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        r2 = float(r2_score(y_test, preds))
        mae = float(mean_absolute_error(y_test, preds))
        metrics = {"name": name, "r2": r2, "mae": mae}
        results.append(metrics)

        if best is None or r2 > best["metrics"]["r2"]:
            best = {"pipeline": pipeline, "metrics": metrics}

    artifact = {
        "pipeline": best["pipeline"],
        "metrics": best["metrics"],
        "all_results": results,
        "trained_at": datetime.utcnow().isoformat(),
        "xgboost_available": XGBOOST_AVAILABLE,
    }
    joblib.dump(artifact, MODEL_PATH, compress=3)
    return artifact


def initialize_assets(force_rebuild=False):
    ensure_folders()
    if force_rebuild or not (os.path.exists(DB_PATH) and os.path.exists(MODEL_PATH)):
        df = load_and_clean_dataset()
        populate_database(df)
        run_eda(df)
        artifact = train_models(df)
    else:
        artifact = joblib.load(MODEL_PATH)

    ensure_users_table()
    return artifact


def query_one(sql, params=()):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(sql, params)
    row = cur.fetchone()
    conn.close()
    return row[0] if row else 0


def query_all(sql, params=()):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    return rows


def login_required(route_func):
    @wraps(route_func)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return route_func(*args, **kwargs)

    return wrapper


@app.before_request
def load_current_user():
    g.user = None
    user_id = session.get("user_id")
    if not user_id:
        return
    rows = query_all("SELECT id, username, email FROM users WHERE id = ?", (user_id,))
    if rows:
        row = rows[0]
        g.user = {"id": row[0], "username": row[1], "email": row[2]}


def get_dashboard_data(filters=None):
    filters = filters or {}
    clauses = []
    params = []

    if filters.get("start_date"):
        clauses.append("PurchaseDate >= ?")
        params.append(filters["start_date"])
    if filters.get("end_date"):
        clauses.append("PurchaseDate <= ?")
        params.append(filters["end_date"])
    if filters.get("category"):
        clauses.append("Category = ?")
        params.append(filters["category"])
    if filters.get("country"):
        clauses.append("Country = ?")
        params.append(filters["country"])
    if filters.get("gender"):
        clauses.append("Gender = ?")
        params.append(filters["gender"])

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    kpis = {
        "total_revenue": float(
            query_one(f"SELECT COALESCE(SUM(TotalAmount), 0) FROM sales_data {where_sql}", tuple(params))
        ),
        "total_orders": int(
            query_one(f"SELECT COUNT(*) FROM sales_data {where_sql}", tuple(params))
        ),
        "unique_users": int(
            query_one(f"SELECT COUNT(DISTINCT UserID) FROM sales_data {where_sql}", tuple(params))
        ),
        "avg_rating": float(
            query_one(
                f"SELECT COALESCE(AVG(ReviewScore), 0) FROM sales_data {where_sql}",
                tuple(params),
            )
        ),
    }

    trend_rows = query_all(
        f"""
        SELECT PurchaseDate, SUM(TotalAmount) as Sales
        FROM sales_data
        {where_sql}
        GROUP BY PurchaseDate
        ORDER BY PurchaseDate
        """,
        tuple(params),
    )
    top_products_rows = query_all(
        f"""
        SELECT ProductName, SUM(TotalAmount) as Sales
        FROM sales_data
        {where_sql}
        GROUP BY ProductName
        ORDER BY Sales DESC
        LIMIT 10
        """,
        tuple(params),
    )
    category_rows = query_all(
        f"""
        SELECT Category, SUM(TotalAmount) as Sales
        FROM sales_data
        {where_sql}
        GROUP BY Category
        ORDER BY Sales DESC
        """,
        tuple(params),
    )
    scatter_rows = query_all(
        f"""
        SELECT DiscountRate, TotalAmount
        FROM sales_data
        {where_sql}
        ORDER BY RANDOM()
        LIMIT 1200
        """,
        tuple(params),
    )

    return {
        "kpis": kpis,
        "trend": [{"x": row[0], "y": row[1]} for row in trend_rows],
        "top_products": [{"label": row[0], "value": row[1]} for row in top_products_rows],
        "category_distribution": [{"label": row[0], "value": row[1]} for row in category_rows],
        "discount_scatter": [{"x": row[0], "y": row[1]} for row in scatter_rows],
    }


def get_filter_options():
    categories = [row[0] for row in query_all("SELECT DISTINCT Category FROM sales_data ORDER BY Category")]
    countries = [row[0] for row in query_all("SELECT DISTINCT Country FROM sales_data ORDER BY Country")]
    genders = [row[0] for row in query_all("SELECT DISTINCT Gender FROM sales_data ORDER BY Gender")]
    return {"categories": categories, "countries": countries, "genders": genders}


def get_date_bounds():
    min_date = query_one("SELECT MIN(PurchaseDate) FROM sales_data")
    max_date = query_one("SELECT MAX(PurchaseDate) FROM sales_data")
    return {"min": min_date or "", "max": max_date or ""}


MODEL_ARTIFACT = initialize_assets(force_rebuild=False)


@app.route("/")
def home():
    metrics = MODEL_ARTIFACT["all_results"]
    preview = get_dashboard_data()
    return render_template("home.html", metrics=metrics, preview=json.dumps(preview["trend"][-30:]))


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        user_rows = query_all(
            "SELECT id, username, email, password FROM users WHERE email = ?",
            (email,),
        )
        if not user_rows:
            flash("Invalid email or password.", "danger")
            return render_template("login.html")

        user = user_rows[0]
        if not check_password_hash(user[3], password):
            flash("Invalid email or password.", "danger")
            return render_template("login.html")

        session["user_id"] = user[0]
        flash(f"Welcome back, {user[1]}! Login successful.", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("signup.html")
        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("signup.html")
        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "danger")
            return render_template("signup.html")

        existing = query_all(
            "SELECT id FROM users WHERE email = ? OR username = ?",
            (email, username),
        )
        if existing:
            flash("Email or username already exists. Please use another.", "warning")
            return render_template("signup.html")

        hashed_password = generate_password_hash(password)
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, hashed_password),
        )
        conn.commit()
        conn.close()

        flash("Signup successful. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


@app.route("/dashboard")
@login_required
def dashboard():
    filters = {
        "start_date": request.args.get("start_date", "").strip(),
        "end_date": request.args.get("end_date", "").strip(),
        "category": request.args.get("category", "").strip(),
        "country": request.args.get("country", "").strip(),
        "gender": request.args.get("gender", "").strip(),
    }
    data = get_dashboard_data(filters)
    options = get_filter_options()
    date_bounds = get_date_bounds()
    return render_template(
        "dashboard.html",
        data_json=json.dumps(data),
        filters=filters,
        options=options,
        date_bounds=date_bounds,
    )


@app.route("/predict")
@login_required
def predict_page():
    options = get_filter_options()
    model_metrics = MODEL_ARTIFACT.get("metrics", {})
    return render_template(
        "predict.html",
        categories=options["categories"],
        countries=options["countries"],
        model_metrics=model_metrics,
    )


@app.route("/eda")
@login_required
def eda_page():
    insights = {
        "sales_trend": "Sales show clear monthly seasonality with stable demand over time.",
        "category_distribution": "A few categories dominate volume, signaling focused merchandising opportunities.",
        "age_distribution": "Most buyers are concentrated in mid-age groups, ideal for targeted campaigns.",
        "gender_distribution": "Customer mix remains balanced, supporting broad marketing creatives.",
        "discount_vs_sales": "Higher discounts generally increase order value, but with diminishing returns.",
    }
    return render_template("eda.html", insights=insights)


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    payload = request.get_json(silent=True) or {}

    try:
        price = float(payload.get("price"))
        quantity = float(payload.get("quantity"))
        discount_rate = float(payload.get("discount_rate"))

        # Accept both decimal format (0.15) and percentage format (15).
        if discount_rate > 1:
            discount_rate = discount_rate / 100.0
        discount_rate = max(0.0, min(discount_rate, 0.95))

        if price < 0 or quantity <= 0:
            raise ValueError("Invalid range")

        features = pd.DataFrame(
            [
                {
                    "Price": price,
                    "Quantity": quantity,
                    "DiscountRate": discount_rate,
                    "Category": str(payload.get("category")),
                    "Country": str(payload.get("country")),
                }
            ]
        )
    except Exception:
        return jsonify({"error": "Invalid input values. Check numeric fields."}), 400

    pred_value = float(MODEL_ARTIFACT["pipeline"].predict(features)[0])
    pred_value = max(pred_value, 0.0)
    r2 = float(MODEL_ARTIFACT["metrics"]["r2"])
    confidence = max(60.0, min(99.0, 70 + (r2 * 30)))

    response = {
        "predicted_sales": round(pred_value, 2),
        "confidence_score": round(confidence, 2),
        "model": MODEL_ARTIFACT["metrics"]["name"],
        "evaluated_r2": round(r2, 4),
        "evaluated_mae": round(float(MODEL_ARTIFACT["metrics"]["mae"]), 4),
    }
    return jsonify(response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
