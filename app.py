from pathlib import Path
from datetime import datetime
import csv
import json
import os
from functools import wraps
from io import StringIO

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
import joblib
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "heart-app-dev-secret")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.getenv("SESSION_COOKIE_SECURE", "0") == "1"
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heart_model_ucitrained.joblib"
SCALER_PATH = BASE_DIR / "heart_scaler_ucitrained.joblib"

model = None
scaler = None
model_load_error = None

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as exc:
    model_load_error = str(exc)

APP_USERNAME = os.getenv("APP_USERNAME", "admin")
APP_PASSWORD = os.getenv("APP_PASSWORD", "admin123")

# Expected fields and validation bounds (min, max)
FEATURE_SPECS = {
    "age": (1, 120),
    "sex": (0, 1),
    "cp": (0, 3),
    "trestbps": (70, 250),
    "chol": (100, 700),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (50, 250),
    "exang": (0, 1),
    "oldpeak": (0, 10),
    "slope": (0, 2),
    "ca": (0, 3),
    "thal": (0, 2),
}

FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex (0=Female, 1=Male)",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Serum Cholesterol",
    "fbs": "Fasting Blood Sugar > 120",
    "restecg": "Resting ECG",
    "thalach": "Max Heart Rate",
    "exang": "Exercise Induced Angina",
    "oldpeak": "Oldpeak (ST Depression)",
    "slope": "Slope of ST Segment",
    "ca": "Major Vessels (CA)",
    "thal": "Thalassemia",
}

INTEGER_ONLY_FIELDS = set(FEATURE_SPECS.keys()) - {"oldpeak"}

try:
    PREDICTION_HISTORY_LIMIT = max(int(os.getenv("PREDICTION_HISTORY_LIMIT", "8")), 1)
except ValueError:
    PREDICTION_HISTORY_LIMIT = 8

SUGGESTION_CONTENT = {
    "low": {
        "title": "Low Risk Suggestions",
        "tagline": "Maintain protective habits and track health trends early.",
        "priority_actions": [
            "Keep a weekly exercise routine (at least 150 minutes per week).",
            "Follow a heart-friendly diet with low salt and low trans-fat intake.",
            "Schedule routine BP, sugar, and lipid checks annually.",
        ],
        "diet_guidance": [
            "Prefer whole grains, vegetables, and lean proteins.",
            "Limit processed foods and sugary drinks.",
            "Hydrate regularly and reduce late-night heavy meals.",
        ],
        "monitoring_plan": [
            "Record blood pressure once or twice per month.",
            "Track body weight and waist circumference monthly.",
            "Book preventive doctor review every 6 to 12 months.",
        ],
        "red_flags": [
            "New chest discomfort during exertion.",
            "Unusual breathlessness or fatigue with routine activity.",
            "Sudden dizziness, palpitations, or fainting episode.",
        ],
    },
    "moderate": {
        "title": "Moderate Risk Suggestions",
        "tagline": "Start structured prevention and early clinical follow-up.",
        "priority_actions": [
            "Consult a physician within 1 to 2 weeks for risk review.",
            "Begin a supervised lifestyle correction plan immediately.",
            "Discuss baseline ECG, blood sugar profile, and lipid profile.",
        ],
        "diet_guidance": [
            "Reduce sodium and saturated fats aggressively.",
            "Prioritize high-fiber meals and avoid frequent fried foods.",
            "Restrict tobacco and alcohol; stop smoking completely.",
        ],
        "monitoring_plan": [
            "Track BP and pulse at least 2 to 3 times per week.",
            "Maintain a symptom log (chest pain, breathlessness, fatigue).",
            "Repeat follow-up clinical review in 1 to 3 months.",
        ],
        "red_flags": [
            "Chest tightness lasting more than a few minutes.",
            "Breathlessness at rest or while lying flat.",
            "Radiating pain to jaw, back, or left arm.",
        ],
    },
    "high": {
        "title": "High Risk Suggestions",
        "tagline": "Prioritize urgent cardiology review and symptom safety.",
        "priority_actions": [
            "Arrange cardiologist consultation within 24 to 72 hours.",
            "Avoid heavy exertion until reviewed by a specialist.",
            "Carry prior reports and medicine details to your appointment.",
        ],
        "diet_guidance": [
            "Use a strict low-salt, low-saturated-fat meal plan.",
            "Avoid smoking, alcohol, and stimulant-heavy energy drinks.",
            "Prefer smaller, frequent, balanced meals with vegetables/protein.",
        ],
        "monitoring_plan": [
            "Check BP, pulse, and blood sugar daily if applicable.",
            "Ensure family/caregiver awareness of emergency warning signs.",
            "Discuss further testing (echo, stress test, advanced labs).",
        ],
        "red_flags": [
            "Persistent chest pain above 15 minutes.",
            "Cold sweats, nausea, or sudden severe breathlessness.",
            "Fainting, confusion, or severe left arm/jaw pain.",
        ],
    },
}


def is_prediction_ready():
    return model is not None and scaler is not None and not model_load_error


def parse_and_validate(payload):
    values = []

    for field, (min_value, max_value) in FEATURE_SPECS.items():
        raw = payload.get(field, "")
        raw_text = raw.strip() if isinstance(raw, str) else str(raw).strip()

        if raw is None or raw_text == "":
            return None, f"Missing required input: {field}."

        try:
            value = float(raw_text)
        except ValueError:
            return None, f"Invalid numeric value for {field}."

        if field in INTEGER_ONLY_FIELDS and not value.is_integer():
            return None, f"{field} must be an integer."

        if not (min_value <= value <= max_value):
            return None, f"{field} must be between {min_value} and {max_value}."

        values.append(int(value) if field in INTEGER_ONLY_FIELDS else value)

    return values, None


def run_prediction(features):
    if not is_prediction_ready():
        raise RuntimeError(
            f"Model artifacts could not be loaded ({model_load_error or 'unknown error'})."
        )

    scaled_features = scaler.transform([features])
    prediction = int(model.predict(scaled_features)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(scaled_features)[0][1])

    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
    return {
        "prediction": prediction,
        "result": result,
        "probability": probability,
        "risk_band": get_risk_band(probability),
        "care_plan": build_care_plan(prediction),
        "feature_rows": format_feature_rows(features),
        "generated_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
    }


def build_prediction_record(features, prediction_output):
    feature_map = {
        field: value for field, value in zip(FEATURE_SPECS.keys(), features, strict=False)
    }
    return {
        "result": prediction_output["result"],
        "probability": prediction_output["probability"],
        "risk_band": prediction_output["risk_band"],
        "generated_at": prediction_output["generated_at"],
        "features": feature_map,
    }


def get_prediction_history():
    history = session.get("prediction_history")
    if isinstance(history, list):
        return history
    return []


def add_prediction_to_history(prediction_record):
    history = get_prediction_history()
    history.append(prediction_record)
    session["prediction_history"] = history[-PREDICTION_HISTORY_LIMIT:]
    session.modified = True


def format_feature_rows(features):
    rows = []
    for field, value in zip(FEATURE_SPECS.keys(), features):
        rows.append(
            {
                "field": field,
                "label": FEATURE_LABELS.get(field, field),
                "value": f"{value:g}",
            }
        )
    return rows


def get_risk_band(probability):
    if probability is None:
        return "Not Available"
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Moderate"
    return "Low"


def format_probability(probability):
    if probability is None:
        return "N/A"
    return f"{probability * 100:.2f}%"


def build_prediction_comparison(previous_prediction, current_prediction):
    if not previous_prediction:
        return {"has_previous": False}

    previous_probability = previous_prediction.get("probability")
    current_probability = current_prediction.get("probability")

    trend = "Not Available"
    direction = "same"
    delta_label = "N/A"

    if (
        isinstance(previous_probability, (float, int))
        and isinstance(current_probability, (float, int))
    ):
        delta = float(current_probability) - float(previous_probability)
        delta_label = f"{abs(delta) * 100:.2f}%"

        if abs(delta) < 1e-6:
            trend = "No Change"
            direction = "same"
        elif delta > 0:
            trend = "Increased"
            direction = "up"
        else:
            trend = "Decreased"
            direction = "down"

    return {
        "has_previous": True,
        "previous_result": previous_prediction.get("result", "N/A"),
        "previous_risk_band": previous_prediction.get("risk_band", "N/A"),
        "previous_probability_label": format_probability(previous_probability),
        "previous_generated_at": previous_prediction.get("generated_at", "N/A"),
        "current_result": current_prediction.get("result", "N/A"),
        "current_risk_band": current_prediction.get("risk_band", "N/A"),
        "current_probability_label": format_probability(current_probability),
        "trend": trend,
        "direction": direction,
        "delta_label": delta_label,
    }


def build_care_plan(prediction):
    if prediction == 1:
        return {
            "doctor_to_meet": "Cardiologist",
            "where_to_meet": "Cardiology OPD or nearest multi-specialty hospital.",
            "when_to_meet": "Within 24 to 72 hours.",
            "next_steps": [
                "Book a cardiology consultation as soon as possible.",
                "Avoid heavy physical exertion until reviewed by doctor.",
                "Track BP, blood sugar, and heart rate daily.",
                "Follow a low-salt and low-saturated-fat diet.",
            ],
            "tests_to_discuss": [
                "ECG",
                "Echocardiogram",
                "Lipid profile",
                "Treadmill/Stress test (if advised)",
            ],
            "what_to_carry": [
                "Previous medical records and prescriptions.",
                "Recent blood reports (sugar/cholesterol).",
                "List of current medications and allergies.",
            ],
            "emergency_signs": [
                "Chest pain lasting more than 15 minutes.",
                "Breathlessness at rest or sudden sweating.",
                "Fainting, severe dizziness, or pain to jaw/left arm.",
            ],
            "medication_note": (
                "Take heart medicines only under doctor supervision. "
                "Do not self-start or stop any medicine."
            ),
        }

    return {
        "doctor_to_meet": "General Physician (Routine) / Cardiologist (If symptoms persist)",
        "where_to_meet": "Nearby primary care clinic; cardiology clinic if symptoms develop.",
        "when_to_meet": "Routine preventive check in 6 to 12 months.",
        "next_steps": [
            "Maintain regular exercise and healthy diet.",
            "Continue routine monitoring of BP, sugar, and cholesterol.",
            "Sleep well and manage stress.",
            "Avoid smoking and limit alcohol intake.",
        ],
        "tests_to_discuss": [
            "Annual BP and sugar check",
            "Lipid profile",
            "ECG if symptoms appear",
        ],
        "what_to_carry": [
            "Previous annual health reports.",
            "Any ongoing medicine list.",
        ],
        "emergency_signs": [
            "New chest pain, breathlessness, or unexplained fainting.",
            "Sudden severe fatigue with sweating or nausea.",
        ],
        "medication_note": "No preventive medicine should be started without medical advice.",
    }


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login", next=request.path))
        return view_func(*args, **kwargs)

    return wrapped_view


def api_login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not session.get("user"):
            return jsonify({"error": "Authentication required. Login first."}), 401
        return view_func(*args, **kwargs)

    return wrapped_view


@app.route("/healthz")
def healthz():
    status_code = 200 if is_prediction_ready() else 503
    return (
        jsonify(
            {
                "status": "ok" if is_prediction_ready() else "degraded",
                "model_ready": is_prediction_ready(),
            }
        ),
        status_code,
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    next_url = request.args.get("next", "/")

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        next_url = request.form.get("next", "/")

        if username == APP_USERNAME and password == APP_PASSWORD:
            session["user"] = username
            if next_url and next_url.startswith("/"):
                return redirect(next_url)
            return redirect(url_for("home"))

        error = "Invalid username or password."

    return render_template("login.html", error=error, next_url=next_url)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def home():
    if not is_prediction_ready():
        return render_template(
            "index.html",
            result=f"Error: Model artifacts unavailable ({model_load_error}).",
            probability=None,
        )

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    features, error = parse_and_validate(request.form)
    if error:
        return render_template("index.html", result=f"Error: {error}", probability=None)

    try:
        prediction_output = run_prediction(features)
        current_prediction = build_prediction_record(features, prediction_output)

        history = get_prediction_history()
        previous_prediction = history[-1] if history else None
        comparison = build_prediction_comparison(previous_prediction, current_prediction)

        add_prediction_to_history(current_prediction)
        session["last_prediction"] = current_prediction

        return render_template(
            "result.html",
            result=prediction_output["result"],
            probability=prediction_output["probability"],
            risk_band=prediction_output["risk_band"],
            care_plan=prediction_output["care_plan"],
            feature_rows=prediction_output["feature_rows"],
            generated_at=prediction_output["generated_at"],
            comparison=comparison,
        )
    except Exception as exc:
        return render_template(
            "index.html",
            result=f"Error: Prediction failed ({exc}).",
            probability=None,
        )


@app.route("/history")
@login_required
def history():
    history_entries = list(reversed(get_prediction_history()))
    return render_template(
        "history.html",
        history_entries=history_entries,
        total_entries=len(history_entries),
        history_limit=PREDICTION_HISTORY_LIMIT,
    )


@app.route("/suggestions")
@login_required
def suggestions_hub():
    selected_risk = request.args.get("risk", "").strip().lower()
    if selected_risk not in SUGGESTION_CONTENT:
        selected_risk = ""
    return render_template(
        "suggestions_hub.html",
        suggestions=SUGGESTION_CONTENT,
        selected_risk=selected_risk,
    )


@app.route("/suggestions/<risk_level>")
@login_required
def suggestion_detail(risk_level):
    normalized_level = risk_level.strip().lower()
    suggestion = SUGGESTION_CONTENT.get(normalized_level)
    if suggestion is None:
        abort(404)

    return render_template(
        "suggestion_detail.html",
        risk_level=normalized_level,
        suggestion=suggestion,
    )


@app.route("/history/export", methods=["GET"])
@login_required
def export_history():
    history_entries = get_prediction_history()

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "generated_at",
            "result",
            "risk_band",
            "probability_percent",
            "input_features",
        ]
    )

    for entry in history_entries:
        probability = entry.get("probability")
        probability_percent = (
            f"{probability * 100:.2f}" if isinstance(probability, (float, int)) else "N/A"
        )
        writer.writerow(
            [
                entry.get("generated_at", ""),
                entry.get("result", ""),
                entry.get("risk_band", ""),
                probability_percent,
                json.dumps(entry.get("features", {}), separators=(",", ":")),
            ]
        )

    csv_data = output.getvalue()
    output.close()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=heart_prediction_history.csv"
        },
    )


@app.route("/api/predict", methods=["POST"])
@api_login_required
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON body is required."}), 400

    features, error = parse_and_validate(payload)
    if error:
        return jsonify({"error": error}), 400

    try:
        prediction_output = run_prediction(features)
        current_prediction = build_prediction_record(features, prediction_output)

        history = get_prediction_history()
        previous_prediction = history[-1] if history else None
        comparison = build_prediction_comparison(previous_prediction, current_prediction)

        add_prediction_to_history(current_prediction)
        session["last_prediction"] = current_prediction

        return jsonify(
            {
                "result": prediction_output["result"],
                "probability": prediction_output["probability"],
                "risk_band": prediction_output["risk_band"],
                "generated_at": prediction_output["generated_at"],
                "care_plan": prediction_output["care_plan"],
                "comparison": comparison,
                "input_features": current_prediction["features"],
            }
        )
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Prediction failed ({exc})."}), 500


if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("FLASK_RUN_PORT", "5000")))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
