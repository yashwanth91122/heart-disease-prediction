# Heart Disease Prediction Web App

Flask-based web application that predicts heart disease risk from patient clinical inputs using a trained machine learning model (`joblib`).

## Project Structure

- `app.py`: Flask application (authentication, routes, validation, prediction logic)
- `wsgi.py`: WSGI entrypoint for production servers
- `Procfile`: Process definition for PaaS deployment
- `render.yaml`: Render deployment configuration
- `Dockerfile`: Container image definition
- `templates/index.html`: Multi-step input form
- `templates/welcome.html`: Public welcome/landing page
- `templates/result.html`: Separate printable result paper
- `templates/history.html`: Session-based prediction history page
- `templates/suggestions_hub.html`: Suggestions hub page
- `templates/suggestion_detail.html`: Risk-specific suggestion page
- `heart_model_ucitrained.joblib`: Trained ML model
- `heart_scaler_ucitrained.joblib`: Feature scaler used before prediction

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run (Local Development)

```bash
venv/bin/python app.py
```

Open: `http://127.0.0.1:5000`

## Run (Production-Style)

```bash
gunicorn --bind 0.0.0.0:5000 --workers 2 --threads 4 --timeout 120 wsgi:app
```

## Deploy As Website

### Option 1: Render

1. Push this project to GitHub.
2. Create a new Render Web Service from the repo.
3. Render auto-detects `render.yaml` and uses the configured build/start commands.
4. Set environment variables from dashboard if needed.

### Option 2: Docker

```bash
docker build -t heart-disease-site .
docker run -p 5000:5000 heart-disease-site
```

## Access

No username/password login is required. The app opens to a welcome page and then enters the dashboard.

Environment variables:

- `FLASK_SECRET_KEY`
- `SESSION_COOKIE_SECURE` (`1` in HTTPS deployments)
- `PREDICTION_HISTORY_LIMIT` (default: `8`)

## Routes

- `GET /`: Welcome page
- `GET /dashboard`: Input dashboard
- `GET /login`: Redirects to dashboard (legacy route)
- `POST /logout`: Resets session and redirects to welcome page
- `POST /predict`: Web prediction result page
- `GET /history`: Session prediction history
- `GET /suggestions`: Suggestion hub (low/moderate/high)
- `GET /suggestions/<risk_level>`: Detailed suggestion page for a risk band
- `GET /history/export`: Download history as CSV
- `POST /api/predict`: JSON prediction API
- `GET /healthz`: Health status endpoint

## API Example

Send:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 240,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 0,
    "thal": 1
  }'
```

## Input Features

1. `age`
2. `sex`
3. `cp`
4. `trestbps`
5. `chol`
6. `fbs`
7. `restecg`
8. `thalach`
9. `exang`
10. `oldpeak`
11. `slope`
12. `ca`
13. `thal`

## Notes

- This project is for educational use and not a medical diagnosis tool.
- Model output should be interpreted by qualified healthcare professionals.
