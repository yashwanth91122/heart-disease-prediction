import unittest

from app import app


class HeartPredictionAppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def _valid_payload(self):
        return {
            "age": "55",
            "sex": "1",
            "cp": "2",
            "trestbps": "140",
            "chol": "240",
            "fbs": "0",
            "restecg": "1",
            "thalach": "150",
            "exang": "0",
            "oldpeak": "1.5",
            "slope": "2",
            "ca": "0",
            "thal": "1",
        }

    def test_welcome_page_is_public(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Heart Disease Prediction Website", response.data)
        self.assertIn(b"Enter Dashboard", response.data)

    def test_dashboard_page_is_public(self):
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Patient Input Form", response.data)

    def test_login_redirects_to_dashboard_without_credentials(self):
        response = self.client.get("/login", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.location.endswith("/dashboard"))

    def test_predict_success(self):
        response = self.client.post("/predict", data=self._valid_payload())
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b"Heart Disease Detected" in response.data
            or b"No Heart Disease" in response.data
        )
        self.assertIn(b"Doctor Consultation Guidance", response.data)

    def test_predict_rejects_invalid_input_range(self):
        payload = self._valid_payload()
        payload["age"] = "999"

        response = self.client.post("/predict", data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Error:", response.data)
        self.assertIn(b"age must be between", response.data)

    def test_predict_rejects_non_integer_for_integer_field(self):
        payload = self._valid_payload()
        payload["sex"] = "0.5"

        response = self.client.post("/predict", data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Error:", response.data)
        self.assertIn(b"sex must be an integer", response.data)

    def test_predict_compares_with_last_prediction(self):
        first_response = self.client.post("/predict", data=self._valid_payload())
        self.assertEqual(first_response.status_code, 200)
        self.assertIn(
            b"No last prediction found yet. Submit one more report to compare last and this.",
            first_response.data,
        )

        second_response = self.client.post("/predict", data=self._valid_payload())
        self.assertEqual(second_response.status_code, 200)
        self.assertIn(b"Comparison With Last Prediction", second_response.data)
        self.assertIn(b"Last Prediction", second_response.data)
        self.assertIn(b"Current Prediction", second_response.data)
        self.assertIn(b"Probability Trend:", second_response.data)

    def test_history_page_and_export(self):
        self.client.post("/predict", data=self._valid_payload())
        self.client.post("/predict", data=self._valid_payload())

        history_response = self.client.get("/history")
        self.assertEqual(history_response.status_code, 200)
        self.assertIn(b"Prediction History", history_response.data)
        self.assertIn(b"Recent entries:", history_response.data)

        export_response = self.client.get("/history/export")
        self.assertEqual(export_response.status_code, 200)
        self.assertIn(b"generated_at,result,risk_band,probability_percent,input_features", export_response.data)

    def test_api_predict_public_access(self):
        response = self.client.post("/api/predict", json=self._valid_payload())
        self.assertEqual(response.status_code, 200)
        self.assertIn("result", response.get_json())

    def test_api_predict_success(self):
        response = self.client.post("/api/predict", json=self._valid_payload())
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn(payload["result"], ["Heart Disease Detected", "No Heart Disease"])
        self.assertIn(payload["risk_band"], ["Low", "Moderate", "High", "Not Available"])
        self.assertIn("care_plan", payload)
        self.assertIn("input_features", payload)

    def test_suggestions_are_public(self):
        response = self.client.get("/suggestions")
        self.assertEqual(response.status_code, 200)

    def test_suggestions_hub_and_detail(self):
        hub_response = self.client.get("/suggestions")
        self.assertEqual(hub_response.status_code, 200)
        self.assertIn(b"Suggestion Center", hub_response.data)
        self.assertIn(b"Low Risk Suggestions", hub_response.data)
        self.assertIn(b"Moderate Risk Suggestions", hub_response.data)
        self.assertIn(b"High Risk Suggestions", hub_response.data)

        detail_response = self.client.get("/suggestions/high")
        self.assertEqual(detail_response.status_code, 200)
        self.assertIn(b"High Risk Suggestions", detail_response.data)
        self.assertIn(b"Emergency Red Flags", detail_response.data)

    def test_suggestion_detail_invalid_level(self):
        response = self.client.get("/suggestions/not-a-level")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
