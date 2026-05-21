import os
import sys
import pytest
import numpy as np
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

MOCK_MODEL_DIR = "test_artifacts"
MOCK_LOG_DIR = "test_logs"
os.makedirs(MOCK_MODEL_DIR, exist_ok=True)
os.makedirs(MOCK_LOG_DIR, exist_ok=True)

TEST_MODEL_PATH = os.path.join(MOCK_MODEL_DIR, "test_forecaster.joblib")
TEST_LOG_PATH = os.path.join(MOCK_LOG_DIR, "test_drift.log")

import src.app as app_module
app_module.MODEL_PATH = TEST_MODEL_PATH
app_module.LOG_FILE_PATH = TEST_LOG_PATH
from src.app import app

@pytest.fixture(scope="session", autouse=True)
def init_isolated_model_artifact():
    from sklearn.linear_model import HuberRegressor
    mock_engine = HuberRegressor()
    X_fake = np.random.rand(5, 14)
    y_fake = np.random.rand(5) * 5000
    mock_engine.fit(X_fake, y_fake)
    joblib.dump(mock_engine, TEST_MODEL_PATH)
    yield
    if os.path.exists(TEST_MODEL_PATH): os.remove(TEST_MODEL_PATH)
    if os.path.exists(TEST_LOG_PATH): os.remove(TEST_LOG_PATH)

def test_api_prediction_global_combined():
    with app.test_client() as client:
        payload = {
            "lag_rev_1": 12000.0, "lag_rev_2": 11000.0, "lag_rev_3": 13000.0,
            "lag_rev_4": 12500.0, "lag_rev_5": 11500.0, "lag_rev_6": 14000.0,
            "lag_rev_7": 13500.0, "rolling_mean_3d": 12000.0, "rolling_mean_7d": 12428.5,
            "rolling_std_7d": 500.0, "day_of_month": 15, "day_of_week": 2,
            "is_mid_month": 1, "is_month_end": 0
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

def test_api_prediction_country_specific():
    with app.test_client() as client:
        payload = {
            "lag_rev_1": 5000.0, "lag_rev_2": 4800.0, "lag_rev_3": 5200.0,
            "lag_rev_4": 5100.0, "lag_rev_5": 4900.0, "lag_rev_6": 5500.0,
            "lag_rev_7": 5300.0, "rolling_mean_3d": 5000.0, "rolling_mean_7d": 5114.2,
            "rolling_std_7d": 200.0, "day_of_month": 30, "day_of_week": 4,
            "is_mid_month": 0, "is_month_end": 1
        }
        response = client.post("/predict?country=usa", json=payload)
        assert response.status_code == 200