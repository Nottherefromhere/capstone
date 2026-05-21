import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# Core production file directory targets
MODEL_PATH = "final_revenue_forecaster.joblib"
LOG_FILE_PATH = "logs/api_operational_drift.log"
DATA_DIR = "data"

os.makedirs("logs", exist_ok=True)

def append_to_logfile(event_type, status, message):
    """Writes standardized operational metadata metrics for drift auditing."""
    timestamp = datetime.utcnow().isoformat()
    log_entry = f"[{timestamp}] | EVENT: {event_type} | STATUS: {status} | MSG: {message}\n"
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(log_entry)

@app.route("/train", methods=["POST"])
def trigger_training_pipeline():
    """Triggers autonomous data ingestion, profiling, and model retraining optimization."""
    try:
        from src.main import train_and_deploy_best_model
        model_name, mae = train_and_deploy_best_model(DATA_DIR)
        
        append_to_logfile("TRAIN", "SUCCESS", f"Model updated using {model_name}. MAE: ${mae:.2f}")
        return jsonify({"status": "training complete", "model_selected": model_name, "mae": round(mae, 2)}), 200
    except Exception as e:
        append_to_logfile("TRAIN", "FAILED", f"Retraining task crash: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def generate_revenue_forecast():
    """Executes live single-step matrix feature scoring against your saved model artifact."""
    if not os.path.exists(MODEL_PATH):
        append_to_logfile("PREDICT", "ERROR", "Model binary missing on file disk.")
        return jsonify({"error": "Model binary not loaded on server instance."}), 503
    
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "Invalid or missing JSON payload structure."}), 400
        
    country = request.args.get("country")
    
    try:
        model = joblib.load(MODEL_PATH)
        input_data = pd.DataFrame([payload])
        prediction = model.predict(input_data)
        final_pred = max(0.0, float(prediction))
        
        scope = f"Country Specific ({country.upper()})" if country else "Global Combined"
        append_to_logfile("PREDICT", "SUCCESS", f"Served forecast for {scope}: ${final_pred:.2f}")
        
        return jsonify({
            "projected_revenue": round(final_pred, 2), 
            "scope": scope, 
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        append_to_logfile("PREDICT", "FAILED", f"Inference pipeline crash: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict_horizon", methods=["GET"])
def get_future_horizon_forecast():
    """Exposes a dynamic API endpoint to fetch multi-month future forecasts for the 15th and 30th."""
    if not os.path.exists(MODEL_PATH):
        append_to_logfile("HORIZON", "ERROR", "Model binary missing on file disk.")
        return jsonify({"error": "Model binary not loaded on server instance."}), 503
        
    country = request.args.get("country")
    months_input = request.args.get("months", default=6, type=int)
    
    # Safety constraint validation
    if months_input <= 0 or months_input > 36:
        return jsonify({"error": "Forecast window violation. Please request between 1 and 36 months."}), 400
        
    try:
        model = joblib.load(MODEL_PATH)
        from src.main import load_all_transaction_data, generate_variable_horizon_forecast
        
        raw_data = load_all_transaction_data(DATA_DIR)
        
        # Run our newly integrated internal variable simulation engine
        forecast_results = generate_variable_horizon_forecast(
            model, raw_data, months_to_forecast=months_input, country_code=country
        )
        
        scope = f"Country Specific ({country.upper()})" if country else "Global Combined"
        append_to_logfile("HORIZON", "SUCCESS", f"Served {months_input}-month forecast horizon for {scope}")
        
        return jsonify({
            "scope": scope,
            "months_forecasted": months_input,
            "horizon_forecasts": forecast_results,
            "timestamp": datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        append_to_logfile("HORIZON", "FAILED", f"Horizon projection failure: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.get("/logfile")
def stream_operational_logs():
    """Exposes real-time tail auditing logs directly for system checking and drift scraping."""
    if not os.path.exists(LOG_FILE_PATH):
        return jsonify({"logs": []}), 200
    with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return jsonify({"logs": [line.strip() for line in lines[-100:]]}), 200
