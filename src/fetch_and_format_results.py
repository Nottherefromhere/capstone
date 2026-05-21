import os
import requests
import json
import pandas as pd

API_URL = "http://localhost:5000/predict_horizon"
LOG_URL = "http://localhost:5000/logfile"

def generate_report_markdown(country="usa", months=6):
    """Queries the live container API and outputs formatted report markdown tables."""
    print("Connecting to live pricing forecaster microservice container...")
    
    # 1. Fetch future predictions
    params = {"country": country, "months": months}
    try:
        response = requests.get(API_URL, params=params)
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code} - {response.text}")
            return
        
        data = response.json()
        forecasts = data["horizon_forecasts"]
        
        # Format predictions into markdown
        df_forecast = pd.DataFrame(forecasts)
        df_forecast.columns = ["Forecast Date", "Target Milestone", "Projected Consumption Revenue ($)"]
        
        # Clean currency display strings
        df_forecast["Projected Consumption Revenue ($)"] = df_forecast["Projected Consumption Revenue ($)"].apply(lambda x: f"${x:,.2f}")
        
        print("\n### 📊 FOR CAPSTONE SLIDES: FORECAST REPORT TABLE")
        print(f"**Analysis Scope:** {data['scope']} | **Window:** {data['months_forecasted']} Months")
        print(df_forecast.to_markdown(index=False))
        
    except requests.exceptions.ConnectionError:
        print("CRITICAL: Cannot connect to container. Verify 'sudo docker ps' shows port 5000 listening.")
        return

    # 2. Fetch the corresponding operational logging trace
    print("\nConnecting to container operational auditing channel...")
    try:
        log_response = requests.get(LOG_URL)
        if log_response.status_code == 200:
            logs = log_response.json()["logs"]
            print("\n### Proof: Recent Activity Trace")
            # Pull the final 3 log lines to verify the query was recorded
            for log_line in logs[-3:]:
                print(f"`{log_line}`")
    except Exception as e:
        print(f"Could not parse live audit logs: {str(e)}")

if __name__ == "__main__":
    generate_report_markdown(country="usa", months=6)
