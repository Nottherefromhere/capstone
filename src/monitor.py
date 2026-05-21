import os

LOG_FILE_PATH = "logs/api_operational_drift.log"

def profile_operational_performance_and_drift():
    if not os.path.exists(LOG_FILE_PATH):
        print("Monitoring Diagnostic: No activity log trace found yet.")
        return
        
    print("--- Real-Time System Monitoring Report ---")
    with open(LOG_FILE_PATH, "r") as f:
        log_lines = f.readlines()
        
    total_calls = len([line for line in log_lines if "PREDICT" in line])
    failed_calls = len([line for line in log_lines if "FAILED" in line or "ERROR" in line])
    success_calls = len([line for line in log_lines if "STATUS: SUCCESS" in line])
    
    error_rate = (failed_calls / total_calls * 100) if total_calls > 0 else 0.0
    
    print(f"Total Inference Workloads Handled : {total_calls}")
    print(f"Pipeline Request Success Pool     : {success_calls}")
    print(f"Runtime Operational Error Rate    : {error_rate:.2f}%")

if __name__ == "__main__":
    profile_operational_performance_and_drift()