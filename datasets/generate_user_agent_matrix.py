"""
UserAgent-API Matrix Generator Script
Parses CloudTrail log JSON files and generates a frequency cross-tabulation matrix
(userAgent vs. service:eventName) exported to user_agent_api_matrix.csv.
"""

import os
import glob
import json
import time
import pandas as pd

def generate_user_agent_api_matrix(log_dir=None, output_csv="user_agent_api_matrix.csv"):
    start_time = time.time()
    
    # Auto-detect log directory candidates
    candidates = [
        log_dir,
        os.path.join("datasets", "flaws_cloudtrail_logs"),
        os.path.join("..", "datasets", "flaws_cloudtrail_logs"),
        r"C:\Users\Piyush\Desktop\Presidency Documents\SEM_7\CAPSTONE_Proj\datasets\flaws_cloudtrail_logs",
        "."
    ]
    
    valid_dir = None
    json_files = []
    
    for cand in candidates:
        if cand and os.path.exists(cand):
            if os.path.isdir(cand):
                files = glob.glob(os.path.join(cand, "*.json"))
                if files:
                    valid_dir = cand
                    json_files = sorted(files)
                    break
            elif cand.endswith(".json"):
                json_files = [cand]
                break

    if not json_files:
        print("No CloudTrail JSON log files found!")
        return None

    print(f"Found {len(json_files)} JSON log files. Extracting userAgent and API call records...")
    
    records = []
    for idx, filepath in enumerate(json_files):
        t0 = time.time()
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                file_records = data.get("Records", [])
                for r in file_records:
                    if isinstance(r, dict):
                        ua = r.get("userAgent", "Unknown_UA") or "Unknown_UA"
                        evt_name = r.get("eventName", "Unknown_Event") or "Unknown_Event"
                        evt_src = r.get("eventSource", "Unknown_Source") or "Unknown_Source"
                        api_call = f"{evt_src}:{evt_name}"
                        records.append({"userAgent": ua, "api_call": api_call})
        except Exception as e:
            print(f"Error reading {os.path.basename(filepath)}: {e}")
            
        print(f"[{idx+1}/{len(json_files)}] Read {os.path.basename(filepath)} in {time.time() - t0:.2f}s")

    print(f"\nTotal extracted records: {len(records):,}. Generating cross-tabulation matrix...")
    df = pd.DataFrame(records)
    
    # Build UserAgent x API Call Frequency Matrix
    matrix_df = pd.crosstab(index=df["userAgent"], columns=df["api_call"])
    
    # Save Matrix to CSV
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    matrix_df.to_csv(output_csv)
    
    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Matrix saved to: {os.path.abspath(output_csv)}")
    print(f"Matrix Shape: {matrix_df.shape[0]} UserAgents x {matrix_df.shape[1]} API Calls")
    print(f"Completed in {elapsed:.2f} seconds.")
    
    return matrix_df

if __name__ == "__main__":
    generate_user_agent_api_matrix()
