"""
IP-API Frequency Matrix Generator Script
Parses CloudTrail log JSON files in sorted order (flaws_cloudtrail00.json to flaws_cloudtrail19.json),
maintains a single cumulative DataFrame throughout the loop, increments call frequencies,
adds new IP rows and API columns dynamically as they appear, and exports the final matrix to CSV.
"""

import os
import glob
import json
import time
import pandas as pd


def generate_ip_api_matrix(log_dir=None, output_csv=None):
    start_time = time.time()

    if output_csv is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        # Use root output dir
        output_dir = os.path.join(os.path.dirname(project_root), "output")
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, "ip_api_matrix.csv")

    # Auto-detect log directory candidates
    candidates = [
        log_dir,
        os.path.join(project_root, "flaws_cloudtrail_logs"),
        os.path.join("flaws_cloudtrail_logs"),
        os.path.join("datasets", "flaws_cloudtrail_logs"),
        os.path.join("..", "flaws_cloudtrail_logs"),
        ".",
    ]

    json_files = []
    for cand in candidates:
        if cand and os.path.exists(cand):
            if os.path.isdir(cand):
                files = glob.glob(os.path.join(cand, "flaws_cloudtrail*.json"))
                if not files:
                    files = glob.glob(os.path.join(cand, "*.json"))
                if files:
                    json_files = sorted(files)
                    break
            elif cand.endswith(".json"):
                json_files = [cand]
                break

    if not json_files:
        print("No CloudTrail JSON log files found!")
        return None

    print(
        f"Found {len(json_files)} JSON log files. Processing in sorted order (00, 01, ...)..."
    )

    # Keep the same dataframe throughout the process
    cumulative_df = pd.DataFrame()

    for idx, filepath in enumerate(json_files):
        t0 = time.time()
        filename = os.path.basename(filepath)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                file_records = data.get("Records", [])

                recs = []
                for r in file_records:
                    if isinstance(r, dict):
                        ip = r.get("sourceIPAddress", "Unknown_IP") or "Unknown_IP"
                        api = r.get("eventName", "Unknown_Event") or "Unknown_Event"
                        recs.append({"ip": ip, "api": api})

                if recs:
                    temp_df = pd.DataFrame(recs)
                    file_matrix = pd.crosstab(temp_df["ip"], temp_df["api"])
                    # Update cumulative DataFrame (adding new rows/columns and incrementing counts)
                    cumulative_df = cumulative_df.add(file_matrix, fill_value=0)

        except Exception as e:
            print(f"Error reading {filename}: {e}")

        print(
            f"[{idx + 1}/{len(json_files)}] Processed {filename} in {time.time() - t0:.2f}s | Current shape: {cumulative_df.shape}"
        )

    # Fill NaN with 0 and convert integer counts
    cumulative_df = cumulative_df.fillna(0).astype(int)

    # Save Matrix to CSV
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cumulative_df.to_csv(output_csv)

    elapsed = time.time() - start_time
    print(
        f"\n[SUCCESS] IP-API Frequency Matrix saved to: {os.path.abspath(output_csv)}"
    )
    print(
        f"Final Matrix Shape: {cumulative_df.shape[0]} unique IPs x {cumulative_df.shape[1]} unique APIs"
    )
    print(f"Completed in {elapsed:.2f} seconds.")

    return cumulative_df


if __name__ == "__main__":
    generate_ip_api_matrix()
