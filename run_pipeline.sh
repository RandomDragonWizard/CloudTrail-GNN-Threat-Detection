#!/bin/bash

# Define paths
# Use absolute path to the project root
PROJECT_ROOT="$(pwd)"
OUTPUT_DIR="$PROJECT_ROOT/output"
CODE_DIR="$PROJECT_ROOT/code"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Clean up any pre-existing output files before starting
rm -f "$OUTPUT_DIR/user_agent_api_matrix.csv"
rm -f "$OUTPUT_DIR/ua_graph_data.pt"
rm -f "$OUTPUT_DIR/ua_gcn_anomaly_scores.csv"
rm -f "$OUTPUT_DIR/ip_api_matrix.csv"
rm -f "$OUTPUT_DIR/ip_graph_data.pt"
rm -f "$OUTPUT_DIR/ip_gcn_anomaly_scores.csv"

# Run User-Agent Pipeline
echo "Running User-Agent Pipeline..."
python "$CODE_DIR/datasets/generate_user_agent_matrix.py"
python "$CODE_DIR/build_graph.py"
python "$CODE_DIR/gcn_anomaly_detection.py"

# Run IP Pipeline
echo "Running IP Pipeline..."
python "$CODE_DIR/datasets/generate_ip_api_matrix.py"
python "$CODE_DIR/build_graph_ip.py"
python "$CODE_DIR/gcn_anomaly_detection_ip.py"

# Cleanup function to remove files that might be in code/ but should be in output/
cleanup() {
    echo "Cleaning up generated files from $CODE_DIR..."
    # These are the files known to be generated in code/ by mistake
    rm -f "$CODE_DIR/user_agent_api_matrix.csv"
    rm -f "$CODE_DIR/ua_graph_data.pt"
    rm -f "$CODE_DIR/ua_gcn_anomaly_scores.csv"
    rm -f "$CODE_DIR/ip_api_matrix.csv"
    rm -f "$CODE_DIR/ip_graph_data.pt"
    rm -f "$CODE_DIR/ip_gcn_anomaly_scores.csv"
    
    # Also clean up the actuall_code subdirectory if files were generated there
    rm -f "$CODE_DIR/actuall_code/ua_graph_data.pt"
    rm -f "$CODE_DIR/actuall_code/ip_graph_data.pt"
}

# Clean up
cleanup

echo "Pipeline complete. Outputs are in the $OUTPUT_DIR/ folder."
