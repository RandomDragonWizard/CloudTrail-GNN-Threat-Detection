# CloudTrail GNN Threat Detection & Anomaly Detection

This repository implements a Graph Neural Network (GNN) pipeline for threat hunting and anomaly detection on AWS CloudTrail log datasets.

By structuring User-Agent clients and API invocation patterns into a K-partite graph, a Graph Convolutional Network (GCN) Autoencoder learns normal behavior distributions and identifies suspicious offensive tooling, reconnaissance scanners, and abnormal AWS API call patterns.

---

## 📁 Repository Structure

```text
CloudTrail-GNN-Threat-Detection/
├── code/               # Pipeline scripts (parsing, graph building, training)
├── flaws_cloudtrail_logs/ # Directory for raw CloudTrail JSON logs
├── output/             # Generated matrices, graphs, and anomaly reports
├── requirements.txt    # Python dependencies
├── run_pipeline.sh     # Main execution script
└── README.md           # This file
```

---

## 🚀 Setup & Execution

### 1. Prerequisites
- Python 3.9+
- AWS CloudTrail JSON logs placed in the `flaws_cloudtrail_logs/` directory.

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/RandomDragonWizard/CloudTrail-GNN-Threat-Detection.git
cd CloudTrail-GNN-Threat-Detection

# Install Dependencies
pip install -r requirements.txt
```

### 3. Running the Pipeline
Use the provided shell script to run the full end-to-end pipeline. This handles parsing, graph construction, training, and automatic cleanup of intermediate files.

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

*Output*: All results (CSV matrices, graph data, and anomaly score reports) will be saved in the `output/` directory.

---

## 📜 License
MIT License.
