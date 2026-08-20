# CloudTrail GNN Threat Detection & Anomaly Detection

This repository implements a Graph Neural Network (GNN) pipeline for threat hunting and anomaly detection on AWS CloudTrail log datasets (specifically the **flaws.cloud** dataset containing ~1.94 million log records).

By structuring User-Agent clients and API invocation patterns into a K-partite graph, a Graph Convolutional Network (GCN) Autoencoder learns normal behavior distributions and identifies suspicious offensive tooling, reconnaissance scanners (e.g. Kali Linux, CloudSploit, custom Boto3/CLI loops), and abnormal AWS API call patterns.

---

## 📁 Repository Structure

```text
CloudTrail-GNN-Threat-Detection/
├── requirements.txt            # Required Python packages
├── README.md                   # Setup and execution guide
└── code/
    ├── datasets/
    │   └── generate_user_agent_matrix.py # Log parsing & cross-tabulation script
    ├── build_graph.py          # K-partite Agent graph builder ($k$-NN API profile similarity)
    ├── gcn_anomaly_detection.py# GCN Autoencoder model training & anomaly scoring script
    ├── user_agent_api_matrix.csv# Generated User-Agent x API call frequency matrix
    └── ua_gcn_anomaly_scores.csv # Output anomaly scores per User-Agent node
```

---

## 🛠️ Prerequisites & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/RandomDragonWizard/CloudTrail-GNN-Threat-Detection.git
cd CloudTrail-GNN-Threat-Detection
```

### 2. Install Dependencies
Ensure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

---

## 🚀 Execution Pipeline

### Step 1: Parse Logs & Generate Matrix (Optional if using pre-generated CSV)
If starting from raw CloudTrail JSON logs (e.g. `flaws_cloudtrail*.json`):
```bash
python code/datasets/generate_user_agent_matrix.py
```
*Output*: Generates `code/user_agent_api_matrix.csv` (8,812 User-Agents x 1,382 API Calls).

### Step 2: Build K-Partite Agent Node Graph
Constructs a $k$-NN sparse graph where nodes represent User-Agents and edge weights represent API invocation profile similarities:
```bash
python code/build_graph.py
```
*Output*: Saves `code/ua_graph_data.pt` containing 8,812 nodes and ~175k weighted edges.

### Step 3: Train GCN Autoencoder & Detect Anomalies
Trains a 2-layer Graph Convolutional Autoencoder (`GCNConv`) to reconstruct node feature distributions and calculate anomaly scores:
```bash
python code/gcn_anomaly_detection.py
```
*Output*: Exports `code/ua_gcn_anomaly_scores.csv` ranking all User-Agents by reconstruction error.

---

## 📊 Model Architecture & Node Features

1. **Graph Representation**:
   - **Nodes**: 8,812 unique `userAgent` strings.
   - **Edge Weights**: Cosine similarity of API call vectors across 1,382 distinct AWS API endpoints (`eventSource:eventName`).
2. **Node Features**:
   - `log_total_calls`: $\ln(1 + \text{total\_volume})$
   - `out_degree`: Count of unique APIs invoked.
   - `entropy`: Shannon entropy of API call distribution (measures diversity of calls).
   - `max_call`: Maximum call frequency to a single API endpoint.
3. **GCN Autoencoder**:
   - **Encoder**: 2-layer GCNConv ($4 \rightarrow 16 \rightarrow 8$)
   - **Decoder**: Feature reconstruction MLP + Inner Product structural link decoder.
   - **Anomaly Score**: Mean Squared Error (MSE) of standardized feature reconstruction.

---

## 📜 License
MIT License.
