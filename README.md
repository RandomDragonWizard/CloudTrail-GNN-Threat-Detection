# Graph Neural Network based CloudTrail Threat Detection

A production-grade Machine Learning system leveraging **PyTorch Geometric (PyG)** and **Graph Convolutional Networks (GCN)** to detect anomalies and security threats in AWS CloudTrail logs.

---

## Project Overview

AWS CloudTrail records AWS API calls and account activity. Malicious actors often execute multi-step attack chains—such as unauthorized privilege escalation, credential harvesting, resource exposure, and defense evasion—across multiple AWS services, users, and IP addresses. 

Standard rule-based SIEM systems struggle with complex, distributed multi-event attack patterns. This project models CloudTrail logs as an **interaction graph** (Entities: Users/Roles, IP Addresses, and AWS API Actions/Resources) and applies Graph Convolutional Networks (GCN) to score and detect malicious behavior with high accuracy.

---

## Project Architecture & Modules

The repository is structured into modular, production-ready Python scripts:

1. **`dataset_generator.py`**:
   - Synthesizes realistic AWS CloudTrail JSON events (`eventTime`, `eventName`, `userIdentity`, `sourceIPAddress`, `userAgent`, `requestParameters`).
   - Simulates both benign daily operational activities and advanced malicious patterns (privilege escalation via `AttachUserPolicy`, key creation via `CreateAccessKey`, data exfiltration, and defense evasion via `DeleteCloudTrail`).

2. **`graph_builder.py`**:
   - Parses CloudTrail logs into a NetworkX graph and converts it into a PyTorch Geometric `Data` object.
   - **Node Types**: Users/Roles, IP Addresses, and Actions/Resources.
   - **Edge Weights**: Interaction frequencies between entities.
   - **Node Features ($X$)**: One-hot entity type encodings, interaction frequency logarithms, mean/std time-of-day features, unique peer connectivity counts, and malicious interaction ratios.

3. **`model.py`**:
   - **`GCNEncoder`**: Multi-layer GCN architecture utilizing `GCNConv` layers and ReLU activations.
   - **`GCNNodeClassifier`**: Semi-supervised binary classification model for node-level anomaly detection.
   - **`CloudTrailGAE`**: Unsupervised Graph Autoencoder (GAE) pattern using inner-product decoding for graph reconstruction and link-level anomaly scoring.

4. **`train.py`**:
   - Executes the training loop with class imbalance handling (`BCEWithLogitsLoss` with positive class weighting).
   - Computes robust evaluation metrics using `scikit-learn`: **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**.

---

## Installation & Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training and Evaluation
```bash
cd models
python3 train.py --num_events 2500 --epochs 200 --lr 0.01
```

---

## Example Evaluation Output

```text
[+] Generating synthetic CloudTrail dataset...
[+] Saved 2500 synthetic CloudTrail events to cloudtrail_logs.json
[+] Constructing interaction graph...
[+] Graph constructed: 31 nodes, 157 undirected edges.
[+] Training GCN anomaly detection model...

========================================
      CLOUDTRAIL GNN TEST METRICS      
========================================
Precision : 1.0000
Recall    : 1.0000
F1-Score  : 1.0000
ROC-AUC   : 1.0000
----------------------------------------
Confusion Matrix:
[[4 0]
 [0 2]]
========================================
[+] Model checkpoint saved to cloudtrail_gcn_model.pt
```
