"""
train.py

Training and evaluation loop for CloudTrail Graph Neural Network (GCN) threat detection.
Computes metrics: Precision, Recall, F1-Score, and ROC-AUC using scikit-learn.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

from dataset_generator import CloudTrailDatasetGenerator
from graph_builder import CloudTrailGraphBuilder
from model import GCNNodeClassifier


def train_model(
    data,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    hidden_channels: int = 64,
    out_channels: int = 32,
    dropout: float = 0.3
):
    """Trains the GCN node classification model and evaluates performance metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using device: {device}")

    # Move data to device
    data = data.to(device)

    # Initialize model
    in_channels = data.x.size(1)
    model = GCNNodeClassifier(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=dropout
    ).to(device)

    # Optimizer and Loss function with class weighting for imbalanced threat data
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Compute class weights for handling class imbalance (benign vs malicious nodes)
    train_labels = data.y[data.train_mask]
    num_pos = (train_labels == 1).sum().item()
    num_neg = (train_labels == 0).sum().item()
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = 0.0
    best_model_state = None

    print("[+] Starting training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask].float())
        
        loss.backward()
        optimizer.step()

        # Validation evaluation
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_logits = model(data.x, data.edge_index)
                val_probs = torch.sigmoid(val_logits[data.val_mask]).cpu().numpy()
                val_true = data.y[data.val_mask].cpu().numpy()

                try:
                    val_auc = roc_auc_score(val_true, val_probs)
                except ValueError:
                    val_auc = 0.0

            if val_auc >= best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()

            if epoch % 50 == 0:
                print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val ROC-AUC: {val_auc:.4f}")

    # Load best model for testing
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("\n[+] Evaluating best model on Test set...")
    model.eval()
    with torch.no_grad():
        test_logits = model(data.x, data.edge_index)
        test_probs = torch.sigmoid(test_logits[data.test_mask]).cpu().numpy()
        test_preds = (test_probs >= 0.5).astype(int)
        test_true = data.y[data.test_mask].cpu().numpy()

        # Compute evaluation metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_true, test_preds, average="binary", zero_division=0
        )
        try:
            roc_auc = roc_auc_score(test_true, test_probs)
        except ValueError:
            roc_auc = 0.0

        cm = confusion_matrix(test_true, test_preds)

    print("========================================")
    print("      CLOUDTRAIL GNN TEST METRICS      ")
    print("========================================")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print("----------------------------------------")
    print("Confusion Matrix:")
    print(cm)
    print("========================================")

    return model, {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


def main():
    parser = argparse.ArgumentParser(description="Train GCN Threat Detection Model on CloudTrail Logs")
    parser.add_argument("--num_events", type=int, default=2500, help="Number of synthetic CloudTrail events")
    parser.add_argument("--malicious_ratio", type=float, default=0.15, help="Ratio of malicious events")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()

    # Step 1: Generate dataset
    print("[1/3] Generating synthetic CloudTrail dataset...")
    generator = CloudTrailDatasetGenerator(seed=42)
    logs = generator.generate_dataset(num_events=args.num_events, malicious_ratio=args.malicious_ratio)
    generator.save_to_json(logs, "cloudtrail_logs.json")

    # Step 2: Build Graph
    print("\n[2/3] Constructing interaction graph...")
    builder = CloudTrailGraphBuilder("cloudtrail_logs.json")
    data, metadata = builder.build_graph()

    # Step 3: Train & Evaluate Model
    print("\n[3/3] Training GCN anomaly detection model...")
    model, metrics = train_model(data, epochs=args.epochs, lr=args.lr)

    # Save trained model checkpoint
    torch.save(model.state_dict(), "cloudtrail_gcn_model.pt")
    print("[+] Model checkpoint saved to cloudtrail_gcn_model.pt")


if __name__ == "__main__":
    main()
