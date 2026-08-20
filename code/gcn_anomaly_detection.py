import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Set directories
base_dir = r"C:\Users\Piyush\Desktop\Presidency Documents\SEM_7\CAPSTONE_Proj\CloudTrail-GNN-Threat-Detection\code\actuall_code"
graph_pt = os.path.join(base_dir, "ua_graph_data.pt")

if not os.path.exists(graph_pt):
    raise FileNotFoundError(f"Graph file not found at {graph_pt}. Please run build_graph.py first.")

# Load Graph Data
data_dict = torch.load(graph_pt)
x_raw = data_dict['x'].numpy()
edge_index = data_dict['edge_index']
edge_weight = data_dict['edge_weight']
user_agents = data_dict['user_agents']

# Standardize node features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_raw)
x = torch.tensor(x_scaled, dtype=torch.float)

print(f"Loaded graph: {x.shape[0]} nodes, {x.shape[1]} features, {edge_index.shape[1]} edges.")

# GCN Autoencoder Architecture
class GCNAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNAutoencoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.decoder = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels)
        )

    def encode(self, x, edge_index, edge_weight):
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.conv2(h, edge_index, edge_weight)
        return h

    def decode_features(self, z):
        return self.decoder(z)

    def decode_structure(self, z, edge_index):
        src, dst = edge_index[0], edge_index[1]
        return torch.sigmoid((z[src] * z[dst]).sum(dim=-1))

# Training parameters
in_dim = x.shape[1]
hidden_dim = 16
latent_dim = 8
epochs = 200
lr = 0.01

model = GCNAutoencoder(in_dim, hidden_dim, latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print("\n--- Training GCN Autoencoder ---")
model.train()
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    
    z = model.encode(x, edge_index, edge_weight)
    x_rec = model.decode_features(z)
    
    loss_feat = F.mse_loss(x_rec, x)
    edge_pred = model.decode_structure(z, edge_index)
    loss_struct = F.mse_loss(edge_pred, edge_weight)
    
    total_loss = loss_feat + loss_struct
    total_loss.backward()
    optimizer.step()
    
    if epoch % 25 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.4f} | Feat Loss: {loss_feat.item():.4f} | Struct Loss: {loss_struct.item():.4f}")

# Compute Anomaly Scores
model.eval()
with torch.no_grad():
    z = model.encode(x, edge_index, edge_weight)
    x_rec = model.decode_features(z)
    feat_error = torch.mean((x - x_rec) ** 2, dim=1).numpy()

df_anom = pd.DataFrame({
    'userAgent': user_agents,
    'anomaly_score': feat_error,
    'log_total_calls': x_raw[:, 0],
    'out_degree': x_raw[:, 1],
    'entropy': x_raw[:, 2],
    'max_call': x_raw[:, 3]
})

df_anom = df_anom.sort_values(by='anomaly_score', ascending=False).reset_index(drop=True)

output_csv = os.path.join(base_dir, "ua_gcn_anomaly_scores.csv")
df_anom.to_csv(output_csv, index=False)

print(f"\n[SUCCESS] Anomaly scores saved to: {output_csv}")
print("\nTop 10 Most Anomalous User-Agents:")
print(df_anom.head(10).to_string())
