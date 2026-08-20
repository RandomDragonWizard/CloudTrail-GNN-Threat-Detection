import os
import pandas as pd
import numpy as np
import torch
from scipy.stats import entropy

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Search for CSV in multiple common locations
possible_csv_paths = [
    os.path.join(project_root, "output", "user_agent_api_matrix.csv"),
    os.path.join(project_root, "code", "user_agent_api_matrix.csv"),
    os.path.join(project_root, "user_agent_api_matrix.csv"),
    os.path.join(script_dir, "user_agent_api_matrix.csv"),
    os.path.join(project_root, "datasets", "user_agent_api_matrix.csv"),
    os.path.join(project_root, "data", "user_agent_api_matrix.csv"),
]

csv_path = None
for path in possible_csv_paths:
    if os.path.exists(path):
        csv_path = path
        break

if csv_path is None:
    searched = "\n  ".join(possible_csv_paths)
    raise FileNotFoundError(
        f"user_agent_api_matrix.csv not found in any expected location.\n"
        f"Searched:\n  {searched}\n"
        f"Please ensure the dataset is present in the repository."
    )

output_dir = os.path.join(project_root, "output")
os.makedirs(output_dir, exist_ok=True)

print(f"Loading matrix from {csv_path}...")
df = pd.read_csv(csv_path, index_col=0)

user_agents = df.index.tolist()
api_names = df.columns.tolist()

N_ua = len(user_agents)
N_api = len(api_names)
print(f"Loaded {N_ua} User-Agent nodes and {N_api} API columns.")

matrix_vals = df.values.astype(np.float32)

# Node feature engineering per User-Agent
total_calls = matrix_vals.sum(axis=1, keepdims=True)
log_total_calls = np.log1p(total_calls)
out_degree = (matrix_vals > 0).sum(axis=1, keepdims=True).astype(np.float32)

probs = matrix_vals / np.maximum(total_calls, 1e-8)
ent = np.array([entropy(p) if np.sum(p) > 0 else 0.0 for p in probs])[:, None]
max_call = matrix_vals.max(axis=1, keepdims=True)

ua_features = np.hstack([log_total_calls, out_degree, ent, max_call])

# Compute API profile similarity for K-partite graph edge weights
norm_matrix = matrix_vals / np.maximum(
    np.linalg.norm(matrix_vals, axis=1, keepdims=True), 1e-8
)

k = 20
print(f"Building sparse top-{k} nearest neighbor Agent-Agent graph...")
similarity_matrix = np.dot(norm_matrix, norm_matrix.T)
np.fill_diagonal(similarity_matrix, 0.0)

topk_indices = np.argsort(similarity_matrix, axis=1)[:, -k:]

src_list = []
dst_list = []
weight_list = []

for i in range(N_ua):
    for neighbor in topk_indices[i]:
        weight = similarity_matrix[i, neighbor]
        if weight > 0.01:
            src_list.append(i)
            dst_list.append(neighbor)
            weight_list.append(weight)

edge_indices = np.array([src_list, dst_list], dtype=np.int64)
edge_weights = np.array(weight_list, dtype=np.float32)

print(
    f"Constructed graph with {N_ua} nodes and {edge_indices.shape[1]} weighted edges."
)

graph_output_pt = os.path.join(output_dir, "ua_graph_data.pt")
torch.save(
    {
        "x": torch.tensor(ua_features, dtype=torch.float),
        "edge_index": torch.tensor(edge_indices, dtype=torch.long),
        "edge_weight": torch.tensor(edge_weights, dtype=torch.float),
        "user_agents": user_agents,
        "api_names": api_names,
    },
    graph_output_pt,
)

print(f"[SUCCESS] Saved graph dataset to: {graph_output_pt}")
