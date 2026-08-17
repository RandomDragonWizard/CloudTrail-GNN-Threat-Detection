"""
graph_builder.py

Builds a NetworkX and PyTorch Geometric (PyG) graph from CloudTrail JSON logs.
Defines node types (Users/Roles, IP addresses, Actions), edge weights based on frequencies,
and extracts node feature vectors (one-hot encodings, interaction counts, time-of-day features).
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Any


class CloudTrailGraphBuilder:
    """Constructs a graph representation from AWS CloudTrail logs for PyG GCN models."""

    def __init__(self, logs_filepath: str = "cloudtrail_logs.json"):
        with open(logs_filepath, "r") as f:
            self.logs = json.load(f)
        self.df = pd.DataFrame(self.logs)
        # Parse eventTime to datetime
        self.df["parsedTime"] = pd.to_datetime(self.df["eventTime"])
        self.df["hour"] = self.df["parsedTime"].dt.hour

    def build_graph(self) -> Tuple[Data, Dict[str, Any]]:
        """
        Builds a NetworkX graph and converts it to a PyTorch Geometric Data object.
        Nodes represent:
          - Users/Roles (prefix: 'user_')
          - IP Addresses (prefix: 'ip_')
          - Actions/API calls (prefix: 'action_')
        Edges represent interactions observed in CloudTrail events, weighted by frequency.
        """
        G = nx.Graph()

        # Track statistics per entity for feature extraction
        entity_stats = {}

        # First pass: collect entities and interactions
        for _, row in self.df.iterrows():
            user_info = row["userIdentity"]
            user_id = f"user_{user_info.get('arn', user_info.get('userName', 'unknown'))}"
            ip_id = f"ip_{row['sourceIPAddress']}"
            action_id = f"action_{row['eventName']}"
            is_mal = row["is_malicious"]
            hour = row["hour"]

            # Initialize node stats
            for node_key, n_type in [(user_id, "user"), (ip_id, "ip"), (action_id, "action")]:
                if node_key not in entity_stats:
                    entity_stats[node_key] = {
                        "type": n_type,
                        "total_interactions": 0,
                        "malicious_count": 0,
                        "hours": [],
                        "unique_peers": set()
                    }

            # Update stats
            entity_stats[user_id]["total_interactions"] += 1
            entity_stats[user_id]["malicious_count"] += is_mal
            entity_stats[user_id]["hours"].append(hour)
            entity_stats[user_id]["unique_peers"].add(ip_id)
            entity_stats[user_id]["unique_peers"].add(action_id)

            entity_stats[ip_id]["total_interactions"] += 1
            entity_stats[ip_id]["malicious_count"] += is_mal
            entity_stats[ip_id]["hours"].append(hour)
            entity_stats[ip_id]["unique_peers"].add(user_id)
            entity_stats[ip_id]["unique_peers"].add(action_id)

            entity_stats[action_id]["total_interactions"] += 1
            entity_stats[action_id]["malicious_count"] += is_mal
            entity_stats[action_id]["hours"].append(hour)
            entity_stats[action_id]["unique_peers"].add(user_id)

            # Add edges with frequency weighting
            # Edge: User <-> IP
            if G.has_edge(user_id, ip_id):
                G[user_id][ip_id]["weight"] += 1
                G[user_id][ip_id]["malicious"] += is_mal
            else:
                G.add_edge(user_id, ip_id, weight=1, malicious=is_mal)

            # Edge: User <-> Action
            if G.has_edge(user_id, action_id):
                G[user_id][action_id]["weight"] += 1
                G[user_id][action_id]["malicious"] += is_mal
            else:
                G.add_edge(user_id, action_id, weight=1, malicious=is_mal)

            # Edge: IP <-> Action
            if G.has_edge(ip_id, action_id):
                G[ip_id][action_id]["weight"] += 1
                G[ip_id][action_id]["malicious"] += is_mal
            else:
                G.add_edge(ip_id, action_id, weight=1, malicious=is_mal)

        # Map nodes to contiguous integers [0, N-1]
        nodes = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        idx_to_node = {i: node for i, node in enumerate(nodes)}

        # Build Node Feature Matrix (x) and Labels (y)
        # Feature dimensions:
        # [0]: Node type one-hot (User: 1,0,0)
        # [1]: Node type one-hot (IP: 0,1,0)
        # [2]: Node type one-hot (Action: 0,0,1)
        # [3]: Logarithm of total interactions (normalized)
        # [4]: Mean hour of activity (normalized by 24.0)
        # [5]: Standard deviation of hour of activity
        # [6]: Number of unique connected peers (normalized)
        # [7]: Ratio of malicious interactions involving this node
        
        feature_list = []
        labels = []

        max_interactions = max([st["total_interactions"] for st in entity_stats.values()]) or 1
        max_peers = max([len(st["unique_peers"]) for st in entity_stats.values()]) or 1

        for node in nodes:
            st = entity_stats[node]
            n_type = st["type"]
            
            # One-hot type encoding
            type_enc = [
                1.0 if n_type == "user" else 0.0,
                1.0 if n_type == "ip" else 0.0,
                1.0 if n_type == "action" else 0.0,
            ]

            total_inter = float(st["total_interactions"])
            norm_inter = np.log1p(total_inter) / np.log1p(max_interactions)

            hours = st["hours"]
            mean_hour = np.mean(hours) / 24.0 if hours else 0.0
            std_hour = (np.std(hours) / 24.0) if len(hours) > 1 else 0.0

            norm_peers = len(st["unique_peers"]) / max_peers
            mal_ratio = st["malicious_count"] / total_inter if total_inter > 0 else 0.0

            feat = type_enc + [norm_inter, mean_hour, std_hour, norm_peers, mal_ratio]
            feature_list.append(feat)

            # Node is labeled malicious (1) if > 30% of its interactions were malicious, else 0
            label = 1 if mal_ratio > 0.3 else 0
            labels.append(label)

        x = torch.tensor(feature_list, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)

        # Build edge_index and edge_attr
        edge_indices = []
        edge_attrs = []

        for u, v, data in G.edges(data=True):
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            weight = float(data.get("weight", 1))
            
            # Add undirected edge pairs for GCN message passing
            edge_indices.append([u_idx, v_idx])
            edge_attrs.append([weight])
            edge_indices.append([v_idx, u_idx])
            edge_attrs.append([weight])

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        # Create Train / Val / Test masks
        num_nodes = x.size(0)
        indices = torch.randperm(num_nodes)
        train_size = int(0.7 * num_nodes)
        val_size = int(0.15 * num_nodes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )

        metadata = {
            "node_to_idx": node_to_idx,
            "idx_to_node": idx_to_node,
            "entity_stats": entity_stats,
            "num_nodes": num_nodes,
            "num_edges": edge_index.size(1) // 2
        }

        print(f"[+] Graph constructed: {metadata['num_nodes']} nodes, {metadata['num_edges']} undirected edges.")
        return data, metadata


if __name__ == "__main__":
    builder = CloudTrailGraphBuilder("cloudtrail_logs.json")
    data, meta = builder.build_graph()
    print(data)
