"""
model.py

Implements PyTorch Geometric Graph Convolutional Network (GCN) architectures:
1. GCNEncoder: Multi-layer GCN encoder producing node embeddings.
2. GCNNodeClassifier: Semi-supervised binary classification model for node anomaly detection.
3. GraphAutoencoder (GAE): Unsupervised autoencoder pattern for graph reconstruction and anomaly scoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GAE


class GCNEncoder(nn.Module):
    """Two-layer GCN Encoder mapping node features to latent space."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GCNNodeClassifier(nn.Module):
    """
    Semi-supervised GCN model for node-level anomaly detection (Benign vs. Malicious).
    Combines GCNEncoder with a classification head.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 16, dropout: float = 0.3):
        super(GCNNodeClassifier, self).__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, out_channels, dropout)
        self.classifier = nn.Linear(out_channels, 1)  # Binary classification logit

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x, edge_index)
        logits = self.classifier(embeddings)
        return logits.squeeze(-1)


class CloudTrailGAE(nn.Module):
    """
    Unsupervised Graph Autoencoder (GAE) for anomaly detection in CloudTrail activity graph.
    Uses GCNEncoder for node representations and inner-product decoder for edge reconstruction.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3):
        super(CloudTrailGAE, self).__init__()
        encoder = GCNEncoder(in_channels, hidden_channels, out_channels, dropout)
        self.gae = GAE(encoder)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.gae.encode(x, edge_index)

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor, sigmoid: bool = True) -> torch.Tensor:
        return self.gae.decode(z, edge_index, sigmoid=sigmoid)

    def recon_loss(self, z: torch.Tensor, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor = None) -> torch.Tensor:
        return self.gae.recon_loss(z, pos_edge_index, neg_edge_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, edge_index)
        return z
