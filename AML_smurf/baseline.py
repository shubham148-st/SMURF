import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv

class GraphSAGEBaseline(nn.Module):
    """
    GraphSAGE (Static Spatial Aggregation) Baseline.
    Evaluates topological connectivity while remaining entirely agnostic to 
    transaction timestamps. Provides a baseline for purely structural learning.
    """
    def __init__(self, node_in_dim, hidden_dim):
        super().__init__()
        # Initialize node embeddings (standard for transaction graphs lacking inherent node features)
        self.node_emb = nn.Embedding(1000000, node_in_dim) 
        self.conv1 = SAGEConv(node_in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        edge_index = data.edge_index
        x = self.node_emb(torch.arange(data.num_nodes, device=edge_index.device))
        
        # Spatial message passing
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Edge aggregation via source and destination node embedding summation
        src, dst = edge_index
        edge_embeddings = x[src] + x[dst] 
        
        out = self.classifier(edge_embeddings)
        return out.squeeze()

class GATBaseline(nn.Module):
    """
    Graph Attention Network (GAT) Baseline.
    Applies anisotropic spatial attention to node neighborhoods, prioritizing 
    highly correlated transactional pathways without temporal context.
    """
    def __init__(self, node_in_dim, hidden_dim, heads=4):
        super().__init__()
        self.node_emb = nn.Embedding(1000000, node_in_dim)
        
        # Multi-head spatial attention
        self.conv1 = GATConv(node_in_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1) 
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        edge_index = data.edge_index
        x = self.node_emb(torch.arange(data.num_nodes, device=edge_index.device))
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Aggregate local edge representations
        src, dst = edge_index
        edge_embeddings = x[src] + x[dst]
        
        out = self.classifier(edge_embeddings)
        return out.squeeze()