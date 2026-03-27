import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import SAGEConv, GATConv, MessagePassing
from torch_geometric.utils import softmax


class TGATTimeEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        div_term = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float32, device=t.device) * -(math.log(10000.0) / self.dim))
        te = torch.zeros(t.size(0), self.dim, device=t.device)
        te[:, 0::2] = torch.sin(t * div_term)
        te[:, 1::2] = torch.cos(t * div_term)
        return te

class TGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, time_dim, heads=4):
        super().__init__(aggr='add', node_dim=0)
        self.heads = heads
        self.out_channels = out_channels
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
        self.lin_key = nn.Linear(in_channels, heads * out_channels)
        self.lin_value = nn.Linear(in_channels, heads * out_channels)
        self.lin_time = nn.Linear(time_dim, heads * out_channels)

    def forward(self, x, edge_index, edge_time):
        query = self.lin_query(x).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x).view(-1, self.heads, self.out_channels)
        value = self.lin_value(x).view(-1, self.heads, self.out_channels)
        out = self.propagate(edge_index, query=query, key=key, value=value, time_enc=edge_time)
        return out.view(-1, self.heads * self.out_channels)

    def message(self, query_i, key_j, value_j, time_enc, index, ptr, size_i):
        t_enc = self.lin_time(time_enc).view(-1, self.heads, self.out_channels)
        key_j_time = key_j + t_enc
        alpha = (query_i * key_j_time).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        return value_j * alpha.unsqueeze(-1)


# --- NODE-LEVEL SPATIAL/TEMPORAL BASELINES ---
class GraphSAGENodeBaseline(nn.Module):
    """Adapts static topological aggregation for Node Classification tasks."""
    def __init__(self, node_in_dim=165, hidden_dim=64):
        super().__init__()
        self.feature_proj = nn.Linear(node_in_dim, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.feature_proj(x))
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.classifier(x).squeeze()

class GATNodeBaseline(nn.Module):
    """Adapts static spatial attention for Node Classification tasks."""
    def __init__(self, node_in_dim=165, hidden_dim=64, heads=4):
        super().__init__()
        self.feature_proj = nn.Linear(node_in_dim, hidden_dim)
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.feature_proj(x))
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.classifier(x).squeeze()

class TGATNodeBaseline(nn.Module):
    """Adapts standard temporal attention for Node Classification tasks."""
    def __init__(self, node_in_dim=165, hidden_dim=64, time_dim=16):
        super().__init__()
        self.feature_proj = nn.Linear(node_in_dim, hidden_dim)
        self.time_encoder = TGATTimeEncoder(time_dim)
        
        self.conv1 = TGATConv(hidden_dim, hidden_dim, time_dim, heads=4)
        self.conv2 = TGATConv(hidden_dim * 4, hidden_dim, time_dim, heads=4)
        self.classifier = nn.Linear(hidden_dim * 4, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.feature_proj(x))
        
        raw_time = edge_attr[:, 1].unsqueeze(1)
        time_encoded = self.time_encoder(raw_time)
        
        x = F.relu(self.conv1(x, edge_index, time_encoded))
        x = F.relu(self.conv2(x, edge_index, time_encoded))
        return self.classifier(x).squeeze()