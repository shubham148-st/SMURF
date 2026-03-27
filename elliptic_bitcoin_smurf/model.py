import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch.distributions.laplace import Laplace
import math


class PrivacyLayer(nn.Module):
    """
    Applies the Laplace Mechanism to timestamps before the model sees them.
    """
    def __init__(self, sensitivity=60, epsilon=1.0):
        super().__init__()
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        
    def forward(self, timestamps):
        if self.epsilon <= 0: # No Differential Privacy
            return timestamps
        
        # Laplace noise scale: \Delta T / \epsilon
        scale = self.sensitivity / self.epsilon
        noise_dist = Laplace(0, scale)
        
        # Sample noise and add to true timestamps
        noise = noise_dist.sample(timestamps.shape).to(timestamps.device)
        return timestamps + noise


class TimeEncoder(nn.Module):
    """
    Maps the noisy timestamp to a vector using sine/cosine encodings.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t shape: [num_edges, 1]
        div_term = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float32, device=t.device) * -(math.log(10000.0) / self.dim))
        
        te = torch.zeros(t.size(0), self.dim, device=t.device)
        te[:, 0::2] = torch.sin(t * div_term)
        te[:, 1::2] = torch.cos(t * div_term)
        return te

class TemporalTransformerConv(MessagePassing):
    """
    Calculates attention scores based on the time difference between transactions.
    """
    def __init__(self, in_channels, out_channels, time_dim, heads=1):
        super().__init__(aggr='add', node_dim=0)
        self.heads = heads
        self.out_channels = out_channels
        
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
        self.lin_key = nn.Linear(in_channels, heads * out_channels)
        self.lin_value = nn.Linear(in_channels, heads * out_channels)
        self.lin_time = nn.Linear(time_dim, heads * out_channels)

    def forward(self, x, edge_index, edge_time_encoded):
        query = self.lin_query(x).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x).view(-1, self.heads, self.out_channels)
        value = self.lin_value(x).view(-1, self.heads, self.out_channels)
        
        # Start message passing
        out = self.propagate(edge_index, query=query, key=key, value=value, time_enc=edge_time_encoded)
        return out.view(-1, self.heads * self.out_channels)

    def message(self, query_i, key_j, value_j, time_enc, index, ptr, size_i):
        # Incorporate time into keys: K_v + TE(t'_{uv})
        t_enc = self.lin_time(time_enc).view(-1, self.heads, self.out_channels)
        key_j_time = key_j + t_enc
        
        # Attention Mechanism: Softmax( (Q_u * (K_v + TE)^T) / sqrt(d) )
        alpha = (query_i * key_j_time).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        
        return value_j * alpha.unsqueeze(-1)


class FocalLoss(nn.Module):
    """
    Handles extreme class imbalance (fraud is rare).
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class PrivateSmurf(nn.Module):
    def __init__(self, node_in_dim, hidden_dim, time_dim, sensitivity=60, epsilon=1.0):
        super().__init__()
        self.privacy_layer = PrivacyLayer(sensitivity, epsilon)
        self.time_encoder = TimeEncoder(time_dim)
        
        # We use a simple embedding for the initial node features since AML data 
        # usually lacks initial node features (we only have edge/transaction features)
        self.node_emb = nn.Embedding(1000000, node_in_dim) 
        
        self.conv1 = TemporalTransformerConv(node_in_dim, hidden_dim, time_dim, heads=4)
        self.conv2 = TemporalTransformerConv(hidden_dim * 4, hidden_dim, time_dim, heads=4)
        
        self.classifier = nn.Linear(hidden_dim * 4, 1)

    def forward(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        
        # Generate dummy node features if none exist
        num_nodes = data.num_nodes
        x = self.node_emb(torch.arange(num_nodes, device=edge_index.device))
        
        # Extract Normalized_Timestamp (Column 1 in edge_attr)
        raw_time = edge_attr[:, 1].unsqueeze(1) 
        
        # Apply Equations
        noisy_time = self.privacy_layer(raw_time)                 # t'_{uv}
        time_encoded = self.time_encoder(noisy_time)              # TE(t'_{uv})
        
        # Pass through Transformer layers
        x = F.relu(self.conv1(x, edge_index, time_encoded))
        x = F.relu(self.conv2(x, edge_index, time_encoded))
        
        # For edge classification (predicting if a transaction is fraud), 
        # we concatenate the source and destination node embeddings
        src, dst = edge_index
        edge_embeddings = x[src] + x[dst] # Simple aggregation for the edge
        
        out = self.classifier(edge_embeddings)
        return out.squeeze()