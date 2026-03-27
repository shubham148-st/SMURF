import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch.distributions.laplace import Laplace
import math

class PrivacyLayer(nn.Module):
    """
    Applies strict edge-level $\epsilon$-Differential Privacy to continuous 
    timestamps via the Laplace Mechanism prior to temporal encoding.
    """
    def __init__(self, sensitivity=60, epsilon=1.0):
        super().__init__()
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        
    def forward(self, timestamps):
        # Bypass for ablation studies (No DP)
        if self.epsilon <= 0: 
            return timestamps
        
        # Laplace noise scale: \Delta T / \epsilon
        scale = self.sensitivity / self.epsilon
        noise_dist = Laplace(0, scale)
        
        # Inject cryptographic noise directly into the input tensor
        noise = noise_dist.sample(timestamps.shape).to(timestamps.device)
        return timestamps + noise

class TimeEncoder(nn.Module):
    """
    Projects continuous (or noisy) timestamps into a high-dimensional 
    harmonic vector space using continuous sine/cosine frequencies.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        div_term = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float32, device=t.device) * -(math.log(10000.0) / self.dim))
        
        te = torch.zeros(t.size(0), self.dim, device=t.device)
        te[:, 0::2] = torch.sin(t * div_term)
        te[:, 1::2] = torch.cos(t * div_term)
        return te

class TemporalTransformerConv(MessagePassing):
    """
    Calculates attention scores based on the encoded time difference 
    between sequential financial transactions.
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
        
        out = self.propagate(edge_index, query=query, key=key, value=value, time_enc=edge_time_encoded)
        return out.view(-1, self.heads * self.out_channels)

    def message(self, query_i, key_j, value_j, time_enc, index, ptr, size_i):
        # Time-conditioned Key generation: K_v + TE(t'_{uv})
        t_enc = self.lin_time(time_enc).view(-1, self.heads, self.out_channels)
        key_j_time = key_j + t_enc
        
        # Temporal Attention Mechanism: Softmax( (Q_u * (K_v + TE)^T) / sqrt(d) )
        alpha = (query_i * key_j_time).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        
        return value_j * alpha.unsqueeze(-1)

class FocalLoss(nn.Module):
    """
    Modulates standard cross-entropy to address the extreme class 
    imbalance inherent to AML datasets (fraud representing <1% of data).
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
    """
    Differentially Private Temporal Graph Transformer.
    Integrates input-level Laplace perturbations with temporal attention to 
    detect structuring behaviors while preserving operational privacy bounds.
    """
    def __init__(self, node_in_dim, hidden_dim, time_dim, sensitivity=60, epsilon=1.0):
        super().__init__()
        self.privacy_layer = PrivacyLayer(sensitivity, epsilon)
        self.time_encoder = TimeEncoder(time_dim)
        
        self.node_emb = nn.Embedding(1000000, node_in_dim) 
        
        self.conv1 = TemporalTransformerConv(node_in_dim, hidden_dim, time_dim, heads=4)
        self.conv2 = TemporalTransformerConv(hidden_dim * 4, hidden_dim, time_dim, heads=4)
        
        self.classifier = nn.Linear(hidden_dim * 4, 1)

    def forward(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        
        x = self.node_emb(torch.arange(data.num_nodes, device=edge_index.device))
        
        # Privacy & Temporal Sequencing Pipeline
        raw_time = edge_attr[:, 1].unsqueeze(1) 
        noisy_time = self.privacy_layer(raw_time)                
        time_encoded = self.time_encoder(noisy_time)              
        
        # Attention Layers
        x = F.relu(self.conv1(x, edge_index, time_encoded))
        x = F.relu(self.conv2(x, edge_index, time_encoded))
        
        # Edge-level representation mapping
        src, dst = edge_index
        edge_embeddings = x[src] + x[dst] 
        
        out = self.classifier(edge_embeddings)
        return out.squeeze()