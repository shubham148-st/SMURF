import torch
import torch.nn as nn
import torch.nn.functional as F

from model import PrivacyLayer, TimeEncoder, TemporalTransformerConv, FocalLoss

class PrivateSmurfNode(nn.Module):
    """
    Cross-Domain Differential Privacy Adaptation.
    Transfers the edge-level temporal DP mechanism to a purely node-centric 
    classification task, utilizing dense 165-dimensional local features.
    """
    def __init__(self, node_in_dim=165, hidden_dim=64, time_dim=16, sensitivity=1.0, epsilon=1.0):
        super().__init__()
        self.privacy_layer = PrivacyLayer(sensitivity, epsilon)
        self.time_encoder = TimeEncoder(time_dim)
        
        # Initial projection mapping for rich local node features
        self.feature_proj = nn.Linear(node_in_dim, hidden_dim) 
        
        self.conv1 = TemporalTransformerConv(hidden_dim, hidden_dim, time_dim, heads=4)
        self.conv2 = TemporalTransformerConv(hidden_dim * 4, hidden_dim, time_dim, heads=4)
        
        # Linear classifier mapped directly to the node embeddings
        self.classifier = nn.Linear(hidden_dim * 4, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = F.relu(self.feature_proj(x))
        
        # Inject Laplace noise into timestamps prior to encoding
        raw_time = edge_attr[:, 1].unsqueeze(1) 
        noisy_time = self.privacy_layer(raw_time)
        time_encoded = self.time_encoder(noisy_time)
        
        # Temporal Attention Message Passing
        x = F.relu(self.conv1(x, edge_index, time_encoded))
        x = F.relu(self.conv2(x, edge_index, time_encoded))
        
        out = self.classifier(x)
        return out.squeeze()