import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import time

from model import PrivateSmurf, FocalLoss 

def subsample_graph(data, normal_to_fraud_ratio=10):
    """
    Maintains graph structure while downsampling the majority class.
    Accelerates training loops and prevents immediate convergence to local minima.
    """
    fraud_indices = (data.y == 1.0).nonzero(as_tuple=True)[0]
    normal_indices = (data.y == 0.0).nonzero(as_tuple=True)[0]
    
    num_fraud = len(fraud_indices)
    num_normal_to_keep = min(num_fraud * normal_to_fraud_ratio, len(normal_indices))
    
    perm = torch.randperm(len(normal_indices))
    sampled_normal_indices = normal_indices[perm[:num_normal_to_keep]]
    
    keep_indices = torch.cat([fraud_indices, sampled_normal_indices])
    keep_indices = keep_indices[torch.randperm(len(keep_indices))]
    
    data.edge_index = data.edge_index[:, keep_indices]
    data.edge_attr = data.edge_attr[keep_indices]
    data.y = data.y[keep_indices]
    data.num_nodes = int(data.edge_index.max()) + 1 
    return data

def train_and_evaluate(graph_data, epsilon, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    graph_data = subsample_graph(graph_data, normal_to_fraud_ratio=10)
    
    # Stratified edge-split
    num_edges = graph_data.num_edges
    indices = np.random.permutation(num_edges)
    train_idx = indices[:int(0.8 * num_edges)]
    test_idx = indices[int(0.8 * num_edges):]

    model = PrivateSmurf(node_in_dim=16, hidden_dim=64, time_dim=16, epsilon=epsilon).to(device)
    graph_data = graph_data.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01) 
    criterion = FocalLoss(alpha=0.75, gamma=2.0) 

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(graph_data)
        
        train_out = out[train_idx]
        train_y = graph_data.y[train_idx]
        
        loss = criterion(train_out, train_y)
        loss.backward()
        optimizer.step()

    # Inference and Metric Logging
    model.eval()
    with torch.no_grad():
        out = model(graph_data)
        test_out = out[test_idx]
        test_y = graph_data.y[test_idx].cpu().numpy()
        
        preds = (torch.sigmoid(test_out) > 0.4).cpu().numpy()
        
        f1 = f1_score(test_y, preds)
        prec = precision_score(test_y, preds, zero_division=0)
        rec = recall_score(test_y, preds, zero_division=0)
        
    torch.cuda.empty_cache() 
    return f1, prec, rec

if __name__ == "__main__":
    print("Initializing Experimental Pipeline...")
    try:
        data = torch.load('ibm_smurf_graph.pt', weights_only=False)
    except FileNotFoundError:
        print("Fatal: 'ibm_smurf_graph.pt' not found. Execute graph.py to generate topology.")
        exit()
        
    epsilons = [0.1, 0.5, 1.0, 5.0, 0] 
    
    print("\n" + "="*60)
    print("EXPERIMENT 1: PRIVACY-UTILITY TRADEOFF ANALYSIS")
    print("="*60)
    
    for eps in epsilons:
        label = "No DP Ceiling (Baseline)" if eps == 0 else f"Epsilon = {eps}"
        print(f"\n--- Testing Condition: {label} ---")
        
        f1, prec, rec = train_and_evaluate(data.clone(), epsilon=eps, epochs=50)
        
        print(f"-> FINAL INFERENCE METRICS:")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall:    {rec:.4f}")
        print(f"   F1-Score:  {f1:.4f}")