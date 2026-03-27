import torch
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import time

from elliptic_model import PrivateSmurfNode
from elliptic_baselines import GraphSAGENodeBaseline, GATNodeBaseline, TGATNodeBaseline
from model import FocalLoss

def run_elliptic_trial(model, graph_data, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    graph_data = graph_data.to(device)
    
    # 1. Isolate Labeled Data
    # The Elliptic dataset contains thousands of unverified transactions labeled as -1.0.
    # We must mask these out to ensure they do not pollute the loss calculation.
    labeled_mask = (graph_data.y != -1.0)
    valid_indices = labeled_mask.nonzero(as_tuple=True)[0]
    
    # 2. Reconstruct Timesteps for Chronological Splitting
    # To prevent data leakage, the exact timestep was stripped from the node features 'x' 
    # and placed securely in 'edge_attr'. We must map it back to the valid nodes for sorting.
    node_times = torch.zeros(graph_data.num_nodes, device=device)
    node_times[graph_data.edge_index[0]] = graph_data.edge_attr[:, 1]
    times = node_times[valid_indices]
    
    # Sort chronologically (Train on historical blocks, test on future blocks)
    sorted_relative_indices = torch.argsort(times)
    sorted_valid_indices = valid_indices[sorted_relative_indices]
    
    split_point = int(0.8 * len(sorted_valid_indices))
    train_idx = sorted_valid_indices[:split_point]
    test_idx = sorted_valid_indices[split_point:]
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(graph_data)
        
        # Calculate loss exclusively on the known, historical training nodes
        loss = criterion(out[train_idx], graph_data.y[train_idx])
        loss.backward()
        optimizer.step()

    # 3. Evaluation Phase
    model.eval()
    with torch.no_grad():
        out = model(graph_data)
        test_out = out[test_idx]
        test_y = graph_data.y[test_idx].cpu().numpy()
        
        preds = (torch.sigmoid(test_out) > 0.5).cpu().numpy()
        
        f1 = f1_score(test_y, preds)
        prec = precision_score(test_y, preds, zero_division=0)
        rec = recall_score(test_y, preds, zero_division=0)
        
    torch.cuda.empty_cache() 
    return prec, rec, f1

if __name__ == "__main__":
    print("Loading Elliptic Graph...")
    try:
        data = torch.load('elliptic_graph.pt', weights_only=False)
    except FileNotFoundError:
        print("Error: Could not find 'elliptic_graph.pt'. Run load_elliptic.py first.")
        exit()
        
    # Baseline comparison lineup adapted for 165-dimensional Node Classification
    models_to_test = [
        ("GraphSAGE (Static)", lambda: GraphSAGENodeBaseline(165, 64)),
        ("GAT (Static Attention)", lambda: GATNodeBaseline(165, 64)),
        ("TGAT (Temporal Baseline)", lambda: TGATNodeBaseline(165, 64, 16)),
        ("PrivateSmurfNode (No DP)", lambda: PrivateSmurfNode(165, 64, 16, epsilon=0)),
        ("PrivateSmurfNode (eps=1.0)", lambda: PrivateSmurfNode(165, 64, 16, epsilon=1.0))
    ]
    
    runs = 3
    print("\n" + "="*70)
    print(f"CROSS-DOMAIN EVALUATION: ELLIPTIC BITCOIN (Chronological Split, {runs} Runs)")
    print("="*70)
    print("| Model | Precision (Mean±SD) | Recall (Mean±SD) | F1-Score (Mean±SD) |")
    print("| :--- | :--- | :--- | :--- |")
    
    for name, model_fn in models_to_test:
        f1s, precs, recs = [], [], []
        for _ in range(runs):
            model = model_fn()
            p, r, f = run_elliptic_trial(model, data.clone(), epochs=100)
            precs.append(p); recs.append(r); f1s.append(f)
            
        p_mean, p_std = np.mean(precs), np.std(precs)
        r_mean, r_std = np.mean(recs), np.std(recs)
        f_mean, f_std = np.mean(f1s), np.std(f1s)
        
        print(f"| {name} | {p_mean:.3f}±{p_std:.3f} | {r_mean:.3f}±{r_std:.3f} | {f_mean:.3f}±{f_std:.3f} |")