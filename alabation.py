import torch
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import time

from model import PrivateSmurf, FocalLoss
from temporal_baseline import TGATBaseline
from baseline import GraphSAGEBaseline
from train import subsample_graph 

def alabation(model, graph_data, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    graph_data = graph_data.to(device)
    
    # 1. Strict Chronological Split
    # In real-world AML systems, models must train on historical data to predict 
    # future illicit structuring. Random shuffling causes "data leakage" from the future.
    # edge_attr[:, 1] contains the normalized timestamps.
    times = graph_data.edge_attr[:, 1]
    sorted_indices = torch.argsort(times)
    
    num_edges = graph_data.num_edges
    split_point = int(0.8 * num_edges) # 80% Train (Past), 20% Test (Future)
    
    train_idx = sorted_indices[:split_point]
    test_idx = sorted_indices[split_point:]
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # Focal Loss handles extreme class imbalance (fraud is <1% of transactions)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(graph_data)
        
        # Calculate loss only on the training timeframe
        loss = criterion(out[train_idx], graph_data.y[train_idx])
        loss.backward()
        optimizer.step()

    # 2. Evaluation Phase
    model.eval()
    with torch.no_grad():
        out = model(graph_data)
        test_out = out[test_idx]
        test_y = graph_data.y[test_idx].cpu().numpy()
        
        # Apply sigmoid to get probabilities, using 0.4 as the operational threshold
        preds = (torch.sigmoid(test_out) > 0.4).cpu().numpy()
        
        f1 = f1_score(test_y, preds)
        prec = precision_score(test_y, preds, zero_division=0)
        rec = recall_score(test_y, preds, zero_division=0)
        
    torch.cuda.empty_cache() 
    return prec, rec, f1

if __name__ == "__main__":
    
    full_data = torch.load('ibm_smurf_graph.pt', weights_only=False)
    # Subsample benign edges to speed up local training while maintaining imbalance
    balanced_data = subsample_graph(full_data, normal_to_fraud_ratio=10)
    
    models_to_test = [
        ("GraphSAGE (Static Baseline)", lambda: GraphSAGEBaseline(16, 64)),
        ("TGAT (Temporal Baseline, No Privacy)", lambda: TGATBaseline(16, 64, 16)),
        ("PrivateSmurf (No DP)", lambda: PrivateSmurf(16, 64, 16, epsilon=0)),
        ("PrivateSmurf (eps=1.0)", lambda: PrivateSmurf(16, 64, 16, epsilon=1.0))
    ]
    
    runs = 3
    print("\n" + "="*70)
    print(f"TEMPORAL ABLATION STUDY: SYNTHETIC AML DATA (Chronological Split, {runs} Runs)")
    print("="*70)
    print("| Model | Precision (Mean±SD) | Recall (Mean±SD) | F1-Score (Mean±SD) |")
    print("| :--- | :--- | :--- | :--- |")
    
    for name, model_fn in models_to_test:
        f1s, precs, recs = [], [], []
        for _ in range(runs):
            # Re-initialize the model to reset weights for a fair, independent trial
            model = model_fn()
            p, r, f = alabation(model, balanced_data.clone(), epochs=100)
            precs.append(p); recs.append(r); f1s.append(f)
            
        p_mean, p_std = np.mean(precs), np.std(precs)
        r_mean, r_std = np.mean(recs), np.std(recs)
        f_mean, f_std = np.mean(f1s), np.std(f1s)
        
        print(f"| {name} | {p_mean:.3f}±{p_std:.3f} | {r_mean:.3f}±{r_std:.3f} | {f_mean:.3f}±{f_std:.3f} |")