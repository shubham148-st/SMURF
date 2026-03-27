import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
import os

def build_elliptic_graph():
    """
    Parses the Elliptic Bitcoin Dataset. Reconstructs temporal routing 
    and handles extensive feature anonymization protocols.
    """
    
    features_csv = "elliptic_txs_features.csv"
    edges_csv = "elliptic_txs_edgelist.csv"
    classes_csv = "elliptic_txs_classes.csv"
    
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Verification Failed: {features_csv} is missing.")
        
    df_features = pd.read_csv(features_csv, header=None)
    df_edges = pd.read_csv(edges_csv)
    df_classes = pd.read_csv(classes_csv)
    
    # Standardize label taxonomy (1.0: Illicit, 0.0: Licit, -1.0: Unknown)
    df_classes['class'] = df_classes['class'].map({'1': 1.0, '2': 0.0, 'unknown': -1.0})
    
    # Topology Mapping
    tx_ids = df_features[0].values
    id_map = {tx_id: i for i, tx_id in enumerate(tx_ids)}
    
    src = [id_map[tx] for tx in df_edges['txId1'] if tx in id_map]
    dst = [id_map[tx] for tx in df_edges['txId2'] if tx in id_map]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # Extract dense local representations (discarding ID and temporal column 1)
    node_features = df_features.loc[:, 2:].values 
    node_timesteps = df_features[1].values
    
    scaler = MinMaxScaler()
    x = torch.tensor(scaler.fit_transform(node_features), dtype=torch.float)
    
    # Isolate temporal features into edge_attr to prevent spatial leakage
    edge_timesteps = node_timesteps[src]
    edge_time_norm = scaler.fit_transform(edge_timesteps.reshape(-1, 1))
    
    dummy_amounts = np.zeros_like(edge_time_norm)
    edge_attr = torch.tensor(np.column_stack((dummy_amounts, edge_time_norm)), dtype=torch.float)
    
    # Ground Truth Alignment
    df_classes = df_classes.set_index('txId').reindex(tx_ids).reset_index()
    y = torch.tensor(df_classes['class'].values, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.num_nodes = len(tx_ids)
    
    print(f"Elliptic Graph Construction Complete. Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    torch.save(data,"elliptic_graph.pt")
    print("Serialization successful.")
    return data

if __name__ == "__main__":
    build_elliptic_graph()