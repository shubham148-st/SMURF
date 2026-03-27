import pandas as pd
import torch
import numpy as np  
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import time
import os

def build_private_smurf_graph(csv_path):
    """
    Parses a synthetic transaction ledger and constructs a PyTorch Geometric 
    Data object suitable for edge-classification temporal networks.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source ledger not found: {csv_path}")

    print(f"Initializing graph construction from ledger: {csv_path}")
    start_time = time.time()
    
    df = pd.read_csv(csv_path)
    
    # 1. Node Mapping (Accounts)
    print("Extracting unique entities and mapping node indices...")
    all_accounts = pd.concat([df['Account'], df['Account.1']]).unique()
    
    encoder = LabelEncoder()
    encoder.fit(all_accounts)
    
    src_nodes = encoder.transform(df['Account'])
    dst_nodes = encoder.transform(df['Account.1'])
    
    # 2. Edge Index Construction
    print("Constructing sparse edge index matrix...")
    edge_index = torch.tensor(np.array([src_nodes, dst_nodes]), dtype=torch.long)
    
    # 3. Edge Feature Extraction and Normalization
    print("Normalizing transactional features (Amount, Timestamp)...")
    scaler = MinMaxScaler()
    
    amount_col = 'Amount Received' if 'Amount Received' in df.columns else 'Amount'
    amount_normalized = scaler.fit_transform(df[[amount_col]].values)
    
    # Temporal processing: convert to nanosecond precision, then scale
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['time_numeric'] = df['Timestamp'].astype('int64') / 10**9 
    time_normalized = scaler.fit_transform(df[['time_numeric']].values)
    
    # Vectorize edge attributes
    edge_features_np = np.column_stack((amount_normalized.flatten(), time_normalized.flatten()))
    edge_attr = torch.tensor(edge_features_np, dtype=torch.float)
    
    # 4. Target Extraction
    print("Isolating illicit transaction labels...")
    y = torch.tensor(df['Is Laundering'].values, dtype=torch.float)
    
    # 5. Graph Instantiation
    data = Data(edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.num_nodes = len(all_accounts)
    
    print(f"--- Graph initialization completed in {time.time() - start_time:.2f} seconds ---")
    print(f"Total Nodes (Entities): {data.num_nodes}")
    print(f"Total Edges (Transactions): {data.num_edges}")
    print(f"Illicit Density: {int(y.sum().item())} / {data.num_edges}")
    
    return data

if __name__ == "__main__":
    file_name = 'HI-Small_Trans.csv'
    
    try:
        graph_data = build_private_smurf_graph(file_name)
        torch.save(graph_data, 'ibm_smurf_graph.pt')
        print("Graph serialized successfully to 'ibm_smurf_graph.pt'.")
    except Exception as e:
        print(f"Critical error during graph construction: {e}")