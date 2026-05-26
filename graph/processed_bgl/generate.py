import os
import pickle
import torch
import networkx as nx
from torch_geometric.utils import from_networkx


gpickle_path = 'cooccurrence_graph.gpickle'
edge_index_path = 'edge_index.pt'

print("读取图（使用 pickle）...")
with open(gpickle_path, 'rb') as f:
    G = pickle.load(f)

print("转换成PyG格式...")
data = from_networkx(G)

print(f"保存 edge_index 到 {edge_index_path} ...")
torch.save(data.edge_index, edge_index_path)
print("完成")
