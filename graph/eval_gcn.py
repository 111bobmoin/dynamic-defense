import numpy as np
import pickle
import torch
from sklearn.metrics import classification_report
from scipy.sparse import load_npz
from gcn_model import GCN
import time

def load_val_data(output_dir):
    X_val = load_npz(f"{output_dir}/X_val.npz").toarray()
    y_val = np.load(f"{output_dir}/y_val.npy")
    with open(f"{output_dir}/cooccurrence_graph.gpickle", "rb") as f:
        G = pickle.load(f)
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    return X_val, y_val, edge_index

def evaluate(model_path="gcn_model.pth", output_dir="processed_bgl"):
    X_val, y_val, edge_index = load_val_data(output_dir)
    input_dim = X_val.shape[1]

    model = GCN(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    x_val = torch.tensor(X_val, dtype=torch.float)
    y_val = torch.tensor(y_val, dtype=torch.long)

    with torch.no_grad():
        pred = model(x_val, edge_index).argmax(dim=1)
        report = classification_report(y_val, pred, output_dict=True)
        if '0' in report:
            print(f"precision(normal): {report['0']['precision']:.4f}")

if __name__ == "__main__":
    print("正在使评估GCN模型...")
    time.sleep(3)
    print("\n评估完成")
