import numpy as np
import pickle
import torch
from scipy.sparse import load_npz
from gcn_model import GCN


def load_data(output_dir):
    X_train = load_npz(f"{output_dir}/X_train.npz")
    y_train = np.load(f"{output_dir}/y_train.npy")
    with open(f"{output_dir}/cooccurrence_graph.gpickle", "rb") as f:
        G = pickle.load(f)
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)  # 双向边
    return X_train, y_train, edge_index


def sparse_matrix_to_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data.astype(np.float32))
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def train_gcn(output_dir="processed_bgl", model_path="gcn_model.pth", epochs=50, lr=0.01):
    X_train, y_train, edge_index = load_data(output_dir)
    x_train_sparse = sparse_matrix_to_sparse_tensor(X_train)
    y_train = torch.tensor(y_train, dtype=torch.long)

    input_dim = X_train.shape[1]
    model = GCN(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 使用稀疏输入张量
        out = model(x_train_sparse, edge_index)
        loss = criterion(out, y_train)

        loss.backward()
        optimizer.step()
        print(f"[Epoch {epoch + 1}/{epochs}] Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"[✅] 模型已保存到 {model_path}")


if __name__ == "__main__":
    train_gcn()
