from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GraphBatch:
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor


def build_knn_edge_index(
    x: torch.Tensor,
    k: int,
    metric: str = "cosine",
    strategy: str = "knn",
) -> torch.Tensor:
    num_nodes = x.size(0)
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    k = max(1, min(k, num_nodes - 1))

    if metric == "cosine":
        normalized = F.normalize(x, p=2, dim=1, eps=1e-12)
        similarity = normalized @ normalized.T
        similarity.fill_diagonal_(-1e9)
        neighbors = similarity.topk(k=k, dim=1).indices
    elif metric == "euclidean":
        distance = torch.cdist(x, x, p=2)
        distance.fill_diagonal_(float("inf"))
        neighbors = distance.topk(k=k, dim=1, largest=False).indices
    else:
        raise ValueError(f"Unsupported graph metric: {metric}")

    src = torch.arange(num_nodes, device=x.device).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = neighbors.reshape(-1)
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=x.device)
    adjacency[src, dst] = True

    if strategy == "knn":
        adjacency = adjacency | adjacency.T
    elif strategy == "mutual_knn":
        adjacency = adjacency & adjacency.T
        # Mutual kNN can become too sparse for tiny batches; fall back if needed.
        if not adjacency.any():
            adjacency[src, dst] = True
            adjacency = adjacency | adjacency.T
    else:
        raise ValueError(f"Unsupported graph strategy: {strategy}")

    edge_index = adjacency.nonzero(as_tuple=False).T.contiguous()
    return edge_index
