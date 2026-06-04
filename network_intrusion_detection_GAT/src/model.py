from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv, GATv2Conv


class IntrusionGAT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        heads: int,
        dropout: float,
        model_name: str = "gat",
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.use_residual = use_residual

        if model_name == "gat":
            conv_cls = GATConv
        elif model_name == "gatv2":
            conv_cls = GATv2Conv
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        first_out_dim = hidden_dim * heads
        second_out_dim = hidden_dim
        self.gat1 = conv_cls(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.gat2 = conv_cls(
            in_channels=first_out_dim,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(first_out_dim)
        self.norm2 = nn.LayerNorm(second_out_dim)
        self.residual1 = nn.Linear(input_dim, first_out_dim) if use_residual else None
        self.residual2 = nn.Linear(first_out_dim, second_out_dim) if use_residual else None
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        residual = self.residual1(x) if self.residual1 is not None else None
        x = self.gat1(x, edge_index)
        if residual is not None:
            x = x + residual
        x = torch.relu(x)
        x = self.norm1(x)
        x = torch.dropout(x, p=self.dropout, train=self.training)

        residual = self.residual2(x) if self.residual2 is not None else None
        x = self.gat2(x, edge_index)
        if residual is not None:
            x = x + residual
        x = torch.relu(x)
        x = self.norm2(x)
        x = torch.dropout(x, p=self.dropout, train=self.training)
        return self.classifier(x)
