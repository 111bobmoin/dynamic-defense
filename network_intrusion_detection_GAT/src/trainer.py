from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .data import PreparedDataset
from .graph import GraphBatch, build_knn_edge_index
from .losses import FocalLoss
from .metrics import classification_metrics
from .model import IntrusionGAT


class TensorDataset2D(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


@dataclass
class TrainConfig:
    device: str
    batch_size: int
    hidden_dim: int
    heads: int
    dropout: float
    model_name: str
    use_residual: bool
    learning_rate: float
    weight_decay: float
    loss_name: str
    focal_gamma: float
    use_class_weights: bool
    use_weighted_sampler: bool
    epochs: int
    patience: int
    k_neighbors: int
    graph_metric: str
    graph_strategy: str
    seed: int


@dataclass
class SearchConfig:
    device: str
    batch_size: int
    hidden_dim: int
    heads: int
    dropout: float
    model_name: str
    use_residual: bool
    learning_rate: float
    weight_decay: float
    loss_name: str
    focal_gamma: float
    use_class_weights: bool
    use_weighted_sampler: bool
    epochs: int
    k_neighbors: int
    graph_metric: str
    graph_strategy: str
    seed: int


@dataclass
class TrainArtifacts:
    best_epoch: int
    history: list[dict[str, float]]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_collate_fn(device: str, k_neighbors: int, graph_metric: str, graph_strategy: str):
    def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> GraphBatch:
        x, y = zip(*batch, strict=True)
        x_tensor = torch.stack(x, dim=0)
        y_tensor = torch.stack(y, dim=0)
        edge_index = build_knn_edge_index(x_tensor, k=k_neighbors, metric=graph_metric, strategy=graph_strategy)
        return GraphBatch(
            x=x_tensor.to(device),
            y=y_tensor.to(device),
            edge_index=edge_index.to(device),
        )

    return collate_fn


def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    device: str,
    k_neighbors: int,
    graph_metric: str,
    graph_strategy: str,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    dataset = TensorDataset2D(x, y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=0,
        collate_fn=make_collate_fn(
            device=device,
            k_neighbors=k_neighbors,
            graph_metric=graph_metric,
            graph_strategy=graph_strategy,
        ),
        drop_last=False,
    )


def build_class_weights(labels: torch.Tensor, num_classes: int, device: str) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = torch.where(counts == 0, torch.ones_like(counts), counts)
    weights = counts.sum() / (counts * num_classes)
    return weights.to(device)


def build_weighted_sampler(labels: torch.Tensor, num_classes: int, seed: int) -> WeightedRandomSampler:
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = torch.where(counts == 0, torch.ones_like(counts), counts)
    class_weights = counts.sum() / (counts * num_classes)
    sample_weights = class_weights[labels.cpu()].double()
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=labels.size(0),
        replacement=True,
        generator=generator,
    )


def build_loss(
    loss_name: str,
    labels: torch.Tensor,
    num_classes: int,
    device: str,
    focal_gamma: float,
    use_class_weights: bool,
) -> nn.Module:
    alpha = build_class_weights(labels, num_classes, device) if use_class_weights else None
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=alpha)
    if loss_name == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=alpha, reduction="mean")
    raise ValueError(f"Unsupported loss: {loss_name}")


def build_model(
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    heads: int,
    dropout: float,
    model_name: str,
    use_residual: bool,
    device: str,
) -> IntrusionGAT:
    return IntrusionGAT(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        heads=heads,
        dropout=dropout,
        model_name=model_name,
        use_residual=use_residual,
    ).to(device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW | None,
    loss_fn: nn.Module,
    num_classes: int,
) -> tuple[float, dict[str, float]]:
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_nodes = 0
    all_targets: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []

    for batch in loader:
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(batch.x, batch.edge_index)
        loss = loss_fn(logits, batch.y)
        if is_training:
            loss.backward()
            optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = batch.y.size(0)
        total_loss += loss.item() * batch_size
        total_nodes += batch_size
        all_targets.append(batch.y.detach().cpu())
        all_predictions.append(predictions.detach().cpu())

    y_true = torch.cat(all_targets, dim=0)
    y_pred = torch.cat(all_predictions, dim=0)
    metrics = classification_metrics(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
    average_loss = total_loss / max(1, total_nodes)
    return average_loss, metrics


def select_tensor_rows_per_class(
    x: torch.Tensor,
    y: torch.Tensor,
    max_rows_per_class: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    indices: list[torch.Tensor] = []
    for class_id in torch.unique(y, sorted=True):
        class_indices = torch.nonzero(y == class_id, as_tuple=False).view(-1)
        permutation = class_indices[torch.randperm(class_indices.numel(), generator=generator)]
        indices.append(permutation[: min(max_rows_per_class, permutation.numel())])
    merged = torch.cat(indices, dim=0)
    merged = merged[torch.randperm(merged.numel(), generator=generator)]
    return x[merged], y[merged]


def search_feature_subset(
    dataset: PreparedDataset,
    feature_indices: list[int],
    search_rows_per_class: int,
    config: SearchConfig,
) -> float:
    set_seed(config.seed)
    x_train, y_train = select_tensor_rows_per_class(
        dataset.x_train[:, feature_indices],
        dataset.y_train,
        max_rows_per_class=search_rows_per_class,
        seed=config.seed,
    )
    x_val, y_val = select_tensor_rows_per_class(
        dataset.x_val[:, feature_indices],
        dataset.y_val,
        max_rows_per_class=max(1, search_rows_per_class // 2),
        seed=config.seed + 1,
    )
    train_sampler = (
        build_weighted_sampler(y_train, len(dataset.class_names), config.seed)
        if config.use_weighted_sampler
        else None
    )
    train_loader = make_loader(
        x_train,
        y_train,
        batch_size=config.batch_size,
        shuffle=True,
        device=config.device,
        k_neighbors=config.k_neighbors,
        graph_metric=config.graph_metric,
        graph_strategy=config.graph_strategy,
        sampler=train_sampler,
    )
    val_loader = make_loader(
        x_val,
        y_val,
        batch_size=config.batch_size,
        shuffle=False,
        device=config.device,
        k_neighbors=config.k_neighbors,
        graph_metric=config.graph_metric,
        graph_strategy=config.graph_strategy,
    )
    model = build_model(
        input_dim=x_train.size(1),
        hidden_dim=config.hidden_dim,
        num_classes=len(dataset.class_names),
        heads=config.heads,
        dropout=config.dropout,
        model_name=config.model_name,
        use_residual=config.use_residual,
        device=config.device,
    )
    loss_fn = build_loss(
        loss_name=config.loss_name,
        labels=y_train,
        num_classes=len(dataset.class_names),
        device=config.device,
        focal_gamma=config.focal_gamma,
        use_class_weights=config.use_class_weights,
    )
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    for _ in range(config.epochs):
        run_epoch(model, train_loader, optimizer, loss_fn, num_classes=len(dataset.class_names))
    _, metrics = run_epoch(model, val_loader, None, loss_fn, num_classes=len(dataset.class_names))
    return 1.0 - float(metrics["macro_f1"])


def fit_and_evaluate(
    dataset: PreparedDataset,
    feature_indices: list[int],
    output_dir: Path,
    config: TrainConfig,
) -> TrainArtifacts:
    set_seed(config.seed)
    x_train = dataset.x_train[:, feature_indices]
    x_val = dataset.x_val[:, feature_indices]
    x_test = dataset.x_test[:, feature_indices]
    train_sampler = (
        build_weighted_sampler(dataset.y_train, len(dataset.class_names), config.seed)
        if config.use_weighted_sampler
        else None
    )

    train_loader = make_loader(
        x_train,
        dataset.y_train,
        batch_size=config.batch_size,
        shuffle=True,
        device=config.device,
        k_neighbors=config.k_neighbors,
        graph_metric=config.graph_metric,
        graph_strategy=config.graph_strategy,
        sampler=train_sampler,
    )
    val_loader = make_loader(
        x_val,
        dataset.y_val,
        batch_size=config.batch_size,
        shuffle=False,
        device=config.device,
        k_neighbors=config.k_neighbors,
        graph_metric=config.graph_metric,
        graph_strategy=config.graph_strategy,
    )
    test_loader = make_loader(
        x_test,
        dataset.y_test,
        batch_size=config.batch_size,
        shuffle=False,
        device=config.device,
        k_neighbors=config.k_neighbors,
        graph_metric=config.graph_metric,
        graph_strategy=config.graph_strategy,
    )

    model = build_model(
        input_dim=x_train.size(1),
        hidden_dim=config.hidden_dim,
        num_classes=len(dataset.class_names),
        heads=config.heads,
        dropout=config.dropout,
        model_name=config.model_name,
        use_residual=config.use_residual,
        device=config.device,
    )
    loss_fn = build_loss(
        loss_name=config.loss_name,
        labels=dataset.y_train,
        num_classes=len(dataset.class_names),
        device=config.device,
        focal_gamma=config.focal_gamma,
        use_class_weights=config.use_class_weights,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_score = -1.0
    patience_counter = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, optimizer, loss_fn, num_classes=len(dataset.class_names)
        )
        val_loss, val_metrics = run_epoch(model, val_loader, None, loss_fn, num_classes=len(dataset.class_names))
        record = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_accuracy": float(train_metrics["accuracy"]),
            "train_macro_f1": float(train_metrics["macro_f1"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
        }
        history.append(record)

        if val_metrics["macro_f1"] > best_score:
            best_score = float(val_metrics["macro_f1"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    model.load_state_dict(best_state)
    _, val_metrics = run_epoch(model, val_loader, None, loss_fn, num_classes=len(dataset.class_names))
    _, test_metrics = run_epoch(model, test_loader, None, loss_fn, num_classes=len(dataset.class_names))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "selected_feature_indices": feature_indices,
            "selected_feature_names": [dataset.feature_names[index] for index in feature_indices],
            "class_names": dataset.class_names,
            "mean": dataset.mean[:, feature_indices],
            "std": dataset.std[:, feature_indices],
            "config": config.__dict__,
        },
        output_dir / "model.pt",
    )
    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)

    return TrainArtifacts(
        best_epoch=best_epoch,
        history=history,
        val_metrics={key: float(value) for key, value in val_metrics.items()},
        test_metrics={key: float(value) for key, value in test_metrics.items()},
    )
