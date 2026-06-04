from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from src.data import (
    PreparedDataset,
    load_balanced_cicids_dataframe,
    prepare_dataset,
    save_label_mapping,
)
from src.pso import BinaryPSOConfig, run_binary_pso
from src.trainer import SearchConfig, TrainConfig, fit_and_evaluate, search_feature_subset


DEFAULT_CONFIG_PATH = Path("configs/default_multiclass_gatv2.json")


def load_config_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return data


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a JSON config file.",
    )
    pre_args, remaining = pre_parser.parse_known_args()

    config_values: dict[str, Any] = {}
    if pre_args.config.exists():
        config_values = load_config_file(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Train a PSO + GAT intrusion detector on CICIDS2017 MachineLearningCVE CSV files."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=pre_args.config,
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(config_values.get("dataset_dir", r"E:\dataset\MachineLearningCVE")),
        help="Directory containing MachineLearningCVE CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(config_values.get("output_dir", "outputs/training")),
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--task",
        choices=("binary", "multiclass"),
        default=config_values.get("task", "multiclass"),
        help="Binary maps all attacks to ATTACK; multiclass keeps original attack labels.",
    )
    parser.add_argument(
        "--max-rows-per-class",
        type=int,
        default=int(config_values.get("max_rows_per_class", 10000)),
        help="Maximum number of rows retained per class while streaming the CSV files.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=int(config_values.get("chunksize", 20000)),
        help="CSV chunk size used during streaming load.",
    )
    parser.add_argument("--seed", type=int, default=int(config_values.get("seed", 42)))
    parser.add_argument("--batch-size", type=int, default=int(config_values.get("batch_size", 192)))
    parser.add_argument("--epochs", type=int, default=int(config_values.get("epochs", 36)))
    parser.add_argument("--patience", type=int, default=int(config_values.get("patience", 10)))
    parser.add_argument("--hidden-dim", type=int, default=int(config_values.get("hidden_dim", 128)))
    parser.add_argument("--heads", type=int, default=int(config_values.get("heads", 4)))
    parser.add_argument("--dropout", type=float, default=float(config_values.get("dropout", 0.10)))
    parser.add_argument(
        "--model-name",
        choices=("gat", "gatv2"),
        default=config_values.get("model_name", "gatv2"),
        help="Graph attention layer type.",
    )
    parser.add_argument(
        "--use-residual",
        action=argparse.BooleanOptionalAction,
        default=bool(config_values.get("use_residual", True)),
        help="Enable residual projections between GAT layers.",
    )
    parser.add_argument("--learning-rate", type=float, default=float(config_values.get("learning_rate", 5e-4)))
    parser.add_argument("--weight-decay", type=float, default=float(config_values.get("weight_decay", 1e-5)))
    parser.add_argument(
        "--loss-name",
        choices=("cross_entropy", "focal"),
        default=config_values.get("loss_name", "cross_entropy"),
        help="Training loss function.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=float(config_values.get("focal_gamma", 2.0)),
        help="Gamma parameter used when --loss-name focal.",
    )
    parser.add_argument(
        "--disable-class-weights",
        action=argparse.BooleanOptionalAction,
        default=bool(config_values.get("disable_class_weights", False)),
        help="Disable inverse-frequency class weights in the loss.",
    )
    parser.add_argument(
        "--use-weighted-sampler",
        action=argparse.BooleanOptionalAction,
        default=bool(config_values.get("use_weighted_sampler", False)),
        help="Use WeightedRandomSampler on the training split.",
    )
    parser.add_argument("--k-neighbors", type=int, default=int(config_values.get("k_neighbors", 10)))
    parser.add_argument(
        "--graph-metric",
        choices=("cosine", "euclidean"),
        default=config_values.get("graph_metric", "cosine"),
    )
    parser.add_argument(
        "--graph-strategy",
        choices=("knn", "mutual_knn"),
        default=config_values.get("graph_strategy", "knn"),
        help="How to convert nearest-neighbor relations into graph edges.",
    )
    parser.add_argument(
        "--device",
        default=config_values.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        help="Training device.",
    )
    parser.add_argument(
        "--use-pso",
        action=argparse.BooleanOptionalAction,
        default=bool(config_values.get("use_pso", False)),
        help="Enable binary PSO feature selection before final GAT training.",
    )
    parser.add_argument("--pso-particles", type=int, default=int(config_values.get("pso_particles", 6)))
    parser.add_argument("--pso-iterations", type=int, default=int(config_values.get("pso_iterations", 4)))
    parser.add_argument("--pso-inertia", type=float, default=float(config_values.get("pso_inertia", 0.72)))
    parser.add_argument("--pso-c1", type=float, default=float(config_values.get("pso_c1", 1.49)))
    parser.add_argument("--pso-c2", type=float, default=float(config_values.get("pso_c2", 1.49)))
    parser.add_argument(
        "--pso-feature-penalty",
        type=float,
        default=float(config_values.get("pso_feature_penalty", 0.08)),
        help="Penalty coefficient encouraging smaller feature subsets.",
    )
    parser.add_argument(
        "--pso-min-features",
        type=int,
        default=int(config_values.get("pso_min_features", 8)),
        help="Minimum number of features kept by PSO.",
    )
    parser.add_argument(
        "--pso-search-rows-per-class",
        type=int,
        default=int(config_values.get("pso_search_rows_per_class", 256)),
        help="Rows per class sampled from the train/val split for the PSO objective.",
    )
    parser.add_argument(
        "--pso-epochs",
        type=int,
        default=int(config_values.get("pso_epochs", 2)),
        help="Number of short GAT epochs used inside the PSO fitness function.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=float(config_values.get("train_ratio", 0.7)),
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=float(config_values.get("val_ratio", 0.15)),
    )
    return parser.parse_args(remaining)


def make_run_dir(output_dir: Path) -> Path:
    run_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def select_features_with_pso(args: argparse.Namespace, dataset: PreparedDataset) -> list[int]:
    search_cfg = SearchConfig(
        device=args.device,
        batch_size=args.batch_size,
        hidden_dim=max(16, args.hidden_dim // 2),
        heads=max(1, min(args.heads, 2)),
        dropout=args.dropout,
        model_name=args.model_name,
        use_residual=args.use_residual,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_name=args.loss_name,
        focal_gamma=args.focal_gamma,
        use_class_weights=not args.disable_class_weights,
        use_weighted_sampler=args.use_weighted_sampler,
        epochs=args.pso_epochs,
        k_neighbors=max(2, min(args.k_neighbors, 6)),
        graph_metric=args.graph_metric,
        graph_strategy=args.graph_strategy,
        seed=args.seed,
    )
    pso_cfg = BinaryPSOConfig(
        n_particles=args.pso_particles,
        n_iterations=args.pso_iterations,
        inertia=args.pso_inertia,
        c1=args.pso_c1,
        c2=args.pso_c2,
        min_selected=args.pso_min_features,
        feature_penalty=args.pso_feature_penalty,
        seed=args.seed,
    )
    objective = lambda feature_idx: search_feature_subset(
        dataset=dataset,
        feature_indices=feature_idx,
        search_rows_per_class=args.pso_search_rows_per_class,
        config=search_cfg,
    )
    result = run_binary_pso(
        n_dimensions=dataset.x_train.shape[1],
        config=pso_cfg,
        objective_fn=objective,
    )
    return result.best_indices


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def namespace_to_jsonable(namespace: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(namespace).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.output_dir)

    raw_df = load_balanced_cicids_dataframe(
        dataset_dir=args.dataset_dir,
        task=args.task,
        max_rows_per_class=args.max_rows_per_class,
        chunksize=args.chunksize,
    )

    dataset = prepare_dataset(
        df=raw_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    selected_indices = list(range(dataset.x_train.shape[1]))
    if args.use_pso:
        selected_indices = select_features_with_pso(args, dataset)

    train_cfg = TrainConfig(
        device=args.device,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        model_name=args.model_name,
        use_residual=args.use_residual,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_name=args.loss_name,
        focal_gamma=args.focal_gamma,
        use_class_weights=not args.disable_class_weights,
        use_weighted_sampler=args.use_weighted_sampler,
        epochs=args.epochs,
        patience=args.patience,
        k_neighbors=args.k_neighbors,
        graph_metric=args.graph_metric,
        graph_strategy=args.graph_strategy,
        seed=args.seed,
    )

    artifacts = fit_and_evaluate(
        dataset=dataset,
        feature_indices=selected_indices,
        output_dir=run_dir,
        config=train_cfg,
    )

    selected_features = [dataset.feature_names[idx] for idx in selected_indices]
    save_json(
        run_dir / "config.json",
        {
            **namespace_to_jsonable(args),
            "dataset_dir": str(args.dataset_dir),
            "output_dir": str(run_dir),
        },
    )
    save_json(
        run_dir / "metrics.json",
        {
            "selected_feature_count": len(selected_indices),
            "selected_features": selected_features,
            "best_epoch": artifacts.best_epoch,
            "history": artifacts.history,
            "val_metrics": artifacts.val_metrics,
            "test_metrics": artifacts.test_metrics,
        },
    )
    save_json(run_dir / "selected_features.json", {"features": selected_features})
    save_label_mapping(run_dir / "label_mapping.json", dataset.class_names)

    print(f"Run directory: {run_dir}")
    print(f"Selected features: {len(selected_features)}")
    print("Validation metrics:")
    print(json.dumps(artifacts.val_metrics, ensure_ascii=False, indent=2))
    print("Test metrics:")
    print(json.dumps(artifacts.test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
