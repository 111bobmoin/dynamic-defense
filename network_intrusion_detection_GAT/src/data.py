from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from src.cicflow_adapter import (
    adapt_cicflowmeter_schema,
    clean_column_names,
    looks_like_cicflowmeter_export,
)


LABEL_COLUMN = "Label"


@dataclass
class PreparedDataset:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    feature_names: list[str]
    class_names: list[str]
    mean: torch.Tensor
    std: torch.Tensor


def discover_csv_files(dataset_dir: Path) -> list[Path]:
    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")
    return csv_files


def clean_columns(columns: Iterable[str]) -> list[str]:
    return clean_column_names(columns)


def normalize_label(label: str, task: str) -> str:
    value = str(label).strip()
    if task == "binary":
        return "BENIGN" if value.upper() == "BENIGN" else "ATTACK"
    return value


def clean_chunk(chunk: pd.DataFrame, task: str) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk.columns = clean_columns(chunk.columns)
    raw_cicflowmeter = looks_like_cicflowmeter_export(chunk.columns)
    chunk = adapt_cicflowmeter_schema(
        chunk,
        label_column=LABEL_COLUMN,
        drop_metadata=raw_cicflowmeter,
        drop_extra=raw_cicflowmeter,
        ensure_all_features=raw_cicflowmeter,
    )
    if LABEL_COLUMN not in chunk.columns:
        raise KeyError(f"Expected '{LABEL_COLUMN}' column, found: {chunk.columns.tolist()}")
    chunk[LABEL_COLUMN] = chunk[LABEL_COLUMN].astype(str).map(lambda value: normalize_label(value, task))
    feature_columns = [column for column in chunk.columns if column != LABEL_COLUMN]
    for column in feature_columns:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
    chunk = chunk.replace([np.inf, -np.inf], np.nan)
    chunk = chunk.dropna(subset=[LABEL_COLUMN])
    return chunk


def load_balanced_cicids_dataframe(
    dataset_dir: Path,
    task: str,
    max_rows_per_class: int,
    chunksize: int,
) -> pd.DataFrame:
    csv_files = discover_csv_files(dataset_dir)
    buffers: dict[str, list[pd.DataFrame]] = {}
    counts: dict[str, int] = {}
    binary_targets = {"BENIGN", "ATTACK"}

    for csv_path in csv_files:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
            cleaned = clean_chunk(chunk, task=task)
            for label_value, group in cleaned.groupby(LABEL_COLUMN, sort=False):
                current = counts.get(label_value, 0)
                remaining = max_rows_per_class - current
                if remaining <= 0:
                    continue
                selected = group.head(remaining)
                if selected.empty:
                    continue
                buffers.setdefault(label_value, []).append(selected)
                counts[label_value] = current + len(selected)
            if task == "binary" and all(counts.get(label, 0) >= max_rows_per_class for label in binary_targets):
                break
        if task == "binary" and all(counts.get(label, 0) >= max_rows_per_class for label in binary_targets):
            break

    if not buffers:
        raise RuntimeError(f"No rows collected from {dataset_dir}")

    frames = [pd.concat(parts, ignore_index=True) for _, parts in sorted(buffers.items())]
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def allocate_split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    if total == 1:
        return 1, 0, 0
    if total == 2:
        return 1, 1, 0

    n_train = max(1, int(round(total * train_ratio)))
    n_val = max(1, int(round(total * val_ratio)))
    n_test = total - n_train - n_val

    while n_test < 1:
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break
        n_test = total - n_train - n_val

    if n_test < 0:
        n_test = 0
    if n_train + n_val + n_test != total:
        n_test = total - n_train - n_val
    return n_train, n_val, n_test


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in df.groupby(LABEL_COLUMN, sort=False):
        indices = np.arange(len(group))
        rng.shuffle(indices)
        shuffled = group.iloc[indices].reset_index(drop=True)
        n_train, n_val, _ = allocate_split_counts(len(shuffled), train_ratio, val_ratio)
        train_parts.append(shuffled.iloc[:n_train])
        val_parts.append(shuffled.iloc[n_train : n_train + n_val])
        test_parts.append(shuffled.iloc[n_train + n_val :])

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df, test_df


def fill_missing_with_train_medians(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    medians = train_df[feature_names].median()
    train_df[feature_names] = train_df[feature_names].fillna(medians)
    val_df[feature_names] = val_df[feature_names].fillna(medians)
    test_df[feature_names] = test_df[feature_names].fillna(medians)
    return train_df, val_df, test_df


def to_tensor_frame(df: pd.DataFrame, feature_names: list[str], class_to_idx: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(df[feature_names].to_numpy(dtype=np.float32), dtype=torch.float32)
    y = torch.tensor([class_to_idx[label] for label in df[LABEL_COLUMN].tolist()], dtype=torch.long)
    return x, y


def standardize_tensors(
    x_train: torch.Tensor,
    x_val: torch.Tensor,
    x_test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, unbiased=False, keepdim=True)
    keep_mask = std.squeeze(0) > 1e-6
    x_train = x_train[:, keep_mask]
    x_val = x_val[:, keep_mask]
    x_test = x_test[:, keep_mask]
    mean = mean[:, keep_mask]
    std = std[:, keep_mask]
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std
    return x_train, x_val, x_test, mean, std, keep_mask


def prepare_dataset(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> PreparedDataset:
    df = df.copy()
    df.columns = clean_columns(df.columns)
    feature_names = [column for column in df.columns if column != LABEL_COLUMN]
    train_df, val_df, test_df = stratified_split(df, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    train_df, val_df, test_df = fill_missing_with_train_medians(train_df, val_df, test_df, feature_names)

    class_names = sorted(train_df[LABEL_COLUMN].unique().tolist())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    x_train, y_train = to_tensor_frame(train_df, feature_names, class_to_idx)
    x_val, y_val = to_tensor_frame(val_df, feature_names, class_to_idx)
    x_test, y_test = to_tensor_frame(test_df, feature_names, class_to_idx)

    x_train, x_val, x_test, mean, std, keep_mask = standardize_tensors(x_train, x_val, x_test)
    kept_features = [name for name, keep in zip(feature_names, keep_mask.tolist(), strict=True) if keep]

    return PreparedDataset(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        feature_names=kept_features,
        class_names=class_names,
        mean=mean,
        std=std,
    )


def save_label_mapping(path: Path, class_names: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump({str(index): label for index, label in enumerate(class_names)}, handle, ensure_ascii=False, indent=2)
