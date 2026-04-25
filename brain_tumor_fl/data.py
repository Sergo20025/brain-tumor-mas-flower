from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100, ImageFolder

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_TRAIN_DIRS = ("train", "Training")
SUPPORTED_TEST_DIRS = ("test", "Testing")


class BrainTumorDataset(Dataset):
    def __init__(self, samples: list[tuple[Any, int]], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        source, label = self.samples[index]
        if isinstance(source, str):
            image = Image.open(source).convert("RGB")
        else:
            image = Image.fromarray(source).convert("RGB")
        return self.transform(image), label


@dataclass
class DatasetLayout:
    train_samples: list[tuple[Any, int]]
    test_samples: list[tuple[Any, int]]
    classes: list[str]


@dataclass
class ClientPartition:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    summary: dict[str, Any]


@dataclass
class CentralizedDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    classes: list[str]
    summary: dict[str, Any]


def build_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _read_imagefolder_samples(root: Path) -> tuple[list[tuple[str, int]], list[str]]:
    dataset = ImageFolder(root=str(root))
    samples = [(str(path), int(label)) for path, label in dataset.samples]
    return samples, dataset.classes


def _is_cifar100_root(dataset_root: str) -> bool:
    normalized = dataset_root.strip().lower().replace("_", "-")
    return normalized in {"cifar100", "cifar-100"}


def _load_cifar100_layout(dataset_root: str) -> DatasetLayout:
    data_root = Path("data") / "cifar100"
    train_dataset = CIFAR100(root=str(data_root), train=True, download=True)
    test_dataset = CIFAR100(root=str(data_root), train=False, download=True)

    train_samples = [
        (image, int(label))
        for image, label in zip(train_dataset.data, train_dataset.targets, strict=True)
    ]
    test_samples = [
        (image, int(label))
        for image, label in zip(test_dataset.data, test_dataset.targets, strict=True)
    ]
    return DatasetLayout(
        train_samples=train_samples,
        test_samples=test_samples,
        classes=list(train_dataset.classes),
    )


def discover_dataset_layout(dataset_root: str, test_split: float, seed: int) -> DatasetLayout:
    if _is_cifar100_root(dataset_root):
        return _load_cifar100_layout(dataset_root)

    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root '{dataset_root}' does not exist. "
            "Place the Brain Tumor MRI dataset into this directory."
        )

    for train_name in SUPPORTED_TRAIN_DIRS:
        for test_name in SUPPORTED_TEST_DIRS:
            train_dir = root / train_name
            test_dir = root / test_name
            if train_dir.exists() and test_dir.exists():
                train_samples, classes = _read_imagefolder_samples(train_dir)
                test_samples, test_classes = _read_imagefolder_samples(test_dir)
                if classes != test_classes:
                    raise ValueError("Train and test class folders do not match.")
                return DatasetLayout(
                    train_samples=train_samples, test_samples=test_samples, classes=classes
                )

    full_samples, classes = _read_imagefolder_samples(root)
    labels = np.array([label for _, label in full_samples])
    train_idx, test_idx = train_test_split(
        np.arange(len(full_samples)),
        test_size=test_split,
        random_state=seed,
        stratify=labels,
    )
    train_samples = [full_samples[idx] for idx in train_idx]
    test_samples = [full_samples[idx] for idx in test_idx]
    return DatasetLayout(train_samples=train_samples, test_samples=test_samples, classes=classes)


def _ensure_non_empty_partitions(partitions: list[list[int]], rng: np.random.Generator) -> None:
    empty_clients = [cid for cid, part in enumerate(partitions) if len(part) == 0]
    while empty_clients:
        donor_id = int(np.argmax([len(part) for part in partitions]))
        if len(partitions[donor_id]) <= 1:
            break
        receiver_id = empty_clients.pop(0)
        move_idx = rng.integers(0, len(partitions[donor_id]))
        partitions[receiver_id].append(partitions[donor_id].pop(int(move_idx)))


def _normalize_positive_counts(
    raw_weights: np.ndarray,
    total_count: int,
    min_count: int = 1,
) -> np.ndarray:
    if total_count < len(raw_weights) * min_count:
        raise ValueError("total_count is too small for the requested minimum count per client.")

    normalized = raw_weights / raw_weights.sum()
    counts = np.floor(normalized * total_count).astype(int)
    counts = np.maximum(counts, min_count)

    current_total = int(counts.sum())
    while current_total > total_count:
        donor_id = int(np.argmax(counts))
        if counts[donor_id] > min_count:
            counts[donor_id] -= 1
            current_total -= 1
        else:
            break

    while current_total < total_count:
        receiver_id = int(np.argmax(raw_weights / np.maximum(counts, 1)))
        counts[receiver_id] += 1
        current_total += 1

    return counts


def _create_quantity_skew_partitions(
    all_indices: np.ndarray,
    num_clients: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    shuffled = rng.permutation(all_indices)
    raw_sizes = rng.lognormal(mean=0.0, sigma=1.0, size=num_clients)
    counts = _normalize_positive_counts(raw_sizes, total_count=len(shuffled), min_count=1)

    partitions: list[list[int]] = []
    start = 0
    for count in counts:
        stop = start + int(count)
        partitions.append(shuffled[start:stop].tolist())
        start = stop
    return partitions


def _create_shard_partitions(
    labels: np.ndarray,
    all_indices: np.ndarray,
    num_clients: int,
    rng: np.random.Generator,
    *,
    quantity_skew: bool,
) -> list[list[int]]:
    classwise_indices: list[int] = []
    num_classes = int(labels.max()) + 1
    for class_id in range(num_classes):
        class_indices = all_indices[labels == class_id].copy()
        rng.shuffle(class_indices)
        classwise_indices.extend(class_indices.tolist())

    num_shards = max(num_clients * 3, num_clients)
    shards = [list(shard) for shard in np.array_split(np.array(classwise_indices), num_shards)]
    shard_order = rng.permutation(num_shards).tolist()

    if quantity_skew:
        raw_shard_sizes = rng.lognormal(mean=0.0, sigma=0.9, size=num_clients)
        shard_counts = _normalize_positive_counts(
            raw_shard_sizes,
            total_count=num_shards,
            min_count=1,
        )
    else:
        base = num_shards // num_clients
        remainder = num_shards % num_clients
        shard_counts = np.array(
            [base + (1 if client_id < remainder else 0) for client_id in range(num_clients)],
            dtype=int,
        )

    partitions: list[list[int]] = [[] for _ in range(num_clients)]
    cursor = 0
    for client_id, shard_count in enumerate(shard_counts.tolist()):
        client_shard_ids = shard_order[cursor : cursor + shard_count]
        cursor += shard_count
        for shard_id in client_shard_ids:
            partitions[client_id].extend(shards[shard_id])
        rng.shuffle(partitions[client_id])

    _ensure_non_empty_partitions(partitions, rng)
    return partitions


def create_client_partitions(
    samples: list[tuple[str, int]],
    num_clients: int,
    partition_mode: str,
    dirichlet_alpha: float,
    seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    labels = np.array([label for _, label in samples])
    all_indices = np.arange(len(samples))

    if partition_mode == "iid":
        shuffled = rng.permutation(all_indices)
        return [list(split) for split in np.array_split(shuffled, num_clients)]

    if partition_mode == "quantity_skew":
        return _create_quantity_skew_partitions(
            all_indices=all_indices,
            num_clients=num_clients,
            rng=rng,
        )

    if partition_mode == "shards":
        return _create_shard_partitions(
            labels=labels,
            all_indices=all_indices,
            num_clients=num_clients,
            rng=rng,
            quantity_skew=False,
        )

    if partition_mode == "shards_quantity_skew":
        return _create_shard_partitions(
            labels=labels,
            all_indices=all_indices,
            num_clients=num_clients,
            rng=rng,
            quantity_skew=True,
        )

    if partition_mode != "dirichlet":
        raise ValueError(f"Unsupported partition mode: {partition_mode}")

    num_classes = int(labels.max()) + 1
    partitions: list[list[int]] = [[] for _ in range(num_clients)]
    for class_id in range(num_classes):
        class_indices = all_indices[labels == class_id]
        rng.shuffle(class_indices)
        if len(class_indices) == 0:
            continue

        proportions = rng.dirichlet(np.full(num_clients, dirichlet_alpha))
        counts = rng.multinomial(len(class_indices), proportions)

        start = 0
        for client_id, count in enumerate(counts):
            stop = start + int(count)
            partitions[client_id].extend(class_indices[start:stop].tolist())
            start = stop

    _ensure_non_empty_partitions(partitions, rng)
    for partition in partitions:
        rng.shuffle(partition)
    return partitions


def _safe_train_val_split(
    samples: list[tuple[str, int]],
    val_split: float,
    seed: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    if len(samples) < 2:
        return samples, samples

    labels = np.array([label for _, label in samples])
    unique, counts = np.unique(labels, return_counts=True)
    val_size = max(1, int(round(len(samples) * val_split)))
    train_size = len(samples) - val_size
    can_stratify = (
        len(unique) > 1
        and np.all(counts >= 2)
        and val_size >= len(unique)
        and train_size >= len(unique)
    )

    indices = np.arange(len(samples))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        random_state=seed,
        stratify=labels if can_stratify else None,
    )
    train_samples = [samples[idx] for idx in train_idx]
    val_samples = [samples[idx] for idx in val_idx]
    return train_samples, val_samples


def _build_loader(
    samples: list[tuple[str, int]],
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = BrainTumorDataset(samples=samples, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def prepare_client_partition(
    dataset_root: str,
    partition_id: int,
    num_clients: int,
    partition_mode: str,
    dirichlet_alpha: float,
    batch_size: int,
    val_split: float,
    test_split: float,
    num_workers: int,
    seed: int,
) -> ClientPartition:
    layout = discover_dataset_layout(dataset_root, test_split=test_split, seed=seed)
    partitions = create_client_partitions(
        samples=layout.train_samples,
        num_clients=num_clients,
        partition_mode=partition_mode,
        dirichlet_alpha=dirichlet_alpha,
        seed=seed,
    )
    client_indices = partitions[partition_id]
    client_samples = [layout.train_samples[idx] for idx in client_indices]
    local_train, local_val = _safe_train_val_split(
        client_samples, val_split=val_split, seed=seed + partition_id
    )

    train_loader = _build_loader(
        samples=local_train,
        transform=build_train_transform(),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = _build_loader(
        samples=local_val,
        transform=build_eval_transform(),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    test_loader = _build_loader(
        samples=layout.test_samples,
        transform=build_eval_transform(),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    label_histogram: dict[str, int] = {}
    for _, label in client_samples:
        label_name = layout.classes[label]
        label_histogram[label_name] = label_histogram.get(label_name, 0) + 1

    return ClientPartition(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=len(layout.classes),
        summary={
            "partition_id": partition_id,
            "num_samples": len(client_samples),
            "num_train": len(local_train),
            "num_val": len(local_val),
            "class_histogram": label_histogram,
            "classes": layout.classes,
        },
    )


def prepare_global_test_loader(
    dataset_root: str,
    batch_size: int,
    test_split: float,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, int]:
    layout = discover_dataset_layout(dataset_root, test_split=test_split, seed=seed)
    test_loader = _build_loader(
        samples=layout.test_samples,
        transform=build_eval_transform(),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    return test_loader, len(layout.classes)


def prepare_centralized_data(
    dataset_root: str,
    batch_size: int,
    val_split: float,
    test_split: float,
    num_workers: int,
    seed: int,
) -> CentralizedDataBundle:
    layout = discover_dataset_layout(dataset_root, test_split=test_split, seed=seed)
    train_samples, val_samples = _safe_train_val_split(
        layout.train_samples, val_split=val_split, seed=seed
    )

    train_loader = _build_loader(
        samples=train_samples,
        transform=build_train_transform(),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = _build_loader(
        samples=val_samples,
        transform=build_eval_transform(),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    test_loader = _build_loader(
        samples=layout.test_samples,
        transform=build_eval_transform(),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return CentralizedDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=len(layout.classes),
        classes=layout.classes,
        summary={
            "num_train": len(train_samples),
            "num_val": len(val_samples),
            "num_test": len(layout.test_samples),
            "classes": layout.classes,
        },
    )
