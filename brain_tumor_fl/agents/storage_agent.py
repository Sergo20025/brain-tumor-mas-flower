from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.utils.data import DataLoader

from brain_tumor_fl.data import prepare_client_partition, prepare_global_test_loader
from brain_tumor_fl.utils import format_class_histogram, print_agent_log


@dataclass
class StorageAgent:
    config: dict[str, Any]

    def load_partition(self, partition_id: int) -> dict[str, Any]:
        partition = prepare_client_partition(
            dataset_root=str(self.config["dataset-root"]),
            partition_id=partition_id,
            num_clients=int(self.config["num-clients"]),
            partition_mode=str(self.config["partition-mode"]),
            dirichlet_alpha=float(self.config["dirichlet-alpha"]),
            soft_mix_ratio=float(self.config.get("soft-mix-ratio", 0.15)),
            soft_min_extra_classes=int(self.config.get("soft-min-extra-classes", 5)),
            batch_size=int(self.config["batch-size"]),
            val_split=float(self.config["val-split"]),
            test_split=float(self.config["test-split"]),
            num_workers=int(self.config["num-workers"]),
            seed=int(self.config["random-seed"]),
        )
        print_agent_log(
            "StorageAgent",
            (
                f"partition loaded: train={partition.summary['num_train']}, "
                f"val={partition.summary['num_val']}, "
                f"samples={partition.summary['num_samples']}, "
                f"class_hist=[{format_class_histogram(partition.summary['class_histogram'])}]"
            ),
            partition_id=partition_id,
        )
        return {
            "train_loader": partition.train_loader,
            "val_loader": partition.val_loader,
            "test_loader": partition.test_loader,
            "num_classes": partition.num_classes,
            "summary": partition.summary,
        }

    def load_global_test_loader(self) -> tuple[DataLoader, int]:
        test_loader, num_classes = prepare_global_test_loader(
            dataset_root=str(self.config["dataset-root"]),
            batch_size=int(self.config["batch-size"]),
            test_split=float(self.config["test-split"]),
            num_workers=int(self.config["num-workers"]),
            seed=int(self.config["random-seed"]),
        )
        print_agent_log(
            "StorageAgent",
            f"global test loader ready: size={len(test_loader.dataset)}, num_classes={num_classes}",
        )
        return test_loader, num_classes
