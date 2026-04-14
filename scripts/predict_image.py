from __future__ import annotations

import argparse
from pathlib import Path
from tkinter import Tk, filedialog

import torch
from PIL import Image

from brain_tumor_fl.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    discover_dataset_layout,
)
from brain_tumor_fl.model import build_model
from brain_tumor_fl.training import get_device
from torchvision import transforms


DEFAULT_CHECKPOINT = Path("outputs/checkpoints/best_model.pt")


def build_inference_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def select_image_via_dialog(initial_dir: str | None = None) -> Path | None:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Select MRI image",
        initialdir=initial_dir,
        filetypes=[
            ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not file_path:
        return None
    return Path(file_path)


def infer_true_label(image_path: Path, classes: list[str]) -> str | None:
    parent_name = image_path.parent.name.lower()
    normalized = {class_name.lower(): class_name for class_name in classes}
    return normalized.get(parent_name)


def load_checkpoint(checkpoint_path: Path) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_path}' was not found. "
            "Run federated training first so the best model can be saved."
        )
    return torch.load(checkpoint_path, map_location="cpu")


def resolve_classes(checkpoint: dict, dataset_root: str | None) -> list[str]:
    classes = checkpoint.get("classes")
    if classes:
        return list(classes)
    if dataset_root is None:
        raise ValueError(
            "Checkpoint does not contain class labels. Please provide --dataset-root."
        )
    layout = discover_dataset_layout(dataset_root=dataset_root, test_split=0.15, seed=42)
    return layout.classes


def predict_image(
    image_path: Path,
    checkpoint_path: Path,
    dataset_root: str | None = None,
) -> None:
    checkpoint = load_checkpoint(checkpoint_path)
    classes = resolve_classes(checkpoint, dataset_root)
    device = get_device()

    model = build_model(
        num_classes=len(classes),
        use_pretrained=bool(checkpoint.get("use_pretrained", True)),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()

    transform = build_inference_transform()
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()

    predicted_index = int(torch.argmax(probabilities).item())
    predicted_class = classes[predicted_index]
    confidence = float(probabilities[predicted_index].item())

    true_label = infer_true_label(image_path, classes)

    print("=" * 72)
    print(f"Image: {image_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model round: {checkpoint.get('round', 'unknown')}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    if true_label is not None:
        print(f"True class: {true_label}")
        print(f"Correct prediction: {'yes' if true_label == predicted_class else 'no'}")
    else:
        print("True class: unavailable (image is not inside a class folder)")
    print("-" * 72)
    print("Class probabilities:")
    for class_name, probability in sorted(
        zip(classes, probabilities.tolist(), strict=True),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"  {class_name:12s} {probability:.4f}")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open an MRI image through a file dialog or pass a path explicitly, "
            "then run inference with the saved federated checkpoint."
        )
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an image file. If omitted, a file dialog will open.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to the saved checkpoint (.pt).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="brain_tumor_mri",
        help="Dataset root used only to recover class names when needed.",
    )
    parser.add_argument(
        "--initial-dir",
        type=str,
        default="brain_tumor_mri",
        help="Initial folder opened by the file dialog.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image) if args.image else select_image_via_dialog(args.initial_dir)
    if image_path is None:
        print("No image selected.")
        return
    predict_image(
        image_path=image_path,
        checkpoint_path=Path(args.checkpoint),
        dataset_root=args.dataset_root,
    )


if __name__ == "__main__":
    main()
