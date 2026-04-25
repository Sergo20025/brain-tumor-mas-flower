from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from tkinter import BOTH, Label, Tk, filedialog

import torch
from PIL import Image, ImageTk
from torchvision import transforms

from brain_tumor_fl.data import IMAGENET_MEAN, IMAGENET_STD, discover_dataset_layout
from brain_tumor_fl.model import build_model
from brain_tumor_fl.training import get_device


DEFAULT_CHECKPOINT = Path("outputs/checkpoints/best_model.pt")
DISPLAY_NAMES = {
    "glioma": "Глиома",
    "meningioma": "Менингиома",
    "pituitary": "Опухоль гипофиза",
    "notumor": "Опухоль не обнаружена",
}


@dataclass
class PredictionResult:
    predicted_class: str
    predicted_display_name: str
    confidence: float
    true_label: str | None
    probabilities: list[tuple[str, str, float]]
    checkpoint_round: int | str


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


def class_display_name(class_name: str) -> str:
    return DISPLAY_NAMES.get(class_name.lower(), class_name)


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
) -> PredictionResult:
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
    predicted_display_name = class_display_name(predicted_class)
    confidence = float(probabilities[predicted_index].item())
    true_label = infer_true_label(image_path, classes)

    probability_rows = [
        (class_name, class_display_name(class_name), probability)
        for class_name, probability in sorted(
            zip(classes, probabilities.tolist(), strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    print("=" * 72)
    print(f"Image: {image_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model round: {checkpoint.get('round', 'unknown')}")
    print(f"Predicted class: {predicted_class} ({predicted_display_name})")
    print(f"Confidence: {confidence:.4f}")
    if true_label is not None:
        print(f"True class: {true_label} ({class_display_name(true_label)})")
        print(f"Correct prediction: {'yes' if true_label == predicted_class else 'no'}")
    else:
        print("True class: unavailable (image is not inside a class folder)")
    print("-" * 72)
    print("Class probabilities:")
    for class_name, display_name, probability in probability_rows:
        print(f"  {class_name:12s} ({display_name}) {probability:.4f}")
    print("=" * 72)

    return PredictionResult(
        predicted_class=predicted_class,
        predicted_display_name=predicted_display_name,
        confidence=confidence,
        true_label=true_label,
        probabilities=probability_rows,
        checkpoint_round=checkpoint.get("round", "unknown"),
    )


def preview_image(image_path: Path, max_size: tuple[int, int] = (520, 520)) -> ImageTk.PhotoImage:
    preview = Image.open(image_path).convert("RGB")
    preview.thumbnail(max_size)
    return ImageTk.PhotoImage(preview)


def show_prediction_window(image_path: Path, result: PredictionResult) -> None:
    root = Tk()
    root.title("Brain Tumor MRI Prediction")
    root.geometry("720x860")
    root.minsize(560, 700)

    image_tk = preview_image(image_path)
    image_label = Label(root, image=image_tk)
    image_label.image = image_tk
    image_label.pack(padx=16, pady=(16, 12), fill=BOTH)

    headline = (
        f"Результат: {result.predicted_display_name}\n"
        f"Точность определения: {result.confidence * 100:.2f}%"
    )
    Label(
        root,
        text=headline,
        justify="center",
        font=("Segoe UI", 16, "bold"),
        wraplength=640,
    ).pack(padx=16, pady=(0, 10))

    details_lines = [
        f"Файл: {image_path.name}",
        f"Раунд модели: {result.checkpoint_round}",
        f"Класс модели: {result.predicted_class}",
    ]
    if result.true_label is not None:
        true_display = class_display_name(result.true_label)
        is_correct = "Да" if result.true_label == result.predicted_class else "Нет"
        details_lines.append(f"Истинный класс: {true_display} ({result.true_label})")
        details_lines.append(f"Совпадение с истинным классом: {is_correct}")

    Label(
        root,
        text="\n".join(details_lines),
        justify="left",
        font=("Segoe UI", 11),
        wraplength=640,
        anchor="w",
    ).pack(padx=16, pady=(0, 12), fill=BOTH)

    probability_lines = ["Вероятности по классам:"]
    for _, display_name, probability in result.probabilities:
        probability_lines.append(f"{display_name}: {probability * 100:.2f}%")

    Label(
        root,
        text="\n".join(probability_lines),
        justify="left",
        font=("Consolas", 11),
        wraplength=640,
        anchor="w",
    ).pack(padx=16, pady=(0, 16), fill=BOTH)

    root.mainloop()


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

    result = predict_image(
        image_path=image_path,
        checkpoint_path=Path(args.checkpoint),
        dataset_root=args.dataset_root,
    )
    show_prediction_window(image_path=image_path, result=result)


if __name__ == "__main__":
    main()
