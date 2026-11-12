import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a YOLO detector on a custom object dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datasets/toy_car/data.yaml"),
        help="YOLO dataset YAML file (default: datasets/toy_car/data.yaml).",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Pretrained model or weights file used to start training.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--name", default="toy-cars")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be greater than zero")
    if args.image_size < 1:
        parser.error("--image-size must be greater than zero")
    if not args.data.is_file():
        parser.error(f"Dataset YAML does not exist: {args.data}")
    return args


def train(
    data: Path,
    model_path: str,
    epochs: int,
    image_size: int,
    project: Path,
    name: str,
) -> object:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is not installed. Run: "
            "python -m pip install -r requirements.txt"
        ) from error

    model = YOLO(model_path)
    return model.train(
        data=str(data),
        epochs=epochs,
        imgsz=image_size,
        project=str(project),
        name=name,
    )


def main() -> None:
    args = parse_args()
    train(
        data=args.data,
        model_path=args.model,
        epochs=args.epochs,
        image_size=args.image_size,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
