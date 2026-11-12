import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from train_model import parse_args, train


class FakeYOLO:
    def __init__(self, model_path):
        self.model_path = model_path
        self.train_kwargs = None

    def train(self, **kwargs):
        self.train_kwargs = kwargs
        return "training result"


class TrainModelTests(unittest.TestCase):
    def test_defaults_to_the_roboflow_toy_car_dataset(self):
        with patch.object(sys, "argv", ["train_model.py"]):
            args = parse_args()

        self.assertEqual(args.data, Path("datasets/toy_car/data.yaml"))
        self.assertEqual(args.project, Path("."))
        self.assertEqual(args.name, "toy-cars")

    def test_train_uses_requested_dataset_and_options(self):
        fake_model = FakeYOLO("starting.pt")
        fake_module = SimpleNamespace(YOLO=lambda path: fake_model)

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_file = Path(temporary_directory) / "dataset.yaml"
            data_file.write_text("names: {0: toy_car}\n", encoding="utf-8")

            with patch.dict(sys.modules, {"ultralytics": fake_module}):
                result = train(
                    data=data_file,
                    model_path="starting.pt",
                    epochs=12,
                    image_size=320,
                    project=Path("runs/detect"),
                    name="toy-car",
                )

        self.assertEqual(result, "training result")
        self.assertEqual(
            fake_model.train_kwargs,
            {
                "data": str(data_file),
                "epochs": 12,
                "imgsz": 320,
                "project": str(Path("runs/detect")),
                "name": "toy-car",
            },
        )


if __name__ == "__main__":
    unittest.main()
