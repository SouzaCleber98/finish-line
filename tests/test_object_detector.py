import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app import ObjectDetector


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    xyxy = FakeTensor([[12.8, 24.2, 110.9, 130.7]])
    conf = FakeTensor([0.91])


class FakeYOLO:
    def __init__(self, model_path):
        self.model_path = model_path
        self.names = {0: "person", 1: "toy-car"}

    def predict(self, **kwargs):
        self.predict_kwargs = kwargs
        return [SimpleNamespace(boxes=FakeBoxes())]


class ObjectDetectorTests(unittest.TestCase):
    def setUp(self):
        self.fake_module = SimpleNamespace(YOLO=FakeYOLO)
        self.modules_patch = patch.dict(
            sys.modules,
            {"ultralytics": self.fake_module},
        )
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)

    def test_detects_custom_class_from_custom_weights(self):
        detector = ObjectDetector(
            target_class="TOY-CAR",
            model_path="custom.pt",
        )
        detections = detector.detect(
            np.zeros((160, 160, 3), dtype=np.uint8)
        )

        self.assertEqual(detector.model.model_path, "custom.pt")
        self.assertEqual(detector.model.predict_kwargs["classes"], [1])
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "toy-car")
        self.assertEqual(
            (detections[0].x1, detections[0].y1, detections[0].x2, detections[0].y2),
            (12, 24, 110, 130),
        )
        self.assertEqual(detections[0].confidence, 0.91)

    def test_defaults_to_the_trained_toy_car_model(self):
        detector = ObjectDetector()

        self.assertEqual(
            detector.model.model_path,
            "runs/detect/toy-cars/weights/best.pt",
        )
        self.assertEqual(detector.target_class, "toy-car")

    def test_rejects_class_missing_from_selected_model(self):
        with self.assertRaisesRegex(ValueError, "Available classes"):
            ObjectDetector(target_class="hand")

    def test_rejects_empty_frame(self):
        detector = ObjectDetector(target_class="toy-car")
        with self.assertRaisesRegex(ValueError, "non-empty image"):
            detector.detect(np.empty((0, 0, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
