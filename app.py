from dataclasses import dataclass
import os
from typing import Any

import cv2


@dataclass(frozen=True)
class Detection:
    """A detected object bounding box and its confidence."""

    x1: int
    y1: int
    x2: int
    y2: int
    class_name: str
    confidence: float


class ObjectDetector:
    def __init__(
        self,
        target_class: str = "toy-car",
        model_path: str = "runs/detect/toy-cars/weights/best.pt",
        confidence: float = 0.75,
        image_size: int = 640,
    ) -> None:
        if not target_class.strip():
            raise ValueError("target_class must not be empty")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if image_size <= 0:
            raise ValueError("image_size must be greater than zero")

        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is not installed. Run: "
                "python -m pip install -r requirements.txt"
            ) from error

        self.model = YOLO(model_path)
        self.target_class = target_class.strip()
        self.confidence = confidence
        self.image_size = image_size
        self.class_names = self._normalize_class_names(self.model.names)

        matching_ids = [
            class_id
            for class_id, class_name in self.class_names.items()
            if class_name.casefold() == self.target_class.casefold()
        ]
        if not matching_ids:
            available_classes = ", ".join(self.class_names.values())
            raise ValueError(
                f"Unknown class '{self.target_class}'. "
                f"Available classes: {available_classes}"
            )

        self.target_class_id = matching_ids[0]
        self.target_class = self.class_names[self.target_class_id]

    @staticmethod
    def _normalize_class_names(names: Any) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(class_id): str(name) for class_id, name in names.items()}
        return {class_id: str(name) for class_id, name in enumerate(names)}

    def detect(self, frame: cv2.typing.MatLike) -> list[Detection]:
        """Detect only the configured class in a BGR camera frame."""
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty image")

        results = self.model.predict(
            source=frame,
            classes=[self.target_class_id],
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes
        coordinates = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()

        return [
            Detection(
                x1=int(coordinates[index][0]),
                y1=int(coordinates[index][1]),
                x2=int(coordinates[index][2]),
                y2=int(coordinates[index][3]),
                class_name=self.target_class,
                confidence=float(confidences[index]),
            )
            for index in range(len(coordinates))
        ]


def main() -> None:
    target_class = os.getenv("YOLO_TARGET_CLASS", "toy-car")
    model_path = os.getenv(
        "YOLO_MODEL_PATH",
        "runs/detect/toy-cars/weights/best.pt",
    )
    detector = ObjectDetector(
        target_class=target_class,
        model_path=model_path,
    )
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open camera.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Failed to read frame from camera.")

            for detection in detector.detect(frame):
                cv2.rectangle(
                    frame,
                    (detection.x1, detection.y1),
                    (detection.x2, detection.y2),
                    (0, 255, 0),
                    2,
                )
                label = f"{detection.class_name} {detection.confidence:.0%}"
                cv2.putText(
                    frame,
                    label,
                    (detection.x1, max(detection.y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("YOLO object detector - Q to quit", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
