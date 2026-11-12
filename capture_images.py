import argparse
from datetime import datetime
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture images from a webcam for an object detection dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/toy_car/unannotated"),
        help="Directory where captured images are saved.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index to open (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera {args.camera}.")

    print(f"Saving images to: {args.output_dir.resolve()}")
    print("Press S to save the current frame, or Q/Esc to quit.")
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Failed to read frame from camera.")

            preview = frame.copy()
            cv2.putText(
                preview,
                "S: save image | Q/Esc: quit",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Capture training images", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                image_path = args.output_dir / f"capture_{timestamp}.jpg"
                if not cv2.imwrite(str(image_path), frame):
                    raise OSError(f"Could not save image to {image_path}.")
                print(f"Saved: {image_path}")
            elif key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
