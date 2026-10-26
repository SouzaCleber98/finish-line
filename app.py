import cv2
import math

class MotionObjectDetector:
    def __init__(
        self,
        minimum_area=500,
        max_distance=80,
        forgotten_frames_limit=20,
        warmup_frames=30,
    ):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120,
            varThreshold=20,
            detectShadows=False,
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.minimum_area = minimum_area
        self.max_distance = max_distance
        self.forgotten_frames_limit = forgotten_frames_limit
        self.warmup_frames = warmup_frames
        self.processed_frames = 0
        self.next_object_id = 1
        self.object_trails = {}

    def update_frame(self, frame):
        """Detect moving objects and return boxes as (x1, y1, x2, y2, object_id)."""
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty image")

        mask = self.background_subtractor.apply(frame, learningRate=0.02)
        self.processed_frames += 1

        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, self.kernel, iterations=1
        )
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for contour in contours:
            if cv2.contourArea(contour) < self.minimum_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            center = (x + width // 2, y + height // 2)
            detections.append((x, y, x + width, y + height, center))

        if self.processed_frames <= self.warmup_frames:
            return []

        active_ids = set()
        detected_objects = []

        for x1, y1, x2, y2, center in detections:
            matched_id = None
            shortest_distance = self.max_distance

            for object_id, trail in self.object_trails.items():
                if object_id in active_ids:
                    continue

                previous_x, previous_y, _ = trail
                distance = math.hypot(
                    center[0] - previous_x,
                    center[1] - previous_y,
                )
                if distance < shortest_distance:
                    matched_id = object_id
                    shortest_distance = distance

            if matched_id is None:
                matched_id = self.next_object_id
                self.next_object_id += 1

            active_ids.add(matched_id)
            self.object_trails[matched_id] = (
                center[0],
                center[1],
                self.processed_frames,
            )
            detected_objects.append((x1, y1, x2, y2, matched_id))

        self.object_trails = {
            object_id: trail
            for object_id, trail in self.object_trails.items()
            if self.processed_frames - trail[2] <= self.forgotten_frames_limit
        }

        return detected_objects

detector = MotionObjectDetector()
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError('Could not open camera.')

try:
    while True:
        success, frame = camera.read()

        if not success:
            raise RuntimeError('Failed to read frame from camera.')

        for x1, y1, x2, y2, object_id in detector.update_frame(frame):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"Object {object_id}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        cv2.imshow("Detector de movimento", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()