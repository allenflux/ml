from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError as exc:  # pragma: no cover - handled at runtime for clearer setup errors.
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None


Rect = tuple[int, int, int, int]


@dataclass(frozen=True)
class VisualDetectionResult:
    accepted: bool
    face_count: int
    person_count: int
    best_person_height_ratio: float
    face_boxes: list[Rect]
    person_boxes: list[Rect]
    reason: str


class VisualPersonFilter:
    def __init__(
        self,
        min_person_height_ratio: float = 0.45,
        require_face: bool = True,
        allow_text_body_fallback: bool = True,
        max_detection_side: int = 640,
    ) -> None:
        if cv2 is None:
            raise RuntimeError(
                "opencv-python-headless is required for visual detection. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from CV2_IMPORT_ERROR

        self.min_person_height_ratio = min_person_height_ratio
        self.require_face = require_face
        self.allow_text_body_fallback = allow_text_body_fallback
        self.max_detection_side = max_detection_side

        cv2.setNumThreads(1)

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.face_cascade.empty():
            raise RuntimeError(f"Unable to load OpenCV face cascade: {cascade_path}")

    def inspect(self, image: Image.Image, *, full_body_text_hint: bool) -> VisualDetectionResult:
        rgb = image.convert("RGB")
        width, height = rgb.size
        resized = _resize_for_detection(rgb, max_side=self.max_detection_side)
        scale_x = width / resized.size[0]
        scale_y = height / resized.size[1]

        frame = np.array(resized)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        face_boxes_small = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(28, 28),
        )
        face_boxes = [_scale_rect(tuple(map(int, rect)), scale_x, scale_y) for rect in face_boxes_small]

        person_boxes_small, _ = self.hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(16, 16),
            scale=1.05,
        )
        person_boxes = [_scale_rect(tuple(map(int, rect)), scale_x, scale_y) for rect in person_boxes_small]
        person_boxes = _suppress_nested_boxes(person_boxes)

        best_ratio = max((box[3] / height for box in person_boxes), default=0.0)
        has_face = len(face_boxes) > 0
        has_full_body_person = best_ratio >= self.min_person_height_ratio

        if self.require_face and not has_face:
            return VisualDetectionResult(False, 0, len(person_boxes), best_ratio, face_boxes, person_boxes, "no_face_detected")

        if has_full_body_person:
            return VisualDetectionResult(True, len(face_boxes), len(person_boxes), best_ratio, face_boxes, person_boxes, "person_and_face_detected")

        if self.allow_text_body_fallback and full_body_text_hint and has_face:
            return VisualDetectionResult(True, len(face_boxes), len(person_boxes), best_ratio, face_boxes, person_boxes, "face_detected_text_full_body_hint")

        return VisualDetectionResult(False, len(face_boxes), len(person_boxes), best_ratio, face_boxes, person_boxes, "no_full_body_person_detected")


def _resize_for_detection(image: Image.Image, max_side: int = 900) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))))


def _scale_rect(rect: Rect, scale_x: float, scale_y: float) -> Rect:
    x, y, width, height = rect
    return (
        int(x * scale_x),
        int(y * scale_y),
        int(width * scale_x),
        int(height * scale_y),
    )


def _suppress_nested_boxes(boxes: list[Rect]) -> list[Rect]:
    kept: list[Rect] = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        if not any(_mostly_inside(box, existing) for existing in kept):
            kept.append(box)
    return kept


def _mostly_inside(inner: Rect, outer: Rect) -> bool:
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    inter_left = max(ix, ox)
    inter_top = max(iy, oy)
    inter_right = min(ix + iw, ox + ow)
    inter_bottom = min(iy + ih, oy + oh)
    if inter_right <= inter_left or inter_bottom <= inter_top:
        return False
    intersection = (inter_right - inter_left) * (inter_bottom - inter_top)
    inner_area = iw * ih
    return inner_area > 0 and intersection / inner_area > 0.75
