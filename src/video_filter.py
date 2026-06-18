from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import requests
from PIL import Image

from src.visual_person_filter import VisualPersonFilter


@dataclass(frozen=True)
class VideoDetectionResult:
    accepted: bool
    duration_seconds: float
    width: int
    height: int
    sampled_frames: int
    face_hits: int
    reason: str


def iter_video_source_rows(path: str | Path) -> Iterable[dict[str, str]]:
    source_path = Path(path)
    if source_path.suffix.lower() == ".jsonl":
        with source_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    row = json.loads(line)
                    yield {key: str(value) for key, value in row.items() if value is not None}
        return

    import csv

    with source_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            yield {key: str(value) for key, value in row.items() if value is not None}


def get_video_url(row: dict[str, str]) -> str:
    for key in ("video_url", "url", "content_url", "download_url"):
        value = row.get(key)
        if value:
            return value.strip()
    return ""


def get_video_text(row: dict[str, str]) -> str:
    for key in ("text", "caption", "description", "title"):
        value = row.get(key)
        if value:
            return value.strip()
    return ""


def download_video(
    session: requests.Session,
    url: str,
    output_path: Path,
    timeout: float,
    max_bytes: int,
) -> tuple[bool, str, str]:
    try:
        response = session.get(url, timeout=timeout, stream=True, allow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "video" not in content_type.lower() and "octet-stream" not in content_type.lower():
            return False, content_type, f"non-video content-type: {content_type or 'unknown'}"

        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            return False, content_type, f"video too large: {content_length} bytes"

        total_bytes = 0
        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    output_path.unlink(missing_ok=True)
                    return False, content_type, f"video too large: exceeded {max_bytes} bytes"
                file.write(chunk)
        return True, content_type, ""
    except requests.RequestException as exc:
        output_path.unlink(missing_ok=True)
        return False, "", str(exc)


def inspect_video(
    video_path: str | Path,
    detector: VisualPersonFilter,
    min_duration: float,
    max_duration: float,
    sample_frames: int,
) -> VideoDetectionResult:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return VideoDetectionResult(False, 0.0, 0, 0, 0, 0, "cannot_open_video")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0

    if duration < min_duration:
        capture.release()
        return VideoDetectionResult(False, duration, width, height, 0, 0, "too_short")
    if duration > max_duration:
        capture.release()
        return VideoDetectionResult(False, duration, width, height, 0, 0, "too_long")

    positions = _sample_positions(frame_count=int(frame_count), sample_frames=sample_frames)
    face_hits = 0
    sampled = 0
    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        ok, frame = capture.read()
        if not ok:
            continue
        sampled += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        result = detector.inspect(image, full_body_text_hint=True)
        if result.face_count > 0:
            face_hits += 1
    capture.release()

    if sampled == 0:
        return VideoDetectionResult(False, duration, width, height, sampled, face_hits, "no_readable_frames")
    if face_hits == 0:
        return VideoDetectionResult(False, duration, width, height, sampled, face_hits, "no_person_face_frames")

    return VideoDetectionResult(True, duration, width, height, sampled, face_hits, "person_face_detected")


def _sample_positions(frame_count: int, sample_frames: int) -> list[int]:
    if frame_count <= 0:
        return [0]
    if sample_frames <= 1:
        return [frame_count // 2]
    return [min(frame_count - 1, int(frame_count * (index + 1) / (sample_frames + 1))) for index in range(sample_frames)]
