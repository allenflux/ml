from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import uvicorn
from datasets import load_dataset
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image

from filter_laion_people import (
    build_session,
    download_image,
    get_text,
    get_url,
    guess_extension,
    image_too_small,
    looks_like_candidate,
)
from src.visual_person_filter import VisualPersonFilter


@dataclass(frozen=True)
class QualifiedImage:
    id: str
    download_url: str
    source_url: str
    text: str
    width: int
    height: int
    content_type: str
    face_count: int
    person_count: int
    detection_reason: str


@dataclass(frozen=True)
class ServiceConfig:
    mode: str
    dataset: str
    split: str
    output_dir: Path
    max_cache: int
    workers: int
    request_timeout: float
    min_width: int
    min_height: int
    min_person_height_ratio: float
    max_detection_side: int
    shuffle_buffer: int
    seed: int
    allow_text_body_fallback: bool
    hf_token: str | None


class QualifiedImagePool:
    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self.images_dir = config.output_dir / "images"
        self.index_path = config.output_dir / "accepted.jsonl"
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self._items: list[QualifiedImage] = []
        self._seen_urls: set[str] = set()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._stats = {
            "checked": 0,
            "text_candidates": 0,
            "downloaded": 0,
            "accepted": 0,
            "rejected_visual": 0,
            "rejected_download": 0,
            "scan_errors": 0,
            "last_error": "",
            "last_stage": "not_started",
            "last_sample_at": 0.0,
        }

    def load_cache(self) -> None:
        self._load_existing()

    def start(self) -> None:
        if self._threads:
            return
        self._load_existing()
        for worker_id in range(self.config.workers):
            thread = threading.Thread(target=self._scan_loop, args=(worker_id,), daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop_event.set()

    def status(self) -> dict[str, Any]:
        with self._lock:
            last_sample_at = float(self._stats["last_sample_at"])
            return {
                "cached": len(self._items),
                "workers": len(self._threads),
                "seconds_since_last_sample": round(time.time() - last_sample_at, 2) if last_sample_at else None,
                **self._stats,
            }

    def get_random(self, count: int = 1) -> list[QualifiedImage]:
        with self._lock:
            if not self._items:
                return []
            if count >= len(self._items):
                items = self._items[:]
                random.shuffle(items)
                return items
            return random.sample(self._items, count)

    def wait_for_items(self, count: int, timeout_seconds: float) -> list[QualifiedImage]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            items = self.get_random(count)
            if len(items) >= min(count, 1):
                return items[:count]
            time.sleep(0.1)
        return self.get_random(count)

    def resolve_file(self, image_id: str) -> Path:
        path = self.images_dir / image_id
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(image_id)
        return path

    def _scan_loop(self, worker_id: int) -> None:
        session = build_session()
        detector = VisualPersonFilter(
            min_person_height_ratio=self.config.min_person_height_ratio,
            allow_text_body_fallback=self.config.allow_text_body_fallback,
            max_detection_side=self.config.max_detection_side,
        )
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    self._stats["last_stage"] = f"worker_{worker_id}_loading_dataset"
                dataset = load_dataset(
                    self.config.dataset,
                    split=self.config.split,
                    streaming=True,
                    token=self.config.hf_token,
                )
                if self.config.shuffle_buffer > 1:
                    with self._lock:
                        self._stats["last_stage"] = f"worker_{worker_id}_shuffling_stream"
                    dataset = dataset.shuffle(buffer_size=self.config.shuffle_buffer, seed=self.config.seed + worker_id)
                with self._lock:
                    self._stats["last_stage"] = f"worker_{worker_id}_waiting_first_sample"
                for sample in dataset:
                    if self._stop_event.is_set():
                        return
                    with self._lock:
                        self._stats["last_stage"] = f"worker_{worker_id}_scanning"
                        self._stats["last_sample_at"] = time.time()
                    self._handle_sample(sample, session, detector)
            except Exception as exc:
                message = str(exc)
                with self._lock:
                    self._stats["scan_errors"] += 1
                    self._stats["last_error"] = message
                print(f"worker={worker_id} scan_error={message}")
                time.sleep(2.0)

    def _handle_sample(self, sample: dict[str, Any], session: Any, detector: VisualPersonFilter) -> None:
        with self._lock:
            self._stats["checked"] += 1
            cache_full = len(self._items) >= self.config.max_cache
        if cache_full:
            time.sleep(0.5)
            return

        text = get_text(sample)
        url = get_url(sample)
        if not url:
            return

        with self._lock:
            if url in self._seen_urls:
                return
            self._seen_urls.add(url)

        keep_text, hints = looks_like_candidate(text)
        if not keep_text:
            return

        with self._lock:
            self._stats["text_candidates"] += 1

        image, image_bytes, _error = download_image(session, url, timeout=self.config.request_timeout)
        if image is None or image_bytes is None:
            with self._lock:
                self._stats["rejected_download"] += 1
            return

        if image_too_small(image, self.config.min_width, self.config.min_height):
            return

        with self._lock:
            self._stats["downloaded"] += 1

        detection = detector.inspect(image, full_body_text_hint=hints["full_body"])
        if not detection.accepted:
            with self._lock:
                self._stats["rejected_visual"] += 1
            return

        item = self._save_item(url, text, image, image_bytes, detection)
        with self._lock:
            self._items.append(item)
            self._items = self._items[-self.config.max_cache :]
            self._stats["accepted"] += 1
        self._append_index(item)

    def _save_item(self, source_url: str, text: str, image: Image.Image, image_bytes: bytes, detection: Any) -> QualifiedImage:
        content_type = Image.MIME.get(image.format, "image/unknown")
        extension = guess_extension(content_type, source_url)
        name = f"{hashlib.sha1(source_url.encode('utf-8')).hexdigest()[:24]}.{extension}"
        path = self.images_dir / name
        path.write_bytes(image_bytes)
        return QualifiedImage(
            id=name,
            download_url=f"/files/{name}",
            source_url=source_url,
            text=text,
            width=image.width,
            height=image.height,
            content_type=content_type,
            face_count=detection.face_count,
            person_count=detection.person_count,
            detection_reason=detection.reason,
        )

    def _append_index(self, item: QualifiedImage) -> None:
        with self.index_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    def _load_existing(self) -> None:
        if not self.index_path.exists():
            return
        with self.index_path.open("r", encoding="utf-8") as file:
            rows = [json.loads(line) for line in file if line.strip()]
        items = [QualifiedImage(**row) for row in rows if (self.images_dir / row["id"]).exists()]
        with self._lock:
            self._items = items[-self.config.max_cache :]
            self._seen_urls.update(item.source_url for item in self._items)
            self._stats["accepted"] = len(self._items)


def create_app(config: ServiceConfig) -> FastAPI:
    app = FastAPI(title="LAION Qualified People Image API")
    pool = QualifiedImagePool(config)

    @app.on_event("startup")
    def startup() -> None:
        if config.mode == "serve":
            pool.load_cache()
        else:
            pool.start()

    @app.on_event("shutdown")
    def shutdown() -> None:
        pool.stop()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, **pool.status()}

    @app.get("/api/image")
    def get_image(request: Request, timeout_seconds: float = 5.0) -> dict[str, Any]:
        items = pool.wait_for_items(count=1, timeout_seconds=timeout_seconds)
        if not items:
            raise HTTPException(status_code=503, detail="No qualified image is cached yet. Try again in a few seconds.")
        return _public_item(items[0], request)

    @app.get("/api/images")
    def get_images(request: Request, count: int = 5, timeout_seconds: float = 5.0) -> dict[str, Any]:
        count = max(1, min(count, 50))
        items = pool.wait_for_items(count=count, timeout_seconds=timeout_seconds)
        if not items:
            raise HTTPException(status_code=503, detail="No qualified images are cached yet. Try again in a few seconds.")
        return {"count": len(items), "items": [_public_item(item, request) for item in items]}

    @app.get("/files/{image_id}")
    def download_file(image_id: str) -> FileResponse:
        try:
            path = pool.resolve_file(image_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Image not found") from exc
        return FileResponse(path)

    return app


def _public_item(item: QualifiedImage, request: Request) -> dict[str, Any]:
    data = asdict(item)
    data["download_url"] = str(request.base_url).rstrip("/") + item.download_url
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve qualified downloadable LAION people images.")
    parser.add_argument("--mode", type=str, default="hybrid", choices=("hybrid", "serve"))
    parser.add_argument("--dataset", type=str, default="laion/laion2B-en")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=str, default="outputs/laion_people_api")
    parser.add_argument("--max-cache", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=6.0)
    parser.add_argument("--min-width", type=int, default=384)
    parser.add_argument("--min-height", type=int, default=384)
    parser.add_argument("--min-person-height-ratio", type=float, default=0.42)
    parser.add_argument("--max-detection-side", type=int, default=640)
    parser.add_argument("--shuffle-buffer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-body-detection", action="store_true")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ServiceConfig(
        mode=args.mode,
        dataset=args.dataset,
        split=args.split,
        output_dir=Path(args.output_dir),
        max_cache=args.max_cache,
        workers=args.workers,
        request_timeout=args.request_timeout,
        min_width=args.min_width,
        min_height=args.min_height,
        min_person_height_ratio=args.min_person_height_ratio,
        max_detection_side=args.max_detection_side,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        allow_text_body_fallback=not args.strict_body_detection,
        hf_token=args.hf_token,
    )
    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
