from __future__ import annotations

import argparse
import json
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse


@dataclass(frozen=True)
class CachedVideo:
    id: str
    download_url: str
    source_url: str
    text: str
    duration_seconds: float
    width: int
    height: int
    content_type: str
    size_bytes: int
    sampled_frames: int
    face_hits: int
    detection_reason: str


class VideoCache:
    def __init__(self, output_dir: Path, max_cache: int, refresh_seconds: float) -> None:
        self.output_dir = output_dir
        self.videos_dir = output_dir / "videos"
        self.index_path = output_dir / "accepted_videos.jsonl"
        self.max_cache = max_cache
        self.refresh_seconds = max(1.0, refresh_seconds)
        self._items: list[CachedVideo] = []
        self._signature: tuple[int, int] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_reload_at = 0.0

    def start(self) -> None:
        self.reload()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "cached_videos": len(self._items),
                "last_reload_at": self.last_reload_at,
                "index_path": str(self.index_path),
            }

    def random_items(self, count: int) -> list[CachedVideo]:
        with self._lock:
            if not self._items:
                return []
            if count >= len(self._items):
                items = self._items[:]
                random.shuffle(items)
                return items
            return random.sample(self._items, count)

    def resolve_file(self, video_id: str) -> Path:
        path = self.videos_dir / video_id
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(video_id)
        return path

    def reload(self) -> None:
        if not self.index_path.exists():
            return
        rows: list[CachedVideo] = []
        with self.index_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                data = json.loads(line)
                if (self.videos_dir / data["id"]).exists():
                    rows.append(CachedVideo(**data))
        with self._lock:
            self._items = rows[-self.max_cache :]
            self._signature = self._get_signature()
            self.last_reload_at = time.time()

    def _loop(self) -> None:
        while not self._stop.is_set():
            signature = self._get_signature()
            if signature != self._signature:
                self.reload()
            time.sleep(self.refresh_seconds)

    def _get_signature(self) -> tuple[int, int] | None:
        if not self.index_path.exists():
            return None
        stat = self.index_path.stat()
        return int(stat.st_mtime_ns), int(stat.st_size)


def create_app(cache: VideoCache) -> FastAPI:
    app = FastAPI(title="People Video Cache API")

    @app.on_event("startup")
    def startup() -> None:
        cache.start()

    @app.on_event("shutdown")
    def shutdown() -> None:
        cache.stop()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, **cache.status()}

    @app.get("/api/video")
    def get_video(request: Request) -> dict[str, Any]:
        items = cache.random_items(1)
        if not items:
            raise HTTPException(status_code=503, detail="No qualified video is cached yet.")
        return _public_item(items[0], request)

    @app.get("/api/videos")
    def get_videos(request: Request, count: int = 5) -> dict[str, Any]:
        count = max(1, min(count, 50))
        items = cache.random_items(count)
        if not items:
            raise HTTPException(status_code=503, detail="No qualified videos are cached yet.")
        return {"count": len(items), "items": [_public_item(item, request) for item in items]}

    @app.get("/video-files/{video_id}")
    def video_file(video_id: str) -> FileResponse:
        try:
            path = cache.resolve_file(video_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Video not found") from exc
        return FileResponse(path, media_type="video/mp4")

    return app


def _public_item(item: CachedVideo, request: Request) -> dict[str, Any]:
    data = item.__dict__.copy()
    data["download_url"] = str(request.base_url).rstrip("/") + item.download_url
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve random cached people videos.")
    parser.add_argument("--output-dir", type=str, default="outputs/people_video_api")
    parser.add_argument("--max-cache", type=int, default=5000)
    parser.add_argument("--refresh-seconds", type=float, default=5.0)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = VideoCache(Path(args.output_dir), max_cache=args.max_cache, refresh_seconds=args.refresh_seconds)
    uvicorn.run(create_app(cache), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
