from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from laion_people_api import QualifiedImagePool, ServiceConfig
from people_video_api import VideoCache


def create_app(image_pool: QualifiedImagePool, video_cache: VideoCache) -> FastAPI:
    app = FastAPI(title="People Media Cache API")

    @app.on_event("startup")
    def startup() -> None:
        image_pool.start_cache_refresh()
        video_cache.start()

    @app.on_event("shutdown")
    def shutdown() -> None:
        image_pool.stop()
        video_cache.stop()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "images": image_pool.status(),
            "videos": video_cache.status(),
        }

    @app.get("/api/image")
    def image(request: Request) -> dict[str, Any]:
        items = image_pool.get_random(1)
        if not items:
            raise HTTPException(status_code=503, detail="No qualified image is cached yet.")
        return _public_item(items[0], request)

    @app.get("/api/images")
    def images(request: Request, count: int = 5) -> dict[str, Any]:
        count = max(1, min(count, 50))
        items = image_pool.get_random(count)
        if not items:
            raise HTTPException(status_code=503, detail="No qualified images are cached yet.")
        return {"count": len(items), "items": [_public_item(item, request) for item in items]}

    @app.get("/api/video")
    def video(request: Request) -> dict[str, Any]:
        items = video_cache.random_items(1)
        if not items:
            raise HTTPException(status_code=503, detail="No qualified video is cached yet.")
        return _public_item(items[0], request)

    @app.get("/api/videos")
    def videos(request: Request, count: int = 5) -> dict[str, Any]:
        count = max(1, min(count, 50))
        items = video_cache.random_items(count)
        if not items:
            raise HTTPException(status_code=503, detail="No qualified videos are cached yet.")
        return {"count": len(items), "items": [_public_item(item, request) for item in items]}

    @app.get("/files/{image_id}")
    def image_file(image_id: str) -> FileResponse:
        try:
            path = image_pool.resolve_file(image_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Image not found") from exc
        return FileResponse(path)

    @app.get("/video-files/{video_id}")
    def video_file(video_id: str) -> FileResponse:
        try:
            path = video_cache.resolve_file(video_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Video not found") from exc
        return FileResponse(path, media_type="video/mp4")

    return app


def _public_item(item: object, request: Request) -> dict[str, Any]:
    data = item.__dict__.copy()
    data["download_url"] = str(request.base_url).rstrip("/") + data["download_url"]
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve random cached people images and videos from one API.")
    parser.add_argument("--image-output-dir", type=str, default="outputs/laion_people_api")
    parser.add_argument("--video-output-dir", type=str, default="outputs/people_video_api")
    parser.add_argument("--image-max-cache", type=int, default=5000)
    parser.add_argument("--video-max-cache", type=int, default=5000)
    parser.add_argument("--refresh-seconds", type=float, default=5.0)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_config = ServiceConfig(
        mode="serve",
        dataset="",
        split="train",
        output_dir=Path(args.image_output_dir),
        max_cache=args.image_max_cache,
        workers=0,
        request_timeout=0.0,
        min_width=0,
        min_height=0,
        min_person_height_ratio=0.0,
        max_detection_side=0,
        detection_mode="face-only",
        max_image_bytes=0,
        shuffle_buffer=0,
        seed=42,
        allow_text_body_fallback=True,
        cache_refresh_seconds=args.refresh_seconds,
        hf_token=None,
    )
    image_pool = QualifiedImagePool(image_config)
    video_cache = VideoCache(Path(args.video_output_dir), max_cache=args.video_max_cache, refresh_seconds=args.refresh_seconds)
    uvicorn.run(create_app(image_pool, video_cache), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
