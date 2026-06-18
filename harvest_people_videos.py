from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets import load_dataset

from filter_laion_people import build_session
from src.video_filter import download_video, get_video_text, get_video_url, inspect_video, iter_video_source_rows
from src.visual_person_filter import VisualPersonFilter


@dataclass(frozen=True)
class QualifiedVideo:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest short people videos into a local cache.")
    parser.add_argument("--source-path", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--shuffle-buffer", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs/people_video_api")
    parser.add_argument("--target-cache", type=int, default=500)
    parser.add_argument("--request-timeout", type=float, default=12.0)
    parser.add_argument("--max-video-bytes", type=int, default=100_000_000)
    parser.add_argument("--min-duration", type=float, default=4.0)
    parser.add_argument("--max-duration", type=float, default=7.0)
    parser.add_argument("--sample-frames", type=int, default=5)
    parser.add_argument("--max-detection-side", type=int, default=384)
    parser.add_argument("--report-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    videos_dir = output_dir / "videos"
    index_path = output_dir / "accepted_videos.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    existing = _load_existing(index_path, videos_dir)
    seen_urls = {item["source_url"] for item in existing}
    accepted_count = len(existing)

    session = build_session()
    detector = VisualPersonFilter(
        allow_text_body_fallback=True,
        max_detection_side=args.max_detection_side,
        detection_mode="face-only",
    )

    checked = 0
    downloaded = 0
    rejected_download = 0
    rejected_visual = 0
    started_at = time.monotonic()

    source_rows = _iter_source_rows(args)

    with index_path.open("a", encoding="utf-8") as index_file:
        for row in source_rows:
            if accepted_count >= args.target_cache:
                break

            checked += 1
            url = get_video_url(row)
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            text = get_video_text(row)
            video_id = f"{hashlib.sha1(url.encode('utf-8')).hexdigest()[:24]}.mp4"
            video_path = videos_dir / video_id
            ok, content_type, error = download_video(
                session=session,
                url=url,
                output_path=video_path,
                timeout=args.request_timeout,
                max_bytes=args.max_video_bytes,
            )
            if not ok:
                rejected_download += 1
                continue
            downloaded += 1

            detection = inspect_video(
                video_path=video_path,
                detector=detector,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                sample_frames=args.sample_frames,
            )
            if not detection.accepted:
                rejected_visual += 1
                video_path.unlink(missing_ok=True)
                continue

            item = QualifiedVideo(
                id=video_id,
                download_url=f"/video-files/{video_id}",
                source_url=url,
                text=text,
                duration_seconds=round(detection.duration_seconds, 3),
                width=detection.width,
                height=detection.height,
                content_type=content_type or "video/mp4",
                size_bytes=video_path.stat().st_size,
                sampled_frames=detection.sampled_frames,
                face_hits=detection.face_hits,
                detection_reason=detection.reason,
            )
            index_file.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
            index_file.flush()
            accepted_count += 1

            if checked % args.report_every == 0 or accepted_count % 5 == 0:
                elapsed = time.monotonic() - started_at
                print(
                    "video_harvest",
                    f"cached={accepted_count}",
                    f"checked={checked}",
                    f"downloaded={downloaded}",
                    f"rejected_download={rejected_download}",
                    f"rejected_visual={rejected_visual}",
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    print(f"Done. cached={accepted_count} checked={checked} downloaded={downloaded}")


def _iter_source_rows(args: argparse.Namespace):
    if args.dataset:
        dataset = load_dataset(args.dataset, split=args.split, streaming=True, token=args.hf_token)
        if args.shuffle_buffer > 1:
            dataset = dataset.shuffle(buffer_size=args.shuffle_buffer, seed=42)
        for sample in dataset:
            yield {key: str(value) for key, value in sample.items() if value is not None}
        return

    if not args.source_path:
        raise ValueError("Provide --source-path for CSV/JSONL input or --dataset for a streaming Hugging Face dataset.")

    yield from iter_video_source_rows(args.source_path)


def _load_existing(index_path: Path, videos_dir: Path) -> list[dict[str, object]]:
    if not index_path.exists():
        return []
    rows: list[dict[str, object]] = []
    with index_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if (videos_dir / str(row["id"])).exists():
                rows.append(row)
    return rows


if __name__ == "__main__":
    main()
