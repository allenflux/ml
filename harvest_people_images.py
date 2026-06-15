from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from laion_people_api import QualifiedImagePool, ServiceConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest qualified LAION people images into a local cache.")
    parser.add_argument("--dataset", type=str, default="laion/laion2B-en")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=str, default="outputs/laion_people_api")
    parser.add_argument("--target-cache", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=4.0)
    parser.add_argument("--min-width", type=int, default=256)
    parser.add_argument("--min-height", type=int, default=256)
    parser.add_argument("--min-person-height-ratio", type=float, default=0.25)
    parser.add_argument("--shuffle-buffer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-body-detection", action="store_true")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--report-every", type=float, default=10.0)
    parser.add_argument("--max-runtime-seconds", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ServiceConfig(
        mode="hybrid",
        dataset=args.dataset,
        split=args.split,
        output_dir=Path(args.output_dir),
        max_cache=args.target_cache,
        workers=args.workers,
        request_timeout=args.request_timeout,
        min_width=args.min_width,
        min_height=args.min_height,
        min_person_height_ratio=args.min_person_height_ratio,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        allow_text_body_fallback=not args.strict_body_detection,
        hf_token=args.hf_token,
    )
    pool = QualifiedImagePool(config)
    pool.start()
    started_at = time.monotonic()

    try:
        while True:
            status = pool.status()
            print(
                "harvest",
                f"cached={status['cached']}",
                f"checked={status['checked']}",
                f"text_candidates={status['text_candidates']}",
                f"downloaded={status['downloaded']}",
                f"accepted={status['accepted']}",
                f"rejected_download={status['rejected_download']}",
                f"rejected_visual={status['rejected_visual']}",
                f"last_stage={status['last_stage']}",
                f"last_error={status['last_error']}",
                flush=True,
            )
            if status["cached"] >= args.target_cache:
                print(f"Target reached: {status['cached']} cached images.")
                break
            if args.max_runtime_seconds > 0 and time.monotonic() - started_at >= args.max_runtime_seconds:
                print("Max runtime reached.")
                break
            time.sleep(args.report_every)
    finally:
        pool.stop()


if __name__ == "__main__":
    main()
