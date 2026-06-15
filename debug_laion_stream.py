from __future__ import annotations

import argparse
import os
import time

from datasets import load_dataset

from filter_laion_people import get_text, get_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug whether a LAION streaming dataset yields samples.")
    parser.add_argument("--dataset", type=str, default="laion/laion2B-en")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--shuffle-buffer", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.monotonic()
    print(f"loading dataset={args.dataset} split={args.split}")
    dataset = load_dataset(args.dataset, split=args.split, streaming=True, token=args.hf_token)
    if args.shuffle_buffer > 1:
        print(f"applying shuffle buffer={args.shuffle_buffer}")
        dataset = dataset.shuffle(buffer_size=args.shuffle_buffer, seed=42)

    print("iterating stream...")
    for index, sample in enumerate(dataset, start=1):
        elapsed = time.monotonic() - started_at
        url = get_url(sample)
        text = get_text(sample)
        print(f"sample={index} elapsed={elapsed:.2f}s url={url[:120]} text={text[:160]}")
        if index >= args.limit:
            break
    print("done")


if __name__ == "__main__":
    main()
