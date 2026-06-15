from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset
from PIL import Image, UnidentifiedImageError


POSITIVE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bperson\b",
        r"\bpeople\b",
        r"\bman\b",
        r"\bwoman\b",
        r"\bboy\b",
        r"\bgirl\b",
        r"\bmale\b",
        r"\bfemale\b",
        r"\bmodel\b",
        r"\bhuman\b",
        r"\bstanding\b",
        r"\bfull[\s-]?body\b",
        r"\bhead[\s-]?to[\s-]?toe\b",
        r"\bface\b",
        r"\bportrait\b",
    ]
]

NEGATIVE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\banime\b",
        r"\billustration\b",
        r"\bdrawing\b",
        r"\bsketch\b",
        r"\bpainting\b",
        r"\bcartoon\b",
        r"\bcomic\b",
        r"\brender\b",
        r"\b3d\b",
        r"\bstatue\b",
        r"\bsculpture\b",
        r"\bdoll\b",
        r"\btoy\b",
        r"\bposter\b",
        r"\bgroup\b",
        r"\bcrowd\b",
        r"\bteam\b",
        r"\bside view\b",
        r"\bback view\b",
        r"\bfrom behind\b",
        r"\bclose[\s-]?up\b",
        r"\bselfie\b",
    ]
]

FULL_BODY_HINTS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bfull[\s-]?body\b",
        r"\bhead[\s-]?to[\s-]?toe\b",
        r"\bstanding\b",
        r"\bstanding pose\b",
        r"\bwhole body\b",
    ]
]

FACE_HINTS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bface\b",
        r"\bfacial\b",
        r"\blooking at camera\b",
        r"\bportrait\b",
    ]
]

SINGLE_PERSON_HINTS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\ba person\b",
        r"\ba man\b",
        r"\ba woman\b",
        r"\ba boy\b",
        r"\ba girl\b",
        r"\bone person\b",
        r"\bsingle person\b",
        r"\blone\b",
        r"\bsolo\b",
    ]
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter downloadable person/full-body/face candidates from LAION.")
    parser.add_argument("--dataset", type=str, default="laion/laion2B-en")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--max-keep", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--min-width", type=int, default=256)
    parser.add_argument("--min-height", type=int, default=256)
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs/laion_people")
    return parser.parse_args()


def get_text(sample: dict[str, Any]) -> str:
    for key in ("TEXT", "caption", "text", "alt_text"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def get_url(sample: dict[str, Any]) -> str:
    for key in ("URL", "url"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def looks_like_candidate(text: str) -> tuple[bool, dict[str, bool]]:
    if not text:
        return False, {"person": False, "single_person": False, "full_body": False, "face": False}

    if any(pattern.search(text) for pattern in NEGATIVE_PATTERNS):
        return False, {"person": False, "single_person": False, "full_body": False, "face": False}

    person = any(pattern.search(text) for pattern in POSITIVE_PATTERNS)
    single_person = any(pattern.search(text) for pattern in SINGLE_PERSON_HINTS)
    full_body = any(pattern.search(text) for pattern in FULL_BODY_HINTS)
    face = any(pattern.search(text) for pattern in FACE_HINTS)

    keep = person and (single_person or full_body or face)
    return keep, {
        "person": person,
        "single_person": single_person,
        "full_body": full_body,
        "face": face,
    }


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; LAIONFilter/1.0; +https://example.invalid)",
            "Accept": "image/*,*/*;q=0.8",
        }
    )
    return session


def download_image(session: requests.Session, url: str, timeout: float) -> tuple[Image.Image | None, bytes | None, str]:
    try:
        response = session.get(url, timeout=timeout, stream=True, allow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type.lower():
            return None, None, f"non-image content-type: {content_type or 'unknown'}"

        content = response.content
        image = Image.open(BytesIO(content))
        image.load()
        return image, content, ""
    except (requests.RequestException, UnidentifiedImageError, OSError) as exc:
        return None, None, str(exc)


def image_too_small(image: Image.Image, min_width: int, min_height: int) -> bool:
    width, height = image.size
    return width < min_width or height < min_height


def save_image_bytes(output_dir: Path, image_bytes: bytes, url: str, content_type: str) -> str:
    extension = guess_extension(content_type, url)
    name = hashlib.sha1(url.encode("utf-8")).hexdigest()[:20]
    path = output_dir / f"{name}.{extension}"
    path.write_bytes(image_bytes)
    return path.name


def guess_extension(content_type: str, url: str) -> str:
    if "png" in content_type.lower():
        return "png"
    if "webp" in content_type.lower():
        return "webp"
    if "gif" in content_type.lower():
        return "gif"
    suffix = Path(url).suffix.lower().lstrip(".")
    if suffix in {"jpg", "jpeg", "png", "webp", "gif"}:
        return suffix
    return "jpg"


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    jsonl_path = output_dir / "filtered.jsonl"
    csv_path = output_dir / "filtered.csv"

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for row in rows:
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = [
        "url",
        "text",
        "width",
        "height",
        "content_type",
        "local_file",
        "person_hint",
        "single_person_hint",
        "full_body_hint",
        "face_hint",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    if args.download_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
    session = build_session()

    kept_rows: list[dict[str, Any]] = []
    checked = 0

    for sample in ds:
        checked += 1
        if checked > args.max_samples or len(kept_rows) >= args.max_keep:
            break

        text = get_text(sample)
        url = get_url(sample)
        if not url:
            continue

        keep_text, hints = looks_like_candidate(text)
        if not keep_text:
            continue

        image, image_bytes, error = download_image(session, url, timeout=args.timeout)
        if image is None or image_bytes is None:
            continue

        if image_too_small(image, args.min_width, args.min_height):
            continue

        content_type = Image.MIME.get(image.format, "image/unknown")
        local_file = ""
        if args.download_images:
            local_file = save_image_bytes(images_dir, image_bytes, url, content_type)

        kept_rows.append(
            {
                "url": url,
                "text": text,
                "width": image.width,
                "height": image.height,
                "content_type": content_type,
                "local_file": local_file,
                "person_hint": hints["person"],
                "single_person_hint": hints["single_person"],
                "full_body_hint": hints["full_body"],
                "face_hint": hints["face"],
            }
        )

        if len(kept_rows) % 10 == 0:
            print(f"kept={len(kept_rows)} checked={checked} last_url={url}")

    write_outputs(kept_rows, output_dir)
    print(f"Done. Checked {checked} samples, kept {len(kept_rows)}.")
    print(f"Results written to: {output_dir.resolve()}")
    print("Note: this tool guarantees downloadable image URLs, but person/full-body/face filtering is text-hint based.")


if __name__ == "__main__":
    main()
