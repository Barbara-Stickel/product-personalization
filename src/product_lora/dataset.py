from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class DatasetReport:
    image_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_dataset(
    path: Path, product_identifier: str, min_dimension: int = 512
) -> DatasetReport:
    report = DatasetReport()
    if not path.is_dir():
        report.errors.append(f"Dataset directory does not exist: {path}")
        return report

    digests: dict[str, Path] = {}
    for image_path in sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        report.image_count += 1
        caption_path = image_path.with_suffix(".txt")
        if not caption_path.is_file():
            report.errors.append(f"Missing caption: {caption_path}")
        else:
            caption = caption_path.read_text(encoding="utf-8").strip()
            if not caption:
                report.errors.append(f"Empty caption: {caption_path}")
            elif product_identifier not in caption:
                report.errors.append(
                    f"Caption omits identifier '{product_identifier}': {caption_path}"
                )
        try:
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if digest in digests:
                report.errors.append(f"Duplicate images: {digests[digest]} and {image_path}")
            else:
                digests[digest] = image_path
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                if min(image.size) < min_dimension:
                    report.warnings.append(
                        f"Small image {image_path}: {image.width}x{image.height} (< {min_dimension})"
                    )
        except (OSError, UnidentifiedImageError) as exc:
            report.errors.append(f"Invalid image {image_path}: {exc}")
    if report.image_count == 0:
        report.errors.append(f"No supported images in {path}")
    return report
