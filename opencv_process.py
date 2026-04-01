from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _load_grayscale_image(image_file: str | Path):
    image_path = Path(image_file)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image_path, image


def _find_split_row(gray_image) -> int | None:
    height, width = gray_image.shape[:2]
    if height < 300 or width < 300:
        return None

    _, binary_inv = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    horizontal_projection = (binary_inv > 0).sum(axis=1)
    row_activity_threshold = max(12, int(width * 0.01))
    active_rows = horizontal_projection >= row_activity_threshold

    page_start = int(height * 0.45)
    min_gap_height = max(18, int(height * 0.012))
    min_bottom_text_height = max(60, int(height * 0.08))
    min_body_height = int(height * 0.55)

    gap_start = None
    candidate_split = None

    for row in range(page_start, height):
        if not active_rows[row]:
            if gap_start is None:
                gap_start = row
            continue

        if gap_start is not None:
            gap_height = row - gap_start
            bottom_text_height = height - row
            if gap_height >= min_gap_height and bottom_text_height >= min_bottom_text_height and gap_start >= min_body_height:
                candidate_split = gap_start + gap_height // 2
            gap_start = None

    if candidate_split is None:
        return None

    return candidate_split


def _write_output_image(
    source_path: Path,
    image,
    suffix: str | None = None,
    output_folder: str | Path | None = None,
) -> str:
    if output_folder is None:
        if suffix is None:
            output_path = source_path
        else:
            output_path = source_path.with_name(f"{source_path.stem}_{suffix}{source_path.suffix}")
    else:
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = source_path.name if suffix is None else f"{source_path.stem}_{suffix}{source_path.suffix}"
        output_path = output_dir / filename

    cv2.imwrite(str(output_path), image)
    return str(output_path)


def split_body_and_footnotes(
    image_files: Sequence[str | Path],
    output_folder: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    body_images: list[str] = []
    footnote_images: list[str] = []

    for image_file in image_files:
        image_path, gray_image = _load_grayscale_image(image_file)
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        split_row = _find_split_row(gray_image)
        if split_row is None:
            body_images.append(_write_output_image(image_path, gray_image, output_folder=output_folder))
            continue

        body_crop = gray_image[:split_row, :]
        footnote_crop = gray_image[split_row:, :]

        if body_crop.size == 0 or footnote_crop.size == 0:
            body_images.append(_write_output_image(image_path, gray_image, output_folder=output_folder))
            continue

        body_images.append(_write_output_image(image_path, body_crop, "body", output_folder))
        footnote_images.append(_write_output_image(image_path, footnote_crop, "footnotes", output_folder))

    return body_images, footnote_images
