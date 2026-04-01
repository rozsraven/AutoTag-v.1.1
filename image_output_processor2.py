import argparse
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import tika
from tika import parser

from opencv_process import split_body_and_footnotes


tika.TikaClientOnly = True

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass
class ProcessedPdfImages:
    body_images: list[str]
    footnote_images: list[str]
    body_text: str
    footnote_text: str


def collect_image_files(
    input_folder: str | Path,
    *,
    recursive: bool = False,
    image_extensions: Iterable[str] | None = None,
) -> list[Path]:
    folder = Path(input_folder)

    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {folder}")

    extensions = {extension.lower() for extension in (image_extensions or IMAGE_EXTENSIONS)}
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    image_files = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in extensions)

    if not image_files:
        raise ValueError(f"No supported image files found in: {folder}")

    return image_files


def extract_text_from_images(image_files: Sequence[str | Path]) -> str:
    extracted_text = ""

    for image_file in image_files:
        raw = parser.from_file(str(image_file))
        content = raw.get("content") if raw else None
        if content:
            extracted_text += content

    return html.escape(extracted_text, quote=False)


def process_pdf_ocr_images(
    image_files: Sequence[str | Path] | None = None,
    *,
    output_folder: str | Path | None = None,
) -> ProcessedPdfImages:
    if not image_files:
        raise ValueError("Provide image_files.")

    body_images, footnote_images = split_body_and_footnotes(image_files, output_folder=output_folder)
    body_text = extract_text_from_images(body_images)
    footnote_text = extract_text_from_images(footnote_images)

    return ProcessedPdfImages(
        body_images=body_images,
        footnote_images=footnote_images,
        body_text=body_text,
        footnote_text=footnote_text,
    )


def process_pdf_ocr_image_folder(
    input_folder: str | Path,
    *,
    output_folder: str | Path | None = None,
    recursive: bool = False,
    image_extensions: Iterable[str] | None = None,
) -> ProcessedPdfImages:
    image_files = collect_image_files(
        input_folder,
        recursive=recursive,
        image_extensions=image_extensions,
    )
    return process_pdf_ocr_images(image_files=image_files, output_folder=output_folder)


def image_processor_main(
    input_folder: str | Path | None = None,
    *,
    output_folder: str | Path | None = None,
    recursive: bool = False,
    image_extensions: Iterable[str] | None = None,
) -> ProcessedPdfImages:
    if input_folder is None:
        arg_parser = argparse.ArgumentParser(description="Process OCR image output from a folder.")
        arg_parser.add_argument("input_folder", type=Path, help="Folder containing OCR image files")
        arg_parser.add_argument(
            "--output-folder",
            type=Path,
            help="Folder where body and footnote images will be written",
        )
        arg_parser.add_argument(
            "--recursive",
            action="store_true",
            help="Search for image files recursively under the input folder",
        )
        args = arg_parser.parse_args()
        input_folder = args.input_folder
        output_folder = args.output_folder
        recursive = args.recursive

    try:
        processed = process_pdf_ocr_image_folder(
            input_folder,
            output_folder=output_folder,
            recursive=recursive,
            image_extensions=image_extensions,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print("Body images:")
    for image_path in processed.body_images:
        print(image_path)

    print("\nFootnote images:")
    for image_path in processed.footnote_images:
        print(image_path)

    print("\nBody text:")
    print(processed.body_text)

    print("\nFootnote text:")
    print(processed.footnote_text)

    return processed


if __name__ == "__main__":
    image_processor_input_path = r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\pdf_ocr_preprocessor\JAN132025_01D6101"
    image_processor_output_path = r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\processed_images\JAN132025_01D6101"
    image_processor_main(input_folder=image_processor_input_path, output_folder=image_processor_output_path)