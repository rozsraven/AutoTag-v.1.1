import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from opencv_process import split_body_and_footnotes
from run_paddle_ocr import create_paddle_ocr, extract_text_from_images


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


def process_pdf_ocr_images(
    image_files: Sequence[str | Path] | None = None,
    *,
    output_folder: str | Path | None = None,
) -> ProcessedPdfImages:
    if not image_files:
        raise ValueError("Provide image_files.")

    body_images, footnote_images = split_body_and_footnotes(image_files, output_folder=output_folder)
    ocr = create_paddle_ocr()
    body_text = extract_text_from_images(body_images, ocr, escape_html=True)
    footnote_text = extract_text_from_images(footnote_images, ocr, escape_html=True)

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
