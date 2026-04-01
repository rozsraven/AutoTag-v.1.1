import os
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
from pdf2image import convert_from_path
from PIL import Image

def find_poppler_bin() -> Optional[str]:
    """Return the Poppler bin directory used by pdf2image on Windows."""
    repo_root = Path(__file__).resolve().parent

    override = Path.cwd()
    env_override = None
    try:
        import os

        env_override = os.environ.get("POPPLER_BIN")
    except Exception:
        env_override = None

    if env_override:
        candidate = Path(env_override)
        if (candidate / "pdfinfo.exe").exists():
            return str(candidate)

    bundled_candidates = [
        repo_root / "poppler-21.11.0" / "Library" / "bin",
        repo_root / "poppler-21.11.0" / "bin",
    ]
    for candidate in bundled_candidates:
        if (candidate / "pdfinfo.exe").exists():
            return str(candidate)

    for candidate in repo_root.glob("poppler-*/**/bin"):
        if (candidate / "pdfinfo.exe").exists():
            return str(candidate)

    if (override / "pdfinfo.exe").exists():
        return str(override)

    return None


def replaceRedactions(image: str):
    """Mirror ExtractPDFText.replaceRedactions for OCR image cleanup."""
    filename = os.path.splitext(image)[0]
    work_dir_folder = os.path.dirname(image)

    loaded_image = cv2.imread(image)
    gray = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        9,
    )

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for contour in contours:
        cv2.drawContours(thresh, [contour], -1, (255, 255, 255), -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

    contours = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 80:
            continue

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "XX"
        textsize = cv2.getTextSize(text, font, 0.5, 1)[0]

        cv2.rectangle(loaded_image, (x, y), (x + w, y + h), (255, 255, 255), -1)

        text_x = x + (w // 3 - textsize[0] // 3)
        text_y = y + h - textsize[1]

        cv2.putText(loaded_image, text, (text_x, text_y), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    output_image = Image.fromarray(loaded_image)
    output_image.save(os.path.join(work_dir_folder, filename + ".png"), format="png")
    return output_image


def pdfToImage(filepath: str | Path, filename: str, workDirFolder: str | Path) -> list[str]:
    """Mirror ExtractPDFText.pdfToImage, including saved PNG output and redaction replacement."""
    pdf_path = Path(filepath).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")

    poppler_bin = find_poppler_bin()
    if not poppler_bin:
        raise RuntimeError(
            "Poppler was not found. Set POPPLER_BIN or use the bundled poppler folder in this repository."
        )

    clean_filename = filename.replace(".pdf", "")
    output_path = Path(workDirFolder) / "images" / clean_filename
    output_path.mkdir(parents=True, exist_ok=True)

    images_from_path = convert_from_path(
        str(pdf_path),
        300,
        poppler_path=poppler_bin,
        output_folder=str(output_path),
        output_file=clean_filename,
        fmt="png",
        grayscale=True,
    )

    for image in images_from_path:
        image.close()

    image_paths = glob(str(output_path / "*.png"))
    for image_path in image_paths:
        replaceRedactions(image_path)

    return image_paths


def prepare_pdf_images_for_ocr(pdf_file: str | Path, work_dir_folder: str | Path) -> list[str]:
    """Compatibility wrapper that uses the legacy PDF-to-image behavior."""
    pdf_path = Path(pdf_file)
    return pdfToImage(str(pdf_path), pdf_path.name, str(work_dir_folder))

if __name__ == "__main__":
    # Hardcoded input and output paths
    pdf_file = "JAN132025_01D6101.pdf"
    output_folder = "output_folder"

    try:
        image_paths = prepare_pdf_images_for_ocr(pdf_file, output_folder)
        print("Conversion complete. Images saved:")
        for path in image_paths:
            print(path)
    except Exception as e:
        print(f"Error: {e}")