import os
from pathlib import Path

import cv2
import fitz  # PyMuPDF
from PIL import Image


def replaceRedactions(image: str):
    """Clean image regions that look like redactions and replace them with 'XX'."""
    filename = os.path.splitext(image)[0]

    loaded_image = cv2.imread(image)
    if loaded_image is None:
        raise FileNotFoundError(f"Could not read image: {image}")

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

        cv2.putText(
            loaded_image,
            text,
            (text_x, text_y),
            font,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    output_image = Image.fromarray(loaded_image)
    output_image.save(filename + ".png", format="PNG")
    return output_image


def pdfToImage(filepath: str | Path, filename: str, workDirFolder: str | Path) -> list[str]:
    """Convert each PDF page to PNG using PyMuPDF, then clean redactions."""
    pdf_path = Path(filepath).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")

    clean_filename = pdf_path.stem
    output_path = Path(workDirFolder) / "images" / clean_filename
    output_path.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    image_paths = []

    try:
        for page_number, page in enumerate(doc, start=1):
            matrix = fitz.Matrix(300 / 72, 300 / 72)  # ~300 DPI
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY)

            image_path = output_path / f"{clean_filename}_{page_number}.png"
            pix.save(str(image_path))
            image_paths.append(str(image_path))
    finally:
        doc.close()

    for image_path in image_paths:
        replaceRedactions(image_path)

    return image_paths


def prepare_pdf_images_for_ocr(pdf_file: str | Path, work_dir_folder: str | Path) -> list[str]:
    """Wrapper function for PDF-to-image OCR preparation."""
    pdf_path = Path(pdf_file)
    return pdfToImage(str(pdf_path), pdf_path.name, str(work_dir_folder))

def pdf_ocr_preprocessor_main(pdf_folder, output_folder):
    output_folder = Path(output_folder) / "pdf_ocr_preprocessor"
    output_folder.mkdir(parents=True, exist_ok=True)

    try:
        pdf_files = sorted(path for path in pdf_folder.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in: {pdf_folder}")

        for pdf_file in pdf_files:
            print(f"Processing {pdf_file.name}...")
            image_paths = prepare_pdf_images_for_ocr(pdf_file, output_folder)
            print("Conversion complete. Images saved:")
            for path in image_paths:
                print(path)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    pdf_folder = Path(r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\PDF\AAO")
    output_folder = "output_folder"

    try:
        pdf_files = sorted(path for path in pdf_folder.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in: {pdf_folder}")

        for pdf_file in pdf_files:
            print(f"Processing {pdf_file.name}...")
            image_paths = prepare_pdf_images_for_ocr(pdf_file, output_folder)
            print("Conversion complete. Images saved:")
            for path in image_paths:
                print(path)
    except Exception as e:
        print(f"Error: {e}")