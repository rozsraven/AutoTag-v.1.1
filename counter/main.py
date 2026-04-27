from pdf_ocr_preprocessor import prepare_pdf_images_for_ocr
from image_to_text_ocr import image_to_text_ocr_main
from txt_counter import txt_counter_main
import os
from pathlib import Path
import multiprocessing


def main() -> None:
    print("Starting PDF OCR preprocessing...")

    pdf_input_folder = Path(r"C:\Users\tior\Downloads\OneDrive_1_4-24-2026")
    pdf_output_folder = pdf_input_folder / "output"

    try:
        pdf_files = sorted(
            path for path in pdf_input_folder.rglob("*.pdf")
            if path.is_file()
        )
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in: {pdf_input_folder}")

        for pdf_file in pdf_files:
            print(f"Processing {pdf_file.relative_to(pdf_input_folder)}...")
            image_paths = prepare_pdf_images_for_ocr(pdf_file, pdf_output_folder)
            print("Conversion complete. Images saved:")
            for path in image_paths:
                print(path)
    except Exception as e:
        print(f"Error: {e}")

    print("\n==============================")

    print("Starting image processing...")

    image_input_folder = pdf_output_folder / "images"
    image_output_folder = pdf_output_folder / "texts"

    image_to_text_ocr_main(image_input_folder, image_output_folder)

    print("\n==============================")

    print("Starting text counting...")

    txt_input_folder = pdf_output_folder / "texts"
    txt_output_file = pdf_output_folder / "file_count.txt"

    txt_counter_main(txt_input_folder, txt_output_file)

    print("Counting completed.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()