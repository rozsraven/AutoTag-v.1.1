from ptt_updated import pdf_to_text_ocr_main
from txt_counter_updated import txt_counter_main
from pathlib import Path
import multiprocessing
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OCR and text counting for a folder containing PDFs."
    )
    parser.add_argument(
        "pdf_input_folder",
        nargs="?",
        help="Path to the folder that contains PDF files.",
    )
    return parser.parse_args()


def main() -> None:
    print("Starting PDF to TXT OCR...")

    args = parse_args()
    input_path = (args.pdf_input_folder or "").strip()

    if not input_path:
        input_path = input("Enter PDF input folder path: ").strip()

    if not input_path:
        print("No path provided. Exiting.")
        return

    pdf_input_folder = Path(input_path)
    if not pdf_input_folder.exists() or not pdf_input_folder.is_dir():
        print(f"Invalid folder path: {pdf_input_folder}")
        return

    pdf_output_folder = pdf_input_folder / "output"
    text_output_folder = pdf_output_folder / "texts"

    try:
        pdf_to_text_ocr_main(pdf_input_folder, text_output_folder)
    except Exception as e:
        print(f"Error: {e}")

    print("\n==============================")
    print("Starting text counting...")

    txt_output_file = pdf_output_folder / "file_count.txt"

    try:
        txt_counter_main(text_output_folder, txt_output_file)
        print("Counting completed.")
    except Exception as e:
        print(f"Counting error: {e}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()