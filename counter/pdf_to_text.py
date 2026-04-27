from ptt import pdf_to_text_ocr_main
from txt_counter import txt_counter_main
from pathlib import Path
import multiprocessing


def main() -> None:
    print("Starting PDF OCR to text...")

    pdf_input_folder = Path(r"C:\Users\tior\Downloads\OneDrive_1_4-24-2026")
    pdf_output_folder = pdf_input_folder / "output_ptt"
    text_output_folder = pdf_output_folder / "texts"

    try:
        pdf_to_text_ocr_main(pdf_input_folder, text_output_folder)
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n==============================")

    print("Starting text counting...")

    txt_input_folder = text_output_folder
    txt_output_file = pdf_output_folder / "file_count.txt"

    txt_counter_main(txt_input_folder, txt_output_file)

    print("Counting completed.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()