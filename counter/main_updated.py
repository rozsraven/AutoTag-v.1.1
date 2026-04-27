from ptt_updated import pdf_to_text_ocr_main
from txt_counter_updated import txt_counter_main
from pathlib import Path
import multiprocessing


def main() -> None:
    print("Starting PDF to TXT OCR...")

    pdf_input_folder = Path(r"C:\Users\tior\Downloads\383 Releases")
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