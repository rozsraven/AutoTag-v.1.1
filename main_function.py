from pathlib import Path

from pdf_ocr_preprocessor import pdf_ocr_preprocessor_main
from image_output_processor import image_processor_main
from image_to_xml import image_to_xml_main

import warnings

from xml_cleanup import convert_unicode_punctuation_in_xml as cleanup_xml_file
warnings.filterwarnings("ignore", category=UserWarning, module="fitz")  # Suppress PyMuPDF warnings


def _get_pdf_filelist(input_folder: str | Path) -> list[Path]:
    pdf_input_folder = Path(input_folder)
    return sorted(
        path.relative_to(pdf_input_folder).with_suffix("")
        for path in pdf_input_folder.rglob("*.pdf")
        if path.is_file()
    )


def run_bia(input_folder: str):
    filelist = _get_pdf_filelist(input_folder)
    updated_list: list[str] = []
    pdf_input_folder = Path(rf"{input_folder}")
    pdf_output_folder = pdf_input_folder / "output"
    preprocessor_output_path = pdf_output_folder / "bia_pdf_ocr_preprocessor"
    processed_images_root = pdf_output_folder / "processed_images"
    xml_root = pdf_output_folder / "xml"

    print("Starting PDF OCR preprocessing...")
    pdf_ocr_preprocessor_main(pdf_input_folder, preprocessor_output_path)
    print("PDF OCR preprocessing completed.\n")

    for relative_pdf_path in filelist:
        filenum = relative_pdf_path.stem

        print("#" * 60)

        print("Starting image processing...")
        image_processor_input_path = preprocessor_output_path / "images" / relative_pdf_path
        image_processor_output_path = processed_images_root / relative_pdf_path
        image_processor_main(input_folder=image_processor_input_path, output_folder=image_processor_output_path)
        print("Image processing completed.\n")    

        print("#" * 60)

        print("Starting image to XML conversion...")
        source_pdf_path = pdf_input_folder / relative_pdf_path.with_suffix(".pdf")
        image_to_xml_input_path = processed_images_root / relative_pdf_path
        image_to_xml_input_path_html = source_pdf_path.parent
        image_to_xml_output_path = xml_root / relative_pdf_path.parent
        image_to_xml_main(
            input_folder=image_to_xml_input_path,
            html_folder_path=image_to_xml_input_path_html,
            output_folder=image_to_xml_output_path,
            process_type_name="BIA",
        )
        print("Image to XML conversion completed.")

        print("#" * 60)

        print("Starting XML cleanup...")
        cleanup_xml_file(image_to_xml_output_path / f"{filenum}.xml")
        print("XML cleanup completed.")
        updated_list.append(filenum)
        print(f"Updated List: {updated_list}\n")

    print(updated_list)


def run_aao(input_folder: str):
    filelist = _get_pdf_filelist(input_folder)
    updated_list: list[str] = []
    pdf_input_folder = Path(rf"{input_folder}")
    pdf_output_folder = pdf_input_folder / "output"
    preprocessor_output_path = pdf_output_folder / "AAO pdf_ocr_preprocessor"
    processed_images_root = pdf_output_folder / "processed_images"
    xml_root = pdf_output_folder / "xml"

    print("Starting PDF OCR preprocessing...")
    pdf_ocr_preprocessor_main(pdf_input_folder, preprocessor_output_path)
    print("PDF OCR preprocessing completed.\n")

    for relative_pdf_path in filelist:
        filenum = relative_pdf_path.stem

        print("#" * 60)

        print("Starting image processing...")
        image_processor_input_path = preprocessor_output_path / "images" / relative_pdf_path
        image_processor_output_path = processed_images_root / relative_pdf_path
        image_processor_main(input_folder=image_processor_input_path, output_folder=image_processor_output_path)
        print("Image processing completed.\n")    

        print("#" * 60)

        print("Starting image to XML conversion...")
        source_pdf_path = pdf_input_folder / relative_pdf_path.with_suffix(".pdf")
        image_to_xml_input_path = processed_images_root / relative_pdf_path
        image_to_xml_input_path_html = source_pdf_path.parent
        image_to_xml_output_path = xml_root / relative_pdf_path.parent
        image_to_xml_main(
            input_folder=image_to_xml_input_path,
            html_folder_path=image_to_xml_input_path_html,
            output_folder=image_to_xml_output_path,
            process_type_name="AAO",
        )
        print("Image to XML conversion completed.")

        print("#" * 60)

        print("Starting XML cleanup...")
        cleanup_xml_file(image_to_xml_output_path / f"{filenum}.xml")
        print("XML cleanup completed.")
        updated_list.append(filenum)
        print(f"Updated List: {updated_list}\n")

    print(updated_list)


if __name__ == "__main__":
    run_bia(r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\PDF\BIA PDF TEST SAMPLE")
    run_aao(r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\PDF\AAO")
        