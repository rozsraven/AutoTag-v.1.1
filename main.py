import argparse
from pathlib import Path

from pdf_ocr_preprocessor2 import pdf_ocr_preprocessor_main
from image_output_processor2 import image_processor_main
from image_to_xml_new import image_to_xml_main

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="fitz")  # Suppress PyMuPDF warnings

preprocessor_output_path = r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\pdf_ocr_preprocessor"
image_processor_input_path = r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\pdf_ocr_preprocessor\JAN132025_01D6101"

if __name__ == "__main__":
    print("Starting PDF OCR preprocessing...")
    preprocessor_input_path = Path(r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\PDF\AAO")
    preprocessor_output_path = r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\pdf_ocr_preprocessor"
    pdf_ocr_preprocessor_main(preprocessor_input_path, preprocessor_output_path)
    print("PDF OCR preprocessing completed.\n")

    print("#" * 60)

    print("Starting image processing...")
    image_processor_input_path = Path(r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\pdf_ocr_preprocessor\pdf_ocr_preprocessor\images\JAN132025_01D6101")
    image_processor_output_path = r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\processed_images\JAN132025_01D6101"
    image_processor_main(input_folder=image_processor_input_path, output_folder=image_processor_output_path)
    print("Image processing completed.\n")    

    print("#" * 60)

    print("Starting image to XML conversion...")
    image_to_xml_input_path = Path(r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\processed_images\JAN132025_01D6101")
    image_to_xml_input_path_html = Path(r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\PDF\AAO")
    image_to_xml_output_path = Path(r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\xml")
    image_to_xml_main(
        input_folder=image_to_xml_input_path,
        html_folder_path=image_to_xml_input_path_html,
        output_folder=image_to_xml_output_path,
        process_type_name="AAO",
    )
    print("Image to XML conversion completed.")