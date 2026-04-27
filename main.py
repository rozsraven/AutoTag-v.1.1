import argparse
from pathlib import Path

from pdf_ocr_preprocessor import pdf_ocr_preprocessor_main
from image_output_processor import image_processor_main
from image_to_xml import image_to_xml_main

import warnings

from xml_cleanup import convert_unicode_punctuation_in_xml as cleanup_xml_file
warnings.filterwarnings("ignore", category=UserWarning, module="fitz")  # Suppress PyMuPDF warnings

def run_bia():
    filelist = ["5223008", "5223012", "5223014", "5223014", "5225801", "5236846"]
    updated_list = []
    for filenum in filelist:
        print("Starting PDF OCR preprocessing...")
        preprocessor_input_path = Path(r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\PDF\BIA PDF TEST SAMPLE")
        preprocessor_output_path = r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\bia_pdf_ocr_preprocessor"
        pdf_ocr_preprocessor_main(preprocessor_input_path, preprocessor_output_path)
        print("PDF OCR preprocessing completed.\n")

        print("#" * 60)

        print("Starting image processing...")
        image_processor_input_path = Path(rf"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\bia_pdf_ocr_preprocessor\images\{filenum}")
        image_processor_output_path = rf"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\processed_images\{filenum}"
        image_processor_main(input_folder=image_processor_input_path, output_folder=image_processor_output_path)
        print("Image processing completed.\n")    

        print("#" * 60)

        print("Starting image to XML conversion...")
        image_to_xml_input_path = Path(rf"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\processed_images\{filenum}")
        image_to_xml_input_path_html = Path(rf"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\PDF\BIA PDF TEST SAMPLE\{filenum}")
        image_to_xml_output_path = Path(rf"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\xml\{filenum}")
        image_to_xml_main(
            input_folder=image_to_xml_input_path,
            html_folder_path=image_to_xml_input_path_html,
            output_folder=image_to_xml_output_path,
            process_type_name="BIA",
        )
        print("Image to XML conversion completed.")

        print("#" * 60)

        print("Starting XML cleanup...")
        cleanup_xml_file(rf"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\xml\{filenum}\{filenum}.xml")
        print("XML cleanup completed.")
        updated_list.append(filenum)
        print(f"Updated List: {updated_list}\n")

    print(updated_list)

def run_aao():
    filelist = ["JAN132025_01D6101", "JAN132025_01E2309", "JAN132025_01H5212", "JAN132025_02B5203"]
    updated_list = []
    for filenum in filelist:
        print("Starting PDF OCR preprocessing...")
        preprocessor_input_path = Path(r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\PDF\AAO")
        preprocessor_output_path = r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\AAO pdf_ocr_preprocessor"
        pdf_ocr_preprocessor_main(preprocessor_input_path, preprocessor_output_path)
        print("PDF OCR preprocessing completed.\n")

        print("#" * 60)

        print("Starting image processing...")
        image_processor_input_path = Path(rf"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\AAO pdf_ocr_preprocessor\images\{filenum}")
        image_processor_output_path = rf"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\processed_images\{filenum}"
        image_processor_main(input_folder=image_processor_input_path, output_folder=image_processor_output_path)
        print("Image processing completed.\n")    

        print("#" * 60)

        print("Starting image to XML conversion...")
        image_to_xml_input_path = Path(rf"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\processed_images\{filenum}")
        image_to_xml_input_path_html = Path(rf"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\PDF\AAO")
        image_to_xml_output_path = Path(rf"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\xml\{filenum}")
        image_to_xml_main(
            input_folder=image_to_xml_input_path,
            html_folder_path=image_to_xml_input_path_html,
            output_folder=image_to_xml_output_path,
            process_type_name="AAO",
        )
        print("Image to XML conversion completed.")

        print("#" * 60)

        print("Starting XML cleanup...")
        cleanup_xml_file(rf"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\xml\{filenum}\{filenum}.xml")
        print("XML cleanup completed.")
        updated_list.append(filenum)
        print(f"Updated List: {updated_list}\n")

    print(updated_list)
if __name__ == "__main__":
    run_bia()
    run_aao()
        