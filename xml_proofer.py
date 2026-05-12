import os
import lxml.etree as ET
from pathlib import Path
from xhtml2pdf import pisa

XSL = r"converters\xml2html.xsl"

def convert_html_to_pdf(source_html, output_filename):
    result_file = open(output_filename, "w+b")
    pisa_status = pisa.CreatePDF(
            source_html,              
            dest=result_file)          

    result_file.close()                
    return pisa_status.err

def convert_xml_to_pdf(filecount, xml_filename, workDirFolder):
    try:
        dom = ET.parse(xml_filename)
    except:
        with open(xml_filename, "rb") as f:
            data = f.read()
            data = data.replace(b"ad:decision-grp volnum", b'ad:decision-grp xmlns:ad="http://www.lexisnexis.com/namespace/sslrp/ad" \
                    xmlns:tr="http://www.lexisnexis.com/namespace/sslrp/tr" \
                    xmlns:ps="http://www.lexisnexis.com/namespace/sslrp/ps" xmlns:form="http://www.lexisnexis.com/namespace/sslrp/form" \
                    xmlns:fn="http://www.lexisnexis.com/namespace/sslrp/fn" xmlns:se="http://www.lexisnexis.com/namespace/sslrp/se" \
                    xmlns:di="http://www.lexisnexis.com/namespace/sslrp/di" xmlns:core="http://www.lexisnexis.com/namespace/sslrp/core" \
                    xmlns:em="http://www.lexisnexis.com/namespace/sslrp/em" xmlns:fm="http://www.lexisnexis.com/namespace/sslrp/fm" \
                    xmlns:pnfo="http://www.lexisnexis.com/namespace/sslrp/pnfo" xmlns:lnci="http://www.lexisnexis.com/namespace/common/lnci" \
                    xmlns:glph="http://www.lexisnexis.com/namespace/sslrp/glph" xmlns:pu="http://www.lexisnexis.com/namespace/sslrp/pu" \
                    xmlns:su="http://www.lexisnexis.com/namespace/sslrp/su" xmlns:nl="http://www.lexisnexis.com/namespace/sslrp/nl" \
                    xmlns:ls="http://www.lexisnexis.com/namespace/sslrp/ls" xmlns:nc="http://www.lexisnexis.com/namespace/sslrp/nc" volnum')
        dom = ET.fromstring(data)
    xslt = ET.parse(XSL)
    transform = ET.XSLT(xslt)
    newdom = transform(dom)
    output_filename = os.path.basename(xml_filename)[:-4] + ".pdf"
    output_file = os.path.join(workDirFolder, output_filename)
    res = convert_html_to_pdf(newdom, output_file)
    res = "Success" if res == 0 else "Failed"
    return filecount, res


def xml_proofer_main(input_folder):

    input_folder = Path(input_folder)

    # Auto-create output folder inside input folder
    output_folder = input_folder / "PDF Output"

    print(f"Output folder: {output_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)

    xml_files = list(input_folder.glob("*.xml"))

    xml_files_count = len(xml_files)

    for count, xml_file in enumerate(xml_files, start=1):

        print(f"{count}/{xml_files_count}")

        result = convert_xml_to_pdf(
            count,
            xml_file,
            output_folder
        )

        print(f"Processing: {xml_file.name} -> {result[1]}")

    return xml_files_count


if __name__ == "__main__":

    input_folder = Path(
        r"C:\Users\tior\Downloads\Conversion_test\output\xml - Copy"
    )

    xml_files_count = xml_proofer_main(input_folder)

    print(f"Found {xml_files_count} XML files.")