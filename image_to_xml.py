import html
import io
import os
import re
import xml.etree.ElementTree as xml
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import bs4
import cv2
from bs4 import BeautifulSoup
from paddleocr import PaddleOCR

from opencv_process import split_body_and_footnotes


os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_INPUT_FOLDER = Path(r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\images\JAN132025_01D6101")
DEFAULT_HTML_FILE = Path(r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\PDF\AAO\JAN132025_01D6101.htm")
DEFAULT_OUTPUT_FOLDER = Path(r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\xml")
REDACTION_MARKERS = ("xx", "XX", "Xx", "xX")
REDACTION_REPLACEMENT = '<core:emph typestyle="bf">[redacted]</core:emph>'
HEADER_PLATE = "<?xml version='1.0' encoding='UTF-8'?><!DOCTYPE ad:decision-grp PUBLIC \"-//LEXISNEXIS//DTD Admin-Interp-pub v005//EN//XML\" \"D:\\Pub383\\DTD\\DTD\\admin-interp-pubV005-0000\\admin-interp-pubV005-0000.dtd\"><?Pub EntList alpha bull copy rArr sect trade para mdash ldquo lsquo rdquo rsquo dagger hellip ndash frac23 amp?>"
FNSTYLE_PATTERN = r"[^.*]*font-size: 7.*?}"
FURTHER_ORDER_PATTERN = re.compile(r"(F+U+R+T+H+E+R+\s+O+R+D+E+R+:)")
ORDER_PATTERN = re.compile(r"(O+R+D+E+R+:)(?!<)")
_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass
class ImageXmlPage:
    source_image: str
    body_image: str
    footnote_image: str | None
    body_text: str
    footnote_text: str


def collect_image_files(
    input_folder: str | Path,
    *,
    recursive: bool = False,
    image_extensions: Iterable[str] | None = None,
) -> list[Path]:
    folder = Path(input_folder)

    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {folder}")

    extensions = {extension.lower() for extension in (image_extensions or IMAGE_EXTENSIONS)}
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    all_image_files = sorted(
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in extensions
    )

    original_page_images = [
        path
        for path in all_image_files
        if not path.stem.endswith(("_body", "_footnotes"))
    ]
    original_page_names = {path.name for path in original_page_images}

    processed_body_images = []
    for path in all_image_files:
        if not path.stem.endswith("_body"):
            continue

        original_name = f"{path.stem[:-5]}{path.suffix}"
        if original_name in original_page_names:
            continue

        processed_body_images.append(path)

    image_files = sorted(original_page_images + processed_body_images)

    if not image_files:
        raise ValueError(f"No supported image files found in: {folder}")

    return image_files


def preprocess_image(image_bgr):
    height, width = image_bgr.shape[:2]
    scale = 1.5
    resized = cv2.resize(
        image_bgr,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 7, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    return cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )


def _box_metrics(box):
    points = box[0]
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    left = min(x_values)
    right = max(x_values)
    top = min(y_values)
    bottom = max(y_values)
    height = max(1.0, bottom - top)
    y_center = (top + bottom) / 2.0
    return left, right, top, bottom, height, y_center


def order_paddleocr_boxes_reading_order(raw_boxes):
    items = []
    for box in raw_boxes:
        text = (box[1][0] or "").strip()
        if not text:
            continue
        left, right, top, bottom, height, y_center = _box_metrics(box)
        items.append((y_center, top, bottom, left, height, box))

    if not items:
        return []

    items.sort(key=lambda item: (item[0], item[3]))
    heights = sorted(item[4] for item in items)
    median_height = heights[len(heights) // 2] if heights else 12.0
    line_y_threshold = max(8.0, 0.55 * median_height)

    lines = []
    current_line = []
    current_y = None

    for y_center, top, bottom, left, _, box in items:
        if current_y is None:
            current_line = [(y_center, top, bottom, left, box)]
            current_y = y_center
            continue

        if abs(y_center - current_y) <= line_y_threshold:
            current_line.append((y_center, top, bottom, left, box))
            current_y = (current_y * 0.8) + (y_center * 0.2)
            continue

        current_line.sort(key=lambda item: item[3])
        lines.append(current_line)
        current_line = [(y_center, top, bottom, left, box)]
        current_y = y_center

    if current_line:
        current_line.sort(key=lambda item: item[3])
        lines.append(current_line)

    return lines


def normalize_text(text: str) -> str:
    normalized = text.replace("\u00A0", " ")
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


def extract_text_from_image(image_path: str | Path, ocr: PaddleOCR) -> str:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    prepared = preprocess_image(image)
    result = ocr.ocr(prepared, cls=False)
    if not result or not result[0]:
        return ""

    lines = order_paddleocr_boxes_reading_order(result[0])
    text_lines: list[str] = []
    for line in lines:
        parts = []
        for _, _, _, _, box in line:
            candidate = (box[1][0] or "").strip()
            if candidate:
                parts.append(candidate)
        if parts:
            text_lines.append(normalize_text(" ".join(parts)))

    return "\n".join(line for line in text_lines if line)


def build_page_record(image_path: Path, ocr: PaddleOCR) -> ImageXmlPage:
    if image_path.stem.endswith("_body"):
        body_image = str(image_path)
        footnote_candidate = image_path.with_name(f"{image_path.stem[:-5]}_footnotes{image_path.suffix}")
        footnote_image = str(footnote_candidate) if footnote_candidate.exists() else None
    else:
        body_images, footnote_images = split_body_and_footnotes([image_path])
        body_image = body_images[0] if body_images else str(image_path)
        footnote_image = footnote_images[0] if footnote_images else None

    body_text = extract_text_from_image(body_image, ocr)
    footnote_text = extract_text_from_image(footnote_image, ocr) if footnote_image else ""

    return ImageXmlPage(
        source_image=str(image_path),
        body_image=body_image,
        footnote_image=footnote_image,
        body_text=body_text,
        footnote_text=footnote_text,
    )


def replace_redaction_markers(text: str) -> str:
    for marker in REDACTION_MARKERS:
        text = text.replace(marker, REDACTION_REPLACEMENT)
    return text


def process_ocr_string(ocr_text: str, filename: str):
    count = 0
    start_parse = False
    core_paras: list[str] = []
    new_page = False
    date_not_found = True
    type_of_case: list[str] = []
    plain_content_start = False
    case_date = "caseDateNotFound"
    file_num = ""
    case_name = "caseNameNotFound"

    type_of_case.append("AAO Designation: " + filename[filename.index("_") + 3 : len(filename) - 7])

    for line in ocr_text.split("\n"):
        if line in ("", "/N"):
            count += 1
            continue
        if line.lower().startswith(("matter of", "in re:", "inre:")) and not start_parse:
            case_name = line.replace("inre:", "In re:")
            if "date:" in case_name.lower():
                case_date = case_name[case_name.lower().index("date:") :]
                case_name = case_name[: case_name.lower().index("date:")]
                date_not_found = False
            count = 0
            start_parse = True
            continue
        if line.lower().startswith("date:") and date_not_found:
            case_date = line
            count = 0
            date_not_found = False
            continue
        if line.lower().startswith("appeal of") or line.lower().startswith("motion on"):
            file_num = line
            count = 0
            continue
        if line.lower().startswith("form ") and not plain_content_start:
            type_of_case.append(line)
            count = 0
            continue
        if not start_parse:
            continue
        if len(line) < 4 or line.replace(" ", "").isdigit():
            new_page = False
            count = 0
            continue
        if new_page:
            core_paras.append(line)
            plain_content_start = True
            count = 0
            new_page = False
            continue
        if count == 0:
            if line.startswith("•"):
                core_paras.append(line)
            elif core_paras:
                core_paras[-1] = core_paras[-1] + " " + line
            else:
                core_paras.append(line)
            plain_content_start = True
            continue
        if count == 1:
            if new_page and core_paras:
                core_paras[-1] = core_paras[-1] + " " + line
                new_page = False
            else:
                core_paras.append(line)
            count = 0
            plain_content_start = True
            continue
        if count > 2:
            new_page = True
            if line[0].islower() and core_paras:
                core_paras[-1] = core_paras[-1] + " " + line
            else:
                core_paras.append(line)
            plain_content_start = True

    return case_name, case_date, core_paras, type_of_case, file_num


def flatten_footnotes(footnote_text: str) -> list[str]:
    return [item for item in footnote_text.split("\n") if len(item) > 3]


def clean_footnotes(footnote_texts: list[str]) -> list[str]:
    final: list[str] = []
    for footnote in footnote_texts:
        if not final:
            final.append(footnote)
        elif final[-1].endswith((".", ".”")) or (final[-1].endswith("</core:emph>") and footnote[0].isdigit()):
            final.append(footnote)
        else:
            final[-1] += footnote
    return final


def continuity_fix(core_paras: list[str]) -> list[str]:
    sorted_paras: list[str] = []
    for count, para in enumerate(core_paras):
        para = para.strip()
        if not para:
            continue
        if count == 0:
            sorted_paras.append(para)
        elif para[0].islower():
            sorted_paras[-1] = sorted_paras[-1] + " " + para
        else:
            sorted_paras.append(para)
    return sorted_paras


def get_xmlstr_index(xml_file: str, marker: str) -> int:
    try:
        if xml_file.count(marker) > 1:
            return 0
        index = xml_file.index(marker)
        return index + len(marker)
    except ValueError:
        marker = marker[len(marker) // 2 :]
        try:
            if xml_file.count(marker) > 1:
                return 0
            index = xml_file.index(marker)
            return index + len(marker)
        except ValueError:
            marker = marker[len(marker) // 3 :]
            try:
                if xml_file.count(marker) > 1:
                    return 0
                index = xml_file.index(marker)
                return index + len(marker)
            except ValueError:
                return 0


def flag_footnotes(unplaced_footnotes: list[str], xml_file: str) -> str:
    flag = "".join(f"<core:flag.para>{fn}</core:flag.para>" for fn in unplaced_footnotes)
    return xml_file.replace(
        "<ad:judgmentbody>",
        "<ad:judgmentbody><core:para><core:flag sender=\"Automated\" recipient=\"PO\"><core:flag-subject>Correct placement for footnote not found</core:flag-subject><core:message><core:flag.para>Please check source file to check proper placement:</core:flag.para>"
        + flag
        + "</core:message></core:flag></core:para>",
    )


def handle_navigable_strings(xml_file: str, span, placed_fns: list[str], fn_number: str, footnote: str, unplaced_fns: list[str]):
    prevtext = span.previous_sibling
    if isinstance(prevtext, bs4.element.NavigableString):
        marker = " ".join(str(prevtext.string).split())
        index = get_xmlstr_index(xml_file, marker)
        if index != 0 and fn_number not in placed_fns:
            xml_file = (
                xml_file[:index]
                + f'<fn:footnote fr="{int(fn_number)}"><fn:para>'
                + footnote.split(" ", 1)[1]
                + "</fn:para></fn:footnote>"
                + xml_file[index + len(fn_number.strip()) :]
            )
            placed_fns.append(fn_number)
        else:
            unplaced_fns.append(footnote)
        return xml_file, placed_fns, unplaced_fns

    if prevtext is not None:
        marker = str(prevtext.previous_sibling) + str(prevtext.contents[0].string)
        marker = " ".join(marker.split())
        index = get_xmlstr_index(xml_file, marker)
        if index != 0 and fn_number not in placed_fns:
            xml_file = (
                xml_file[:index]
                + f'<fn:footnote fr="{int(fn_number)}"><fn:para>'
                + footnote.split(" ", 1)[1]
                + "</fn:para></fn:footnote>"
                + xml_file[index + len(fn_number.strip()) :]
            )
            placed_fns.append(fn_number)
        else:
            unplaced_fns.append(footnote)
        return xml_file, placed_fns, unplaced_fns

    unplaced_fns.append(footnote)
    return xml_file, placed_fns, unplaced_fns


def add_footnotes(clean_footnotes_list: list[str], xml_file: str, html_file: Path) -> str:
    with open(html_file, "r", encoding="utf-8", errors="ignore") as handle:
        soup = BeautifulSoup(handle.read(), "lxml")

    style_soup = soup.find_all(["style"])
    stylesheet = " ".join(str(style_soup).split())
    fn_classes = re.findall(FNSTYLE_PATTERN, stylesheet)
    fn_classes = [item[0:3].strip() for item in fn_classes]
    compiled_classes = re.compile("|".join(fn_classes)) if fn_classes else re.compile(r"^$")

    placed_fns: list[str] = []
    unplaced_fns: list[str] = []
    for footnote in clean_footnotes_list:
        footnote = footnote.strip()
        fn_number = footnote.split(" ", 1)[0]
        if not fn_number.isdigit():
            unplaced_fns.append(footnote)
            continue

        span = soup.find("span", text=fn_number, class_=compiled_classes)
        if span is None:
            span = soup.find("span", text=fn_number, style=re.compile("7.*"))
        if span is None:
            unplaced_fns.append(footnote)
            continue
        xml_file, placed_fns, unplaced_fns = handle_navigable_strings(
            xml_file,
            span,
            placed_fns,
            fn_number,
            footnote,
            unplaced_fns,
        )

    if unplaced_fns:
        xml_file = flag_footnotes(unplaced_fns, xml_file)

    return xml_file


def add_emphasis_to_xml(xml_file: str, html_file: Path) -> str:
    with open(html_file, "r", encoding="utf-8", errors="ignore") as handle:
        soup = BeautifulSoup(handle.read(), "lxml")

    xml_file = " ".join(xml_file.split())
    for emphasis in soup.find_all("i"):
        content = emphasis.string
        if not content:
            continue
        prevtext = emphasis.previous_sibling
        marker = ""
        index = 0
        if prevtext is not None and getattr(prevtext, "string", None) is not None:
            marker = " ".join((prevtext.string + content).split())
            index = get_xmlstr_index(xml_file, marker)
        if index != 0:
            content = content.strip()
            new_text = f"<core:emph typestyle=\"it\">{content}</core:emph>"
            xml_file = xml_file[: index - len(content)] + new_text + xml_file[index:]

    xml_file = re.sub(FURTHER_ORDER_PATTERN, '<core:emph typestyle="bf">FURTHER ORDER</core:emph>:', xml_file)
    xml_file = re.sub(ORDER_PATTERN, '<core:emph typestyle="bf">ORDER</core:emph>:', xml_file)
    return xml_file


def replace_lists(xml_string: str, html_file: Path) -> str:
    xml_string = html.unescape(xml_string)
    with open(html_file, "r", encoding="utf-8", errors="ignore") as handle:
        soup = BeautifulSoup(handle.read(), "lxml")
        for tag in soup.find_all(True):
            tag.attrs = {}

    list_nodes = soup.find_all(["ul"])
    if not list_nodes:
        return xml_string

    final_lists = []
    for ul_list in list_nodes:
        if not final_lists:
            final_lists.append(ul_list)
            continue
        for item in final_lists:
            if str(ul_list) in str(item):
                break
        else:
            final_lists.append(ul_list)

    list_markers = []
    try:
        for list_item in final_lists:
            previous_node = list_item.previous_sibling
            while previous_node and previous_node.text == "":
                previous_node = previous_node.previous_sibling
            next_node = list_item.next_sibling
            while next_node and next_node.text == "":
                next_node = next_node.next_sibling
            list_markers.append((previous_node.text, next_node.text))
    except Exception:
        return xml_string.replace(
            "<ad:judgmentbody>",
            "<ad:judgmentbody><core:para><core:flag sender=\"Automated\" recipient=\"PO\"><core:flag-subject>Lists not Tagged Properly</core:flag-subject><core:message><core:flag.para>Please check source file to check if core:list tagging is needed.</core:flag.para></core:message></core:flag></core:para>",
        )

    xml_lists = []
    for current_list in final_lists:
        soup = current_list
        for node in soup.find_all("ul"):
            node.name = "core:list"
        for node in soup.find_all("li"):
            node.name = "core:listitem"
        for node in soup.find_all("p"):
            node.name = "core:para"
        for node in soup.find_all("h2"):
            node.name = "core:generic-hd"
        for node in soup.find_all("span"):
            node.unwrap()
        soup.name = "core:list"
        xml_lists.append(soup)

    for marker in list_markers:
        previous_text, next_text = marker
        if previous_text in xml_string:
            previous_index = xml_string.index(previous_text) + len(previous_text)
        elif previous_text[-len(previous_text) // 2 :] in xml_string:
            previous_index = xml_string.index(previous_text[-len(previous_text) // 2 :]) + len(previous_text) // 2
        elif previous_text[-len(previous_text) // 3 :] in xml_string:
            previous_index = xml_string.index(previous_text[-len(previous_text) // 3 :]) + len(previous_text) // 3
        elif previous_text[-20:] in xml_string:
            previous_index = xml_string.index(previous_text[-20:]) + 20
        else:
            previous_index = "Not Found"

        if next_text in xml_string:
            next_index = xml_string.index(next_text)
        elif next_text[: len(next_text) // 2] in xml_string:
            next_index = xml_string.index(next_text[: len(next_text) // 2])
        elif next_text[: len(next_text) // 3] in xml_string:
            next_index = xml_string.index(next_text[: len(next_text) // 3])
        elif next_text[:30] in xml_string:
            next_index = xml_string.index(next_text[:30])
        else:
            next_index = "Not Found"

        if previous_index == "Not Found" or next_index == "Not Found" or next_index < previous_index:
            return xml_string.replace(
                "<ad:judgmentbody>",
                "<ad:judgmentbody><core:para><core:flag sender=\"Automated\" recipient=\"PO\"><core:flag-subject>Lists not Tagged Properly</core:flag-subject><core:message><core:flag.para>Please check source file to check if core:list tagging is needed.</core:flag.para></core:message></core:flag></core:para>",
            )

        while True:
            if xml_string[previous_index] == ">":
                previous_index += 1
                break
            previous_index += 1

        while True:
            if xml_string[next_index] == "<":
                break
            next_index -= 1

        xml_string = xml_string[:previous_index] + str(xml_lists[list_markers.index(marker)]) + xml_string[next_index:]

    return xml_string


def clean_xml_spaces(xml_file: str) -> str:
    return re.sub(r"\s+", " ", xml_file)


def write_aao_xml(
    clean_footnotes_list: list[str],
    case_name: str,
    filename: str,
    year: str,
    file_num: str,
    decision_date: str,
    type_of_case: list[str],
    core_paras: list[str],
    output_file: Path,
    html_file: Path,
) -> Path:
    root = xml.Element("ad:decision-grp", volnum=year)
    adjudication = xml.SubElement(root, "ad:adjudication", volnum=year)
    adjudication_info = xml.SubElement(adjudication, "ad:adjudication-info")

    regarding = xml.SubElement(adjudication_info, "ad:regarding")
    file_num_node = xml.SubElement(adjudication_info, "ad:filenum")
    file_num_node.text = "OFFICE: " + file_num
    case_name_node = xml.SubElement(regarding, "core:para", indent="none")
    case_name_node.text = case_name

    adjudicator_info = xml.SubElement(adjudication_info, "ad:adjudicator-info")
    juris = xml.SubElement(adjudicator_info, "core:juris")
    juris_info = xml.SubElement(juris, "lnci:jurisinfo")
    xml.SubElement(juris_info, "lnci:usa")
    juris_name = xml.SubElement(juris, "core:juris-name")
    juris_name.text = "Administrative Appeals Office"

    dates = xml.SubElement(adjudication_info, "ad:dates")
    decision_date_node = xml.SubElement(dates, "ad:decisiondate")
    decision_date_node.text = decision_date.upper().replace(".", "")

    typeofcase = xml.SubElement(adjudication_info, "ad:typeofcase")
    for case_type in type_of_case:
        xml.SubElement(typeofcase, "core:generic-hd", attrib={"align": "left", "typestyle": "ro"}).text = case_type

    content = xml.SubElement(adjudication, "ad:content")
    judgments = xml.SubElement(content, "ad:judgments")
    opinion = xml.SubElement(judgments, "ad:opinion")
    judgment_body = xml.SubElement(opinion, "ad:judgmentbody")

    for para in core_paras:
        match = re.match(r"[A-Z\s.]*", para)
        if para.startswith("Non-Precedent Decision of") or not para.strip():
            continue
        if match is not None and match[0] == para:
            xml.SubElement(judgment_body, "core:generic-hd", attrib={"align": "center", "typestyle": "ro"}).text = para
        else:
            xml.SubElement(judgment_body, "core:para", indent="none").text = para

    xml.SubElement(judgment_body, "core:generic-hd", attrib={"align": "left", "typestyle": "ro"}).text = filename

    output = xml.ElementTree(root)
    out = io.BytesIO()
    output.write(out)
    xml_string = HEADER_PLATE + out.getvalue().decode()
    xml_string = replace_lists(xml_string, html_file)
    xml_string = add_footnotes(clean_footnotes_list, xml_string, html_file)
    xml_string = clean_xml_spaces(xml_string)
    xml_string = add_emphasis_to_xml(xml_string, html_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write(xml_string)

    return output_file


def process_image_folder(
    input_folder: str | Path = DEFAULT_INPUT_FOLDER,
    *,
    recursive: bool = False,
    html_file: str | Path = DEFAULT_HTML_FILE,
    output_folder: str | Path = DEFAULT_OUTPUT_FOLDER,
) -> Path:
    input_path = Path(input_folder)
    html_path = Path(html_file)
    output_path = Path(output_folder)

    if not html_path.exists():
        raise FileNotFoundError(f"HTML file does not exist: {html_path}")
    if not html_path.is_file():
        raise FileNotFoundError(f"HTML path is not a file: {html_path}")

    image_files = collect_image_files(input_path, recursive=recursive)

    ocr = PaddleOCR(
        use_angle_cls=False,
        lang="en",
        use_gpu=False,
        enable_mkldnn=False,
        cpu_threads=2,
        rec_batch_num=2,
    )

    pages = [build_page_record(image_path, ocr) for image_path in image_files]
    raw_body = html.escape("\n".join(page.body_text for page in pages if page.body_text), quote=False)
    raw_footnotes = html.escape("\n".join(page.footnote_text for page in pages if page.footnote_text), quote=False)
    raw_body = replace_redaction_markers(raw_body)
    raw_footnotes = replace_redaction_markers(raw_footnotes)

    filename = f"{input_path.name}.docx"
    year = input_path.name[input_path.name.index("_") - 4 : input_path.name.index("_")]
    case_name, decision_date, core_paras, type_of_case, file_num = process_ocr_string(raw_body, filename)
    clean_footnotes_list = clean_footnotes(flatten_footnotes(raw_footnotes))
    core_paras = continuity_fix(core_paras)

    output_file = output_path / f"{input_path.name}.xml"
    return write_aao_xml(
        clean_footnotes_list,
        case_name,
        filename,
        year,
        file_num,
        decision_date,
        type_of_case,
        core_paras,
        output_file,
        html_path,
    )


def image_to_xml_main(
    input_folder: str | Path,
    html_folder_path: str | Path = DEFAULT_HTML_FILE.parent,
    output_folder: str | Path = DEFAULT_OUTPUT_FOLDER,
) -> Path:
    input_path = Path(input_folder)
    html_directory = Path(html_folder_path)
    html_candidates = [
        html_directory / f"{input_path.name}.htm",
        html_directory / f"{input_path.name}.html",
    ]

    for html_file in html_candidates:
        if html_file.exists() and html_file.is_file():
            return process_image_folder(
                input_folder=input_path,
                html_file=html_file,
                output_folder=output_folder,
            )

    expected_files = ", ".join(str(path) for path in html_candidates)
    raise FileNotFoundError(
        f"Could not find a matching HTML file for {input_path.name}. Expected one of: {expected_files}"
    )


def main() -> Path:
    output_file = image_to_xml_main(DEFAULT_INPUT_FOLDER, DEFAULT_HTML_FILE.parent, DEFAULT_OUTPUT_FOLDER)
    print(f"XML created: {html.escape(str(output_file), quote=False)}")
    return output_file


if __name__ == "__main__":
    main()