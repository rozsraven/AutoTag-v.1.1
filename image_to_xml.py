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
import numpy as np
from bs4 import BeautifulSoup
from dateutil.parser import parse

from run_paddle_ocr import create_paddle_ocr, extract_text_from_image


os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_INPUT_FOLDER = Path(r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\images\JAN132025_01D6101")
DEFAULT_HTML_FILE = Path(r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\PDF\AAO\JAN132025_01D6101.htm")
DEFAULT_OUTPUT_FOLDER = Path(r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\output_folder\xml")
process_type = "AAO"
REDACTION_MARKERS = ("xx", "XX", "Xx", "xX")
REDACTION_REPLACEMENT = '<core:emph typestyle="bf">[redacted]</core:emph>'
HEADER_PLATE = "<?xml version='1.0' encoding='UTF-8'?><!DOCTYPE ad:decision-grp PUBLIC \"-//LEXISNEXIS//DTD Admin-Interp-pub v005//EN//XML\" \"D:\\Pub383\\DTD\\DTD\\admin-interp-pubV005-0000\\admin-interp-pubV005-0000.dtd\"><?Pub EntList alpha bull copy rArr sect trade para mdash ldquo lsquo rdquo rsquo dagger hellip ndash frac23 amp?>"
BIA_HEADER_PLATE = r"""<?xml version="1.0" encoding="UTF-8"?><!--Arbortext, Inc., 1988-2015, v.4002--><!DOCTYPE ad:decision-grp PUBLIC "-//LEXISNEXIS//DTD Admin-Interp-pub v005//EN//XML" "C:\Neptune\NeptuneEditor\doctypes\admin-interp-pubV005-0000\admin-interp-pubV005-0000.dtd"><?Pub UDT delete _display FontColor="red" StrikeThrough="yes" Composed="yes" Save="yes"?><?Pub UDT add _display FontColor="blue" Underline="yes" Composed="yes" Save="yes"?><?Pub UDT plpt_delete _display FontColor="red" StrikeThrough="yes" Composed="yes" Save="yes"?><?Pub UDT plpt_add _display FontColor="blue" Underline="yes" Composed="yes" Save="yes"?><?Pub UDT notes_delete _display FontColor="red" StrikeThrough="yes" Composed="yes" Save="yes"?><?Pub UDT notes_add _display FontColor="blue" Underline="yes" Composed="yes" Save="yes"?><?Pub UDT EditAid-UC _display FontColor="orange" Composed="yes" Save="yes"?><?Pub UDT EditAid-Quotes _display FontColor="green" Composed="yes" Save="yes"?><?Pub UDT EditAid-Hyphens _display FontColor="violet" Composed="yes" Save="yes"?><?Pub UDT Mod _pi?><?Pub UDT AttrModified _display BackColor="#c0c0c0" StrikeThrough="no" Composed="yes" Save="yes"?><?Pub UDT Deleted _display BackColor="#ffffc0" Composed="yes" Save="yes"?><?Pub UDT Inserted _display BackColor="#ffc0ff" Composed="yes" Save="yes"?><?Pub EntList alpha bull copy rArr sect trade para mdash ldquo lsquo rdquo rsquo dagger hellip ndash frac23 amp?><?Pub Inc?>"""
BIA_FOOTER_PLATE = "<?Pub Caret -2?>"
BIA_END_MARKER = "</ad:decision-grp>"
BIA_EXPANDED_FORM = {
    "~/ib~": "</core:emph>",
    "~/IB~": "</core:emph>",
    "~/i~": "</core:emph>",
    "~/I~": "</core:emph>",
    "~/b~": "</core:emph>",
    "~/B~": "</core:emph>",
    "~b~": '<core:emph typestyle = "bf">',
    "~B~": '<core:emph typestyle = "bf">',
    "~i~": '<core:emph typestyle="it">',
    "~I~": '<core:emph typestyle="it">',
    "~ib~": '<core:emph typestyle="ib">',
    "~IB~": '<core:emph typestyle="ib">',
    "~u~": '<core:emph typestyle="un">',
    "~U~": '<core:emph typestyle="un">',
    "&N": "&amp;N",
}
FNSTYLE_PATTERN = r"[^.*]*font-size: 7.*?}"
FURTHER_ORDER_PATTERN = re.compile(r"(F+U+R+T+H+E+R+\s+O+R+D+E+R+:)")
ORDER_PATTERN = re.compile(r"(O+R+D+E+R+:)(?!<)")
_WHITESPACE_PATTERN = re.compile(r"\s+")
VALID_PROCESS_TYPES = {"AAO", "BIA"}
_SPLIT_SCALE = 0.35
_MIN_HORIZONTAL_WIDTH = 300


@dataclass
class ImageXmlPage:
    source_image: str
    body_image: str
    footnote_image: str | None
    body_text: str
    footnote_text: str


@dataclass
class BiaOcrParseResult:
    juris_name: str
    case_name: str
    file_num: str
    decision_date: str
    type_of_case: list[str]
    parties: list[str]
    core_paras: list[str]
    complete: bool
    multiparty: bool


BIA_TYPE_CASES = (
    "IN REMOVAL PROCEEDINGS",
    "APPEAL",
    "APPLICATION",
    "INTERLOCUTORY APPEAL",
    "MOTION",
    "CHARGE",
    "IN DEPORTATION PROCEEDINGS",
)

BIA_PARTICIPANT_MARKERS = (
    "ON BEHALF OF",
    "Assistant Chief",
    "Senior Litigation Coordinator",
    "Senior Attorney",
    "Deputy Chief Counsel",
    "Associate Counsel",
    "Associate Legal Advisor",
    "Assistant Chief Counsel",
)

BIA_REGARDING_MARKERS = (
    "a.k.a. ",
    "In re: ",
    "In re ",
    "Matter of ",
)

BIA_HEADER_SKIP_PREFIXES = (
    "Falls Church",
    "U.S. Department of Justice",
    "Userteam:",
)


def normalize_process_type(value: str) -> str:
    normalized = value.strip().upper()
    if normalized not in VALID_PROCESS_TYPES:
        expected = ", ".join(sorted(VALID_PROCESS_TYPES))
        raise ValueError(f"Unsupported process_type: {value}. Expected one of: {expected}")
    return normalized


def find_matching_html_file(input_path: Path, html_directory: Path) -> Path | None:
    html_candidates = [
        html_directory / f"{input_path.name}.htm",
        html_directory / f"{input_path.name}.html",
    ]

    for html_file in html_candidates:
        if html_file.exists() and html_file.is_file():
            return html_file

    return None


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


def normalize_text(text: str) -> str:
    normalized = text.replace("\u00A0", " ")
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


def iter_ocr_lines(ocr_text: str | Iterable[str]) -> Iterable[str]:
    if isinstance(ocr_text, str):
        yield from ocr_text.splitlines()
        return

    first_page = True
    for page_text in ocr_text:
        if not first_page:
            yield ""
            yield ""
            yield ""
        first_page = False
        yield from page_text.splitlines()


def extract_year_token(text: str) -> str | None:
    match = re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", text)
    if match is None:
        return None
    return match.group(0)


def infer_bia_year(
    input_path: Path,
    bia_result: BiaOcrParseResult,
    raw_body_text: str,
    html_path: Path | None,
) -> str | None:
    candidate_texts = [
        bia_result.decision_date,
        input_path.name,
        bia_result.file_num,
        bia_result.case_name,
        raw_body_text,
    ]

    for candidate in candidate_texts:
        if not candidate:
            continue
        year = extract_year_token(candidate)
        if year is not None:
            return year

    if html_path is None or not html_path.exists() or not html_path.is_file():
        return None

    with open(html_path, "r", encoding="utf-8", errors="ignore") as handle:
        html_text = BeautifulSoup(handle.read(), "lxml").get_text(" ", strip=True)

    return extract_year_token(html_text)


def infer_aao_year(
    input_path: Path,
    decision_date: str,
    raw_body_text: str,
    html_path: Path | None,
) -> str | None:
    candidate_texts = [
        input_path.name,
        decision_date,
        raw_body_text,
    ]

    for candidate in candidate_texts:
        if not candidate:
            continue

        year = extract_year_token(candidate)
        if year is not None:
            return year

        try:
            parsed_date = parse(candidate, fuzzy=True)
        except Exception:
            continue

        if 1900 <= parsed_date.year <= 2099:
            return str(parsed_date.year)

    if html_path is None or not html_path.exists() or not html_path.is_file():
        return None

    with open(html_path, "r", encoding="utf-8", errors="ignore") as handle:
        html_text = BeautifulSoup(handle.read(), "lxml").get_text(" ", strip=True)

    year = extract_year_token(html_text)
    if year is not None:
        return year

    try:
        parsed_date = parse(html_text, fuzzy=True)
    except Exception:
        return None

    if 1900 <= parsed_date.year <= 2099:
        return str(parsed_date.year)

    return None


def _discard_large_connected_components(image: np.ndarray) -> np.ndarray:
    image = np.uint8(image)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=4)

    wide_components = np.isin(labels, np.where(stats[:, cv2.CC_STAT_WIDTH] > _MIN_HORIZONTAL_WIDTH)[0])
    tall_components = np.isin(labels, np.where(stats[:, cv2.CC_STAT_HEIGHT] > _MIN_HORIZONTAL_WIDTH)[0])
    image[(wide_components | tall_components)] = 0
    return image


def split_body_and_footnotes(image_paths: Iterable[Path]) -> tuple[list[str], list[str]]:
    body_images: list[str] = []
    footnote_images: list[str] = []

    for image_path in image_paths:
        source_path = Path(image_path)
        original = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        if original is None:
            body_images.append(str(source_path))
            continue

        resized = cv2.resize(original, None, fx=_SPLIT_SCALE, fy=_SPLIT_SCALE, interpolation=cv2.INTER_LINEAR)
        height, width = resized.shape[:2]
        inverted = 255 - resized
        _, thresholded = cv2.threshold(inverted, 5, 255, cv2.THRESH_BINARY)

        without_lines = _discard_large_connected_components(thresholded.copy())
        horizontal = cv2.bitwise_xor(thresholded, without_lines)
        kernel = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            dtype=np.uint8,
        )
        horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, kernel, iterations=2)

        if not horizontal.any():
            body_images.append(str(source_path))
            continue

        contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            body_images.append(str(source_path))
            continue

        split_y = None
        for contour in sorted(contours, key=lambda current: cv2.boundingRect(current)[2], reverse=True):
            _, y, _, contour_height = cv2.boundingRect(contour)
            candidate = y + contour_height
            if 0 < candidate < height:
                split_y = candidate
                break

        if split_y is None:
            body_images.append(str(source_path))
            continue

        original_image = cv2.imread(str(source_path))
        if original_image is None:
            body_images.append(str(source_path))
            continue

        original_height, original_width = original_image.shape[:2]
        scaled_split_y = max(1, min(original_height - 1, int(split_y / _SPLIT_SCALE)))

        body_output = source_path.with_name(f"{source_path.stem}_body{source_path.suffix}")
        footnote_output = source_path.with_name(f"{source_path.stem}_footnotes{source_path.suffix}")

        body_slice = original_image[0:scaled_split_y, 0:original_width]
        footnote_slice = original_image[scaled_split_y:original_height, 0:original_width]

        if body_slice.size == 0:
            body_images.append(str(source_path))
            continue

        cv2.imwrite(str(body_output), body_slice)
        body_images.append(str(body_output))

        if footnote_slice.size != 0:
            cv2.imwrite(str(footnote_output), footnote_slice)
            footnote_images.append(str(footnote_output))

    return body_images, footnote_images


def extract_page_text(image_path: str | Path, ocr) -> str:
    return extract_text_from_image(image_path, ocr, normalize_legal=True)


def build_page_record(image_path: Path, ocr) -> ImageXmlPage:
    if image_path.stem.endswith("_body"):
        body_image = str(image_path)
        footnote_candidate = image_path.with_name(f"{image_path.stem[:-5]}_footnotes{image_path.suffix}")
        footnote_image = str(footnote_candidate) if footnote_candidate.exists() else None
    else:
        body_images, footnote_images = split_body_and_footnotes([image_path])
        body_image = body_images[0] if body_images else str(image_path)
        footnote_image = footnote_images[0] if footnote_images else None

    body_text = extract_page_text(body_image, ocr)
    footnote_text = extract_page_text(footnote_image, ocr) if footnote_image else ""

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


def process_ocr_string(ocr_text: str | Iterable[str], filename: str):
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

    for line in iter_ocr_lines(ocr_text):
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


def is_date(text: str, *, fuzzy: bool = False) -> bool:
    try:
        parse(text, fuzzy=fuzzy)
        return True
    except Exception:
        return False


def process_bia_ocr_string(ocr_text: str | Iterable[str]):
    start_parse = False
    body_started = False
    complete = False
    multiparty = False
    juris_name = "Decision of the Board of Immigration Appeals"
    case_name_parts: list[str] = []
    type_of_case: list[str] = []
    parties: list[str] = []
    core_paras: list[str] = []
    file_num = "Filenum not found"
    decision_date = "~FD~"
    blank_count = 0
    new_page = False

    for raw_line in iter_ocr_lines(ocr_text):
        paragraph = normalize_text(html.unescape(raw_line))
        if not paragraph:
            blank_count += 1
            continue

        if blank_count > 2:
            new_page = True
        blank_count = 0

        upper_paragraph = paragraph.upper()

        if "Decision of the Board of" in paragraph:
            juris_name = paragraph[paragraph.find("Decision") :]
            start_parse = True
            new_page = False
            continue

        if paragraph.startswith(BIA_HEADER_SKIP_PREFIXES):
            start_parse = True
            new_page = False
            continue

        if upper_paragraph.startswith("UNITED STATES DEPARTMENT OF JUSTICE EXECUTIVE"):
            break

        if not start_parse:
            continue

        if " respectfully dissents " in paragraph:
            core_paras.append(paragraph)
            new_page = False
            continue

        if "FOR THE BOARD" in upper_paragraph:
            complete = True
            new_page = False
            continue

        if paragraph.startswith(("File:", "Files:")):
            file_num = paragraph
            body_started = True
            new_page = False
            continue

        cleaned_date = paragraph.replace("Date:", "").replace("DATE:", "").replace("-", " ").replace(" _", " ").strip()
        if decision_date == "~FD~" and cleaned_date and len(cleaned_date) < 40 and is_date(cleaned_date, fuzzy=True):
            decision_date = cleaned_date
            new_page = False
            continue

        if paragraph.startswith(BIA_REGARDING_MARKERS):
            for current_type in BIA_TYPE_CASES:
                if current_type in upper_paragraph and current_type not in type_of_case:
                    type_of_case.append(current_type)
                    paragraph = paragraph.replace(current_type, "").strip()
                    upper_paragraph = paragraph.upper()
            case_name_parts.append(paragraph)
            body_started = True
            new_page = False
            continue

        if paragraph.startswith(BIA_TYPE_CASES):
            if paragraph not in type_of_case:
                type_of_case.append(paragraph)
            body_started = True
            new_page = False
            continue

        if paragraph.startswith(BIA_PARTICIPANT_MARKERS):
            if " ORDER:" in paragraph:
                paragraph = paragraph.replace(" ORDER:", "")
                core_paras.insert(0, "ORDER:")

            if "RESPONDENTS" in upper_paragraph:
                multiparty = True
                cleaned_party = (
                    paragraph.replace("ON BEHALF OF RESPONDENTS: ", "")
                    .replace("ON BEHALF OF THE RESPONDENTS: ", "")
                    .replace("ON BEHALF OF THE RESPONDENTS : ", "")
                    .replace("ON BEHALF OF RESPONDENTS : ", "")
                )
            else:
                cleaned_party = (
                    paragraph.replace("ON BEHALF OF DHS: ", "")
                    .replace("ON BEHALF OF RESPONDENT: ", "")
                    .replace("ON BEHALF OF THE DHS: ", "")
                    .replace("ON BEHALF OF RESPONDENT : ", "")
                    .replace("ON BEHALF OF DHS : ", "")
                    .replace("ON BEHALF OF DHS:", "")
                )

            parties.append(cleaned_party.strip())
            body_started = True
            new_page = False
            continue

        if len(paragraph) < 3 or (paragraph.startswith("~") and len(paragraph) <= 6):
            continue

        if not body_started and not case_name_parts:
            case_name_parts.append(paragraph)
            new_page = False
            continue

        if new_page and core_paras and paragraph[0].islower():
            core_paras[-1] = core_paras[-1] + " " + paragraph
        else:
            core_paras.append(paragraph)
        new_page = False

    case_name = " ".join(case_name_parts).strip() or "caseNameNotFound"
    if case_name == "caseNameNotFound" and core_paras:
        case_name = core_paras[0]
        core_paras = core_paras[1:]

    core_paras = continuity_fix(core_paras)

    return BiaOcrParseResult(
        juris_name=juris_name,
        case_name=case_name,
        file_num=file_num,
        decision_date=decision_date,
        type_of_case=type_of_case,
        parties=parties,
        core_paras=core_paras,
        complete=complete,
        multiparty=multiparty,
    )


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


def expand_tokens(xml_string: str, replacements: dict[str, str]) -> str:
    for short_token, expanded_token in replacements.items():
        xml_string = xml_string.replace(short_token, expanded_token)
    return xml_string


def unescape_serialized_xml_tags(xml_string: str) -> str:
    return re.sub(
        r"&lt;(/?(?:core|fn|ad|lnci):[A-Za-z0-9._-]+(?:\s+[^&<>]*?)?)&gt;",
        r"<\1>",
        xml_string,
    )


def add_bia_flag(parent, subject: str, message: str):
    para = xml.SubElement(parent, "core:para")
    flag = xml.SubElement(para, "core:flag", sender="Automated", recipient="PO")
    xml.SubElement(flag, "core:flag-subject").text = subject
    flag_message = xml.SubElement(flag, "core:message")
    xml.SubElement(flag_message, "core:flag.para").text = message


def append_bia_participants(parent, parties: list[str], multiparty: bool):
    if not parties:
        return

    participants = xml.SubElement(parent, "ad:participants")

    respondent = xml.SubElement(participants, "ad:participant")
    xml.SubElement(respondent, "ad:relationship-phrase").text = "ON BEHALF OF "
    xml.SubElement(respondent, "ad:party").text = "RESPONDENTS: " if multiparty else "RESPONDENT: "
    respondent_person = xml.SubElement(respondent, "core:person")
    xml.SubElement(respondent_person, "core:name.text").text = parties[0]

    if len(parties) > 1:
        dhs_participant = xml.SubElement(participants, "ad:participant")
        xml.SubElement(dhs_participant, "ad:relationship-phrase").text = "ON BEHALF OF "
        xml.SubElement(dhs_participant, "ad:party").text = "DHS: "
        dhs_person = xml.SubElement(dhs_participant, "core:person")
        xml.SubElement(dhs_person, "core:name.text").text = parties[1]
        for extra_party in parties[2:]:
            xml.SubElement(dhs_person, "core:name.text").text = " " + extra_party


def append_bia_judgment_body(judgment_body, core_paras: list[str]):
    for para in core_paras:
        cleaned_para = para.strip()
        if not cleaned_para:
            continue
        if cleaned_para == "ORDER:":
            xml.SubElement(judgment_body, "core:generic-hd", align="left").text = cleaned_para
            continue
        if cleaned_para.startswith("~gh~"):
            xml.SubElement(judgment_body, "core:generic-hd", align="center").text = cleaned_para.replace("~gh~", "").replace("~li~", "")
            continue
        if cleaned_para.startswith("On Appeal from ") or cleaned_para.startswith("Before:"):
            xml.SubElement(judgment_body, "core:generic-hd", align="center").text = cleaned_para
            continue
        if cleaned_para.startswith("ORDER:") or cleaned_para.startswith("FURTHER ORDER:"):
            xml.SubElement(judgment_body, "core:para", indent="none").text = cleaned_para
            continue
        if re.fullmatch(r"[A-Z][A-Z,\-\s.']*Immigration Judge", cleaned_para):
            xml.SubElement(judgment_body, "core:para").text = cleaned_para
            continue
        xml.SubElement(judgment_body, "core:para", indent="none").text = cleaned_para.replace("~in0~", "")


def append_bia_board_signature(judgment_body, pdf_filename: str):
    xml.SubElement(judgment_body, "core:generic-hd", align="center").text = "FOR THE BOARD"
    xml.SubElement(judgment_body, "core:para").text = pdf_filename


def add_bia_emphasis_to_xml(xml_file: str, html_file: Path | None) -> str:
    if html_file is None or not html_file.exists() or not html_file.is_file():
        return xml_file

    with open(html_file, "r", encoding="utf-8", errors="ignore") as handle:
        soup = BeautifulSoup(handle.read(), "lxml")

    body_start_marker = "<ad:judgmentbody>"
    body_end_marker = "</ad:judgmentbody>"
    body_start = xml_file.find(body_start_marker)
    body_end = xml_file.rfind(body_end_marker)
    if body_start == -1 or body_end == -1 or body_end <= body_start:
        return xml_file

    prefix = xml_file[: body_start + len(body_start_marker)]
    body = xml_file[body_start + len(body_start_marker) : body_end]
    suffix = xml_file[body_end:]
    normalized_body = " ".join(body.split())

    for emphasis in soup.find_all(["i", "em"]):
        content = normalize_text(emphasis.get_text(" ", strip=True))
        if not content:
            continue

        prevtext = emphasis.previous_sibling
        marker = ""
        index = 0
        if isinstance(prevtext, bs4.element.NavigableString):
            marker = normalize_text(f"{prevtext}{content}")
            index = get_xmlstr_index(normalized_body, marker)
        elif prevtext is not None and getattr(prevtext, "string", None) is not None:
            marker = normalize_text(f"{prevtext.string}{content}")
            index = get_xmlstr_index(normalized_body, marker)

        if index == 0:
            continue

        new_text = f'<core:emph typestyle="it">{content}</core:emph>'
        normalized_body = normalized_body[: index - len(content)] + new_text + normalized_body[index:]

    return prefix + normalized_body + suffix


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

    xml.SubElement(
        judgment_body,
        "core:generic-hd",
        attrib={"align": "left", "typestyle": "ro"},
    ).text = filename.replace(".docx", ".pdf")

    output = xml.ElementTree(root)
    out = io.BytesIO()
    output.write(out)
    xml_string = HEADER_PLATE + out.getvalue().decode()
    xml_string = replace_lists(xml_string, html_file)
    xml_string = add_footnotes(clean_footnotes_list, xml_string, html_file)
    xml_string = clean_xml_spaces(xml_string)
    xml_string = add_emphasis_to_xml(xml_string, html_file)
    xml_string = unescape_serialized_xml_tags(xml_string)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write(xml_string)

    return output_file


def write_bia_xml(
    filename: str,
    year: str,
    bia_result: BiaOcrParseResult,
    output_file: Path,
    html_file: Path | None,
) -> Path:
    root = xml.Element("ad:decision-grp", volnum=year)
    comment_tags = ["pub-num=00383", "ch-num=relXXXbiaX", "00383-relXXXbiaX", filename[:-1]]
    for comment in comment_tags:
        root.append(xml.Comment(comment))

    adjudication = xml.SubElement(root, "ad:adjudication", volnum=year)
    adjudication_info = xml.SubElement(adjudication, "ad:adjudication-info")

    regarding = xml.SubElement(adjudication_info, "ad:regarding")
    case_name_node = xml.SubElement(regarding, "core:para", {"indent": "none"})
    case_name_node.text = bia_result.case_name

    file_num_node = xml.SubElement(adjudication_info, "ad:filenum")
    file_num_node.text = bia_result.file_num

    adjudicator_info = xml.SubElement(adjudication_info, "ad:adjudicator-info")
    juris = xml.SubElement(adjudicator_info, "core:juris")
    juris_info = xml.SubElement(juris, "lnci:jurisinfo")
    xml.SubElement(juris_info, "lnci:usa")
    xml.SubElement(juris, "core:juris-name").text = bia_result.juris_name

    dates = xml.SubElement(adjudication_info, "ad:dates")
    normalized_date = bia_result.decision_date.replace("~i~", "").replace("~/i~", "")
    xml.SubElement(dates, "ad:decisiondate").text = "Date: " + normalized_date

    typeofcase = xml.SubElement(adjudication_info, "ad:typeofcase")
    for case_type in bia_result.type_of_case:
        xml.SubElement(typeofcase, "core:generic-hd", {"align": "left", "typestyle": "ro"}).text = case_type

    append_bia_participants(adjudication_info, bia_result.parties, bia_result.multiparty)

    content = xml.SubElement(adjudication, "ad:content")
    judgments = xml.SubElement(content, "ad:judgments")
    opinion = xml.SubElement(judgments, "ad:opinion")
    judgment_body = xml.SubElement(opinion, "ad:judgmentbody")

    append_bia_judgment_body(judgment_body, bia_result.core_paras)
    append_bia_board_signature(judgment_body, filename.replace(".docx", ".pdf"))

    output = xml.ElementTree(root)
    out = io.BytesIO()
    output.write(out)
    xml_string = expand_tokens(BIA_HEADER_PLATE + out.getvalue().decode().replace(" & ", " &amp; "), BIA_EXPANDED_FORM)
    marker_index = xml_string.index(BIA_END_MARKER)
    xml_string = xml_string[:marker_index] + BIA_FOOTER_PLATE + xml_string[marker_index:]
    xml_string = clean_xml_spaces(xml_string)
    xml_string = add_bia_emphasis_to_xml(xml_string, html_file)
    xml_string = unescape_serialized_xml_tags(xml_string)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write(xml_string)

    return output_file

def process_image_folder(
    input_folder: str | Path = DEFAULT_INPUT_FOLDER,
    *,
    recursive: bool = False,
    html_file: str | Path | None = DEFAULT_HTML_FILE,
    output_folder: str | Path = DEFAULT_OUTPUT_FOLDER,
    process_type_name: str = process_type,
) -> Path:
    normalized_process_type = normalize_process_type(process_type_name)
    input_path = Path(input_folder)
    html_path = Path(html_file) if html_file is not None else None
    output_path = Path(output_folder)

    if normalized_process_type == "AAO":
        if html_path is None:
            raise FileNotFoundError("AAO conversion requires a matching HTML file.")
        if not html_path.exists():
            raise FileNotFoundError(f"HTML file does not exist: {html_path}")
        if not html_path.is_file():
            raise FileNotFoundError(f"HTML path is not a file: {html_path}")
    elif html_path is not None and html_path.exists() and not html_path.is_file():
        raise FileNotFoundError(f"HTML path is not a file: {html_path}")

    image_files = collect_image_files(input_path, recursive=recursive)

    ocr = create_paddle_ocr()

    pages = [build_page_record(image_path, ocr) for image_path in image_files]
    body_page_texts = [replace_redaction_markers(html.escape(page.body_text, quote=False)) for page in pages if page.body_text]
    footnote_page_texts = [replace_redaction_markers(html.escape(page.footnote_text, quote=False)) for page in pages if page.footnote_text]
    raw_body = "\n".join(body_page_texts)
    raw_footnotes = "\n".join(footnote_page_texts)

    filename = f"{input_path.name}.docx"
    output_file = output_path / f"{input_path.name}.xml"
    if normalized_process_type == "BIA":
        bia_result = process_bia_ocr_string(body_page_texts)
        year = infer_bia_year(input_path, bia_result, raw_body, html_path)
        if year is None:
            raise ValueError(
                f"Could not determine a 4-digit year for BIA input '{input_path.name}'. "
                "Expected the decision date OCR text, folder name, OCR body text, or matching HTML file to contain a year."
            )
        return write_bia_xml(
            filename,
            year,
            bia_result,
            output_file,
            html_path,
        )

    case_name, decision_date, core_paras, type_of_case, file_num = process_ocr_string(body_page_texts, filename)
    year = infer_aao_year(input_path, decision_date, raw_body, html_path)
    if year is None:
        raise ValueError(
            f"Could not determine a 4-digit year for AAO input '{input_path.name}'. "
            "Expected the folder name, OCR decision date, OCR body text, or matching HTML file to contain a year."
        )
    clean_footnotes_list = clean_footnotes(flatten_footnotes(raw_footnotes))
    core_paras = continuity_fix(core_paras)
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
    process_type_name: str = process_type,
) -> Path:
    normalized_process_type = normalize_process_type(process_type_name)
    input_path = Path(input_folder)
    html_directory = Path(html_folder_path)
    html_file = find_matching_html_file(input_path, html_directory)

    if normalized_process_type == "AAO" and html_file is None:
        html_candidates = [
            html_directory / f"{input_path.name}.htm",
            html_directory / f"{input_path.name}.html",
        ]
        expected_files = ", ".join(str(path) for path in html_candidates)
        raise FileNotFoundError(
            f"Could not find a matching HTML file for {input_path.name}. Expected one of: {expected_files}"
        )

    return process_image_folder(
        input_folder=input_path,
        html_file=html_file,
        output_folder=output_folder,
        process_type_name=normalized_process_type,
    )


def main() -> Path:
    output_file = image_to_xml_main(
        DEFAULT_INPUT_FOLDER,
        DEFAULT_HTML_FILE.parent,
        DEFAULT_OUTPUT_FOLDER,
        process_type_name=process_type,
    )
    print(f"XML created: {html.escape(str(output_file), quote=False)}")
    return output_file


if __name__ == "__main__":
    main()