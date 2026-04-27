import re
import xml.etree.ElementTree as ET
from pathlib import Path

WORD_PATTERN = re.compile(r"\S+")
FALLBACK_TEXT_PATTERN = re.compile(r">([^<]+)<")


def read_xml_file(file_path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError("unknown", b"", 0, 1, "Unable to decode XML file.")


def extract_text_inside_tags(xml_text: str) -> str:
    try:
        root = ET.fromstring(xml_text)
        text_chunks = [chunk.strip() for chunk in root.itertext() if chunk and chunk.strip()]
        if text_chunks:
            return " ".join(text_chunks)
    except ET.ParseError:
        pass

    fallback_chunks = [chunk.strip() for chunk in FALLBACK_TEXT_PATTERN.findall(xml_text) if chunk.strip()]
    return " ".join(fallback_chunks)


def count_words_in_text(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def count_xml_text_words(file_path: Path) -> int:
    xml_text = read_xml_file(file_path)
    extracted_text = extract_text_inside_tags(xml_text)
    return count_words_in_text(extracted_text)


def list_xml_files(folder_path: Path) -> list[Path]:
    return sorted(
        path for path in folder_path.rglob("*.xml")
        if path.is_file()
    )


def write_word_counts(output_path: Path, results: list[str], total_word_count: int) -> None:
    output_path.write_text("\n".join(results + [f"Total - {total_word_count}"]) + "\n", encoding="utf-8")


def main() -> int:
    input_folder = Path(r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\Rel 118 to 133 XMLS")
    output_file = Path(r"c:\Users\tior\Documents\PROJECTS\AutoTag v1.1\counter\file_count.txt")

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_folder}")

    xml_files = list_xml_files(input_folder)
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in: {input_folder}")

    results = []
    total_word_count = 0

    for xml_file in xml_files:
        word_count = count_xml_text_words(xml_file)
        relative_name = xml_file.relative_to(input_folder).as_posix()
        results.append(f"{relative_name} - {word_count}")
        total_word_count += word_count
        print(f"{relative_name} - {word_count}")

    write_word_counts(output_file, results, total_word_count)
    return total_word_count

if __name__ == "__main__":
    main()