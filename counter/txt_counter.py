import re
from pathlib import Path


WORD_PATTERN = re.compile(r"\S+")





def read_text_file(file_path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError("unknown", b"", 0, 1, "Unable to decode text file.")


def count_text_words(file_path: Path) -> int:
    text = read_text_file(file_path)
    return len(WORD_PATTERN.findall(text))


def list_text_files(folder_path: Path) -> list[Path]:
    return sorted(
        path for path in folder_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    )


def write_word_counts(output_path: Path, results: list[str], total_word_count: int) -> None:
    output_path.write_text("\n".join(results + [f"Total - {total_word_count}"]) + "\n", encoding="utf-8")


def txt_counter_main(input_folder, output_file):

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_folder}")

    text_files = list_text_files(input_folder)
    if not text_files:
        raise FileNotFoundError(f"No text files found in: {input_folder}")

    results = []
    total_word_count = 0

    for txt_file in text_files:
        word_count = count_text_words(txt_file)
        results.append(f"{txt_file.name} - {word_count}")
        total_word_count += word_count
        print(f"{txt_file.name} - {word_count}")

    write_word_counts(output_file, results, total_word_count)
    return total_word_count


if __name__ == "__main__":
    DEFAULT_INPUT_FOLDER = Path(r"C:\Users\tior\Downloads\OneDrive_1_4-24-2026\output_ptt\texts")
    DEFAULT_OUTPUT_FILE = Path(r"C:\Users\tior\Downloads\OneDrive_1_4-24-2026\output_ptt\file_count.txt")

    txt_counter_main(DEFAULT_INPUT_FOLDER, DEFAULT_OUTPUT_FILE)