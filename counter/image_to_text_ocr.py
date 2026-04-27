# ==========================================
# MUST BE FIRST (Fix oneDNN / MKL crash)
# ==========================================
import os
import re
import time
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"

from run_paddle_ocr import create_paddle_ocr, extract_text_from_image, force_memory_cleanup


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_dotenv_if_present(env_path: Optional[Path] = None) -> None:
    if env_path is None:
        env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        return


def fix_common_cfr_subsections(text: str) -> str:
    return re.sub(
        r"(8\s*C\.F\.R\.\s*§\s*1003\.1\s*\(\s*d\s*\)\s*\(\s*3\s*\)\s*)\(\s*i\s*\)",
        r"\1(ii)",
        text,
        flags=re.IGNORECASE,
    )


def list_image_files(folder: Path) -> list[Path]:
    return sorted(
        [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def list_image_folders(folder: Path) -> list[Path]:
    return sorted([path for path in folder.iterdir() if path.is_dir()])


def _write_text_file(text: str, text_path: str | Path) -> None:
    output_path = Path(text_path)
    output_path.write_text(text.rstrip() + "\n" if text else "", encoding="utf-8")


def process_image_worker(image_path: str | Path, queue: Optional[Queue] = None) -> None:
    ocr = None

    try:
        ocr = create_paddle_ocr()
        extracted_text = extract_text_from_image(image_path, ocr, normalize_legal=True)
        extracted_text = fix_common_cfr_subsections(extracted_text)

        print(f"Processed: {image_path}")

        if queue is not None:
            queue.put((str(image_path), "success", extracted_text))
    except Exception as exc:
        print(f"Failed processing {image_path}: {exc}")
        if queue is not None:
            queue.put((str(image_path), "fail", ""))
    finally:
        if ocr is not None:
            del ocr
        force_memory_cleanup()


def process_image_folder(input_folder: Path, output_folder: Path) -> tuple[int, int, Path]:
    image_files = list_image_files(input_folder)
    if not image_files:
        raise FileNotFoundError(f"No image files found in folder: {input_folder}")

    print(f"Found {len(image_files)} image(s) in {input_folder.name}\n")

    queue = Queue()
    successes = 0
    fails = 0
    combined_output = []

    for image_path in image_files:
        print(f"Starting process for {image_path.name}")

        process = Process(target=process_image_worker, args=(str(image_path), queue))
        process.start()

        print(f"Waiting for process: {process.pid}")
        process.join(timeout=600)

        if process.is_alive():
            print(f"Process stuck, terminating: {process.pid}")
            process.terminate()
            process.join()
            fails += 1
            continue

        while not queue.empty():
            _, status, extracted_text = queue.get()

            if status == "success":
                successes += 1
                combined_output.append(extracted_text.strip())
                combined_output.append("\n\n")
            else:
                fails += 1

    final_output_path = output_folder / f"{input_folder.name}.txt"
    _write_text_file("".join(combined_output), final_output_path)

    return successes, fails, final_output_path


def image_to_text_ocr_main(input_folder, output_folder):
    print("Image OCR Started\n")

    load_dotenv_if_present()

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_folder}")

    image_folders = list_image_folders(input_folder)
    if not image_folders:
        raise FileNotFoundError(f"No image folders found in folder: {input_folder}")

    total_successes = 0
    total_fails = 0
    output_files: list[Path] = []

    for folder in image_folders:
        print(f"Processing folder: {folder.name}\n")
        successes, fails, final_output_path = process_image_folder(folder, output_folder)
        total_successes += successes
        total_fails += fails
        output_files.append(final_output_path)

    total_seconds = time.time() - start_time

    print("\n==============================")
    print("DONE")
    print("==============================")
    print(f"Success: {total_successes}")
    print(f"Fail: {total_fails}")
    print("Output files:")
    for output_file in output_files:
        print(output_file)
    print(f"Total time: {total_seconds / 60:.2f} minutes")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    image_input_folder = Path(r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\Rel 118 to 133 XMLS\docx\Rel 133\BIA\Test\output\images")
    image_output_folder = Path(r"C:\Users\tior\Documents\PROJECTS\AutoTag v1.1\Rel 118 to 133 XMLS\docx\Rel 133\BIA\Test\output\texts")

    image_to_text_ocr_main(image_input_folder, image_output_folder)