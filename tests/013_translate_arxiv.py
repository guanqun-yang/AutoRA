import os
import shutil
import tarfile
import tempfile
import requests
import subprocess
from pathlib import Path

from setting import setting
from utils.common import get_current_datetime


def download_arxiv_source(arxiv_id: str, save_dir: str) -> str:
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    print(f"Downloading {url}")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Download failed: {response.status_code}")
    archive_path = os.path.join(save_dir, f"{arxiv_id}.tar.gz")
    with open(archive_path, "wb") as f:
        f.write(response.content)
    return archive_path


def extract_archive(archive_path: str, extract_to: str):
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to)


def find_main_tex_file(base_dir: str) -> str:
    tex_files = list(Path(base_dir).rglob("*.tex"))
    candidates = []
    for tex in tex_files:
        try:
            text = tex.read_text(encoding="utf-8", errors="ignore")
            if "\\begin{document}" in text:
                score = text.count("\\section") + text.count("\\begin") + text.count("\\cite")
                candidates.append((score, tex))
        except Exception:
            continue
    if not candidates:
        raise FileNotFoundError("No candidate main .tex file found")
    return str(max(candidates, key=lambda x: x[0])[1])


def flatten_with_latexpand(output_dir, old_main_filename, new_main_filename):
    cmd = [
        "latexpand",
        "--empty-comments",
        "--verbose",
        "--output",
        new_main_filename,
        old_main_filename,
    ]
    result = subprocess.run(cmd, cwd=output_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Latexpand failed:\n", result.stderr)
        raise RuntimeError("Latexpand failed")
    else:
        print("✅ Latexpand completed")


def copy_used_files(src_dir: str, main_tex_path: str, output_dir: str):
    """
    Copy all relevant files that are referenced in the flattened main.tex
    """
    os.makedirs(output_dir, exist_ok=True)

    # Copy everything (safe fallback)
    for item in Path(src_dir).rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(src_dir)
            dest_path = Path(output_dir) / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(item, dest_path)


def normalize_arxiv_full(arxiv_id: str, output_dir: str):
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = download_arxiv_source(arxiv_id, tmp)
        extract_archive(archive_path, tmp)

        old_main_filename = find_main_tex_file(tmp)
        copy_used_files(tmp, old_main_filename, output_dir)

        new_main_filename = "{}.tex".format(get_current_datetime())
        flatten_with_latexpand(
            output_dir,
            old_main_filename,
            new_main_filename,
        )
        os.rename(Path(output_dir) / new_main_filename, Path(output_dir) / "main.tex")

OUTPUT_DIR = setting.DATASETS_PATH / "arxiv"
arxiv_id = "2310.19852"
normalize_arxiv_full(arxiv_id, str(OUTPUT_DIR / arxiv_id))