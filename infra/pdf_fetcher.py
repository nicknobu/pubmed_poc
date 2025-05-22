# -------------------------------
# file: infra/pdf_fetcher.py
# -------------------------------
"""PDF をダウンロードして一時ファイルとして保存"""
from pathlib import Path
import tempfile
import requests


def fetch_pdf(url_or_path: str) -> str:
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        # URLならダウンロード
        response = requests.get(url_or_path, timeout=15)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            return tmp_file.name
    else:
        # ローカルファイルならそのまま返す
        return url_or_path