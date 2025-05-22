# -------------------------------
# file: domain/text_ops.py
# -------------------------------
"""PDF ロードとテキスト分割など純粋ロジック"""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

_CHUNK_SIZE = 1000


def load_and_split(pdf_path: Path):
    """PDF をロードしてチャンク化した Document list を返す。"""
    docs = PyPDFLoader(str(pdf_path)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=_CHUNK_SIZE, chunk_overlap=50)
    return splitter.split_documents(docs)