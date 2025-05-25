# -------------------------------
# file: domain/text_ops.py
# -------------------------------
"""PDF ロードとテキスト分割など純粋ロジック"""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

_CHUNK_SIZE = 1000


def is_valid_pdf(pdf_path: Path) -> bool:
    """PDFファイルが有効かどうかをチェックする"""
    try:
        with open(pdf_path, 'rb') as f:
            header = f.read(10)
            return header.startswith(b'%PDF-')
    except Exception:
        return False


def load_and_split(pdf_path: Path):
    """PDF をロードしてチャンク化した Document list を返す。"""
    
    # PDFファイルの妥当性を事前チェック
    if not is_valid_pdf(pdf_path):
        raise ValueError(f"Invalid PDF file: {pdf_path}. The file may be corrupted or not a PDF.")
    
    try:
        docs = PyPDFLoader(str(pdf_path)).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=_CHUNK_SIZE, chunk_overlap=50)
        return splitter.split_documents(docs)
    except Exception as e:
        # より詳細なエラー情報を提供
        with open(pdf_path, 'rb') as f:
            content_preview = f.read(100)
        
        if content_preview.startswith(b'<!DOCTYPE') or content_preview.startswith(b'<html'):
            raise ValueError(f"File appears to be HTML, not PDF: {pdf_path}")
        elif content_preview.startswith(b'<?xml'):
            raise ValueError(f"File appears to be XML, not PDF: {pdf_path}")
        else:
            raise ValueError(f"Failed to load PDF: {pdf_path}. Error: {str(e)}")


def debug_file_content(pdf_path: Path, preview_bytes: int = 200):
    """デバッグ用：ファイル内容の先頭を確認する"""
    try:
        with open(pdf_path, 'rb') as f:
            content = f.read(preview_bytes)
        
        print(f"File: {pdf_path}")
        print(f"Size: {pdf_path.stat().st_size} bytes")
        print(f"First {preview_bytes} bytes (hex): {content.hex()}")
        print(f"First {preview_bytes} bytes (text, errors ignored): {content.decode('utf-8', errors='ignore')}")
        
        # ファイル形式の判定
        if content.startswith(b'%PDF-'):
            print("✅ Valid PDF format")
        elif content.startswith(b'<!DOCTYPE') or content.startswith(b'<html'):
            print("❌ HTML format detected")
        elif content.startswith(b'<?xml'):
            print("❌ XML format detected")
        else:
            print("❓ Unknown format")
            
    except Exception as e:
        print(f"Error reading file: {e}")