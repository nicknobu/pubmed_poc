# -------------------------------
# file: infra/vector_store.py
# -------------------------------
"""Chroma VectorStore をラップするユーティリティ"""
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings


_DEF_DIR = "./vector_db"


def get_vector_store(persist_directory: str = _DEF_DIR):
    """永続化ディレクトリを指定して VectorStore を返す。"""
    embeddings = OpenAIEmbeddings()
    # LangChain 0.2 では ClientSettings 推奨、簡易のため直接渡し
    vs = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vs
