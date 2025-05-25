# -------------------------------
# file: infra/vector_store.py
# -------------------------------
"""Chroma VectorStore をラップするユーティリティ"""
import os
import chromadb
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

# 環境変数を確実に読み込み
load_dotenv()

_DEF_DIR = "./vector_db"


def get_vector_store(persist_directory: str = _DEF_DIR):
    """永続化ディレクトリを指定して VectorStore を返す。"""
    
    # OpenAI APIキーの確認とデバッグ情報
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print(f"Debug: OPENAI_API_KEY not found in environment variables")
        print(f"Debug: Available env vars: {list(os.environ.keys())}")
        raise ValueError(
            "OPENAI_API_KEY環境変数が設定されていません。"
            ".envファイルにOPENAI_API_KEY=your-key-hereを設定してください。"
        )
    
    print(f"Debug: OpenAI API Key found (length: {len(api_key)})")
    
    # 明示的にAPIキーを渡してEmbeddingsを初期化
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # LangChain 0.2 では ClientSettings 推奨、簡易のため直接渡し
    vs = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vs