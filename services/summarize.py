# -------------------------------
# file: services/summarize.py
# -------------------------------
"""ユースケース層: PDF/テキスト → 要約までのオーケストレーション"""
from pathlib import Path
from infra.pdf_fetcher import fetch_pdf
from infra.vector_store import get_vector_store
from domain.text_ops import load_and_split
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def summarize_pdf(pdf_path: str) -> str:
    """PDF ファイルパスから 3 点要約を返す。"""
    docs = load_and_split(Path(pdf_path))

    vs = get_vector_store()
    vs.add_documents(docs)

    retriever = vs.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    question = """この医学論文について、以下の3点で正確に要約してください：

1. 【研究目的・背景】：なぜこの研究を行ったか、解決したい問題は何か
2. 【方法・主要結果】：どのような手法で、どんな重要な結果が得られたか  
3. 【結論・臨床的意義】：この研究から何が分かり、医療現場にどう活かせるか

各点は1-2文で簡潔に、重要な数値があれば含めて日本語で回答してください。"""
    answer: str = chain.run(question)
    return answer.strip()

def summarize_text(text: str) -> str:
    """テキストから直接 3 点要約を返す（PDF以外用）"""
    try:
        if len(text.strip()) < 100:
            return "テキストが短すぎて要約できません。"
        
        # テキストをLangChainのDocumentオブジェクトに変換
        docs = text_to_documents(text)
        
        # ベクトルストアに追加
        vs = get_vector_store()
        vs.add_documents(docs)

        # RAGチェーンで要約生成
        retriever = vs.as_retriever(search_kwargs={"k": 4})
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        question = "論文を 3 点で要約し、日本語で回答してください。"
        answer: str = chain.run(question)
        return answer.strip()
        
    except Exception as e:
        raise Exception(f"テキスト要約処理でエラーが発生しました: {str(e)}")

def text_to_documents(text: str) -> list[Document]:
    """テキストをLangChainのDocumentオブジェクトのリストに変換する"""
    # RecursiveCharacterTextSplitterでチャンク化
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", ".", " ", ""]
    )
    
    # テキストを分割
    chunks = splitter.split_text(text)
    
    # Documentオブジェクトのリストとして返す
    docs = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "chunk_id": i,
                "source": "extracted_text",
                "chunk_size": len(chunk)
            }
        )
        docs.append(doc)
    
    return docs

def summarize_from_url(url: str) -> str:
    """PubMed PDF URL から 3 点要約を返す（後方互換性のため残存）。"""
    pdf_path: Path = fetch_pdf(url)
    return summarize_pdf(str(pdf_path))