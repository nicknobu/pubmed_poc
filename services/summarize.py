# -------------------------------
# file: services/summarize.py
# -------------------------------
"""ユースケース層: PDF → 要約までのオーケストレーション"""
from pathlib import Path
from infra.pdf_fetcher import fetch_pdf
from infra.vector_store import get_vector_store
from domain.text_ops import load_and_split
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def summarize_pdf(url: str) -> str:
    """PubMed PDF URL から 3 点要約を返す。"""
    pdf_path: Path = fetch_pdf(url)
    docs = load_and_split(pdf_path)

    vs = get_vector_store()
    vs.add_documents(docs)

    retriever = vs.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    question = "論文を 3 点で要約し、日本語で回答してください。"  # noqa: E501
    answer: str = chain.run(question)
    return answer.strip()
