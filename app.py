# -------------------------------
# file: app.py
# -------------------------------
"""Streamlit UI ― URL を入力すると 3 点要約を表示"""
import os
import tempfile
import requests
import streamlit as st
from services.summarize import summarize_pdf
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PubMed Summarizer", page_icon="🧬")
st.title("🧬 PubMed 3 点要約 PoC")
url = st.text_input("PubMed PDF URL (PMCID のフルリンク)")

if st.button("要約"):  # noqa: E501
    if not url:
        st.error("URL を入力してください")
    else:
        with st.spinner("要約生成中…"):
            try:
                input_path = url.strip().strip('"').strip("'")
                if input_path.startswith("http://") or input_path.startswith("https://"):
                    headers = {"User-Agent": "Mozilla/5.0"}
                    response = requests.get(input_path, headers=headers)
                    response.raise_for_status()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(response.content)
                        tmp_pdf_path = tmp_file.name
                else:
                    # ローカルファイル存在チェック
                    if not os.path.exists(input_path):
                        raise FileNotFoundError(f"ファイルが見つかりません: {input_path}")
                    tmp_pdf_path = input_path

                result = summarize_pdf(tmp_pdf_path)
                st.success("要約結果：")
                st.write(result)
            except Exception as e:  # noqa: BLE001
                import traceback
                st.error(f"エラーが発生しました: {e}\n\n{traceback.format_exc()}")