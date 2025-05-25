# -------------------------------
# file: app.py
# -------------------------------
"""Streamlit UI ― URL/DOI を入力すると 3 点要約を表示"""
import os
import re
import tempfile
import requests
import streamlit as st
from services.summarize import summarize_pdf
from dotenv import load_dotenv

load_dotenv()

def is_doi(text):
    """DOI形式かどうかを判定する"""
    doi_pattern = r'^10\.\d+/.+'
    return bool(re.match(doi_pattern, text.strip()))

def is_pmcid(text):
    """PMCID形式かどうかを判定する"""
    pmcid_pattern = r'^PMC\d+$'
    return bool(re.match(pmcid_pattern, text.strip().upper()))

def is_pmid(text):
    """PMID（数字のみ）かどうかを判定する"""
    return text.strip().isdigit()

def get_pdf_url_from_doi(doi):
    """DOIからPDF URLを取得する"""
    try:
        # DOIリゾルバーを使用して論文ページを取得
        doi_url = f"https://doi.org/{doi}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(doi_url, headers=headers, allow_redirects=True)
        
        # PubMed Centralかどうかチェック
        if "ncbi.nlm.nih.gov/pmc" in response.url:
            # PMC記事の場合、PDF URLを構築
            pmc_match = re.search(r'PMC(\d+)', response.url)
            if pmc_match:
                pmc_id = pmc_match.group(1)
                pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"
                return pdf_url
        
        # その他のパブリッシャーの場合の処理を追加可能
        # 例：Nature, Science, etc.
        
        return None
    except Exception:
        return None

def get_pdf_url_from_pmcid(pmcid):
    """PMCIDからPDF URLを取得する"""
    # PMCIDを正規化（PMCプレフィックスを確保）
    if not pmcid.upper().startswith('PMC'):
        pmcid = f"PMC{pmcid}"
    
    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
    return pdf_url

def get_pdf_url_from_pmid(pmid):
    """PMIDからPMCIDを取得してPDF URLを構築する"""
    try:
        # PubMed APIを使用してPMCIDを取得
        api_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
        params = {
            'dbfrom': 'pubmed',
            'db': 'pmc',
            'id': pmid,
            'retmode': 'xml'
        }
        
        response = requests.get(api_url, params=params)
        
        # XMLからPMCIDを抽出
        pmc_match = re.search(r'<Id>(\d+)</Id>', response.text)
        if pmc_match:
            pmc_id = pmc_match.group(1)
            pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"
            return pdf_url
        
        return None
    except Exception:
        return None

def resolve_to_pdf_url(input_text):
    """入力テキストを解析してPDF URLを取得する"""
    input_text = input_text.strip()
    
    # 既にURLの場合
    if input_text.startswith(('http://', 'https://')):
        return input_text
    
    # DOIの場合
    if is_doi(input_text):
        st.info(f"DOI detected: {input_text}")
        pdf_url = get_pdf_url_from_doi(input_text)
        if pdf_url:
            st.success(f"PDF URL found: {pdf_url}")
            return pdf_url
        else:
            st.warning("DOIからPDF URLを取得できませんでした。オープンアクセスでない可能性があります。")
            return None
    
    # PMCIDの場合
    if is_pmcid(input_text):
        st.info(f"PMCID detected: {input_text}")
        pdf_url = get_pdf_url_from_pmcid(input_text)
        st.success(f"PDF URL constructed: {pdf_url}")
        return pdf_url
    
    # PMIDの場合
    if is_pmid(input_text):
        st.info(f"PMID detected: {input_text}")
        pdf_url = get_pdf_url_from_pmid(input_text)
        if pdf_url:
            st.success(f"PDF URL found: {pdf_url}")
            return pdf_url
        else:
            st.warning("PMIDからPMC記事が見つかりませんでした。オープンアクセスでない可能性があります。")
            return None
    
    # ローカルファイルパスの場合
    if os.path.exists(input_text):
        return input_text
    
    st.error("入力形式が認識できません。URL、DOI、PMCID、またはPMIDを入力してください。")
    return None

# Streamlit UI
st.set_page_config(page_title="PubMed Summarizer", page_icon="🧬")
st.title("🧬 PubMed 3 点要約 PoC")

# 入力例を表示
st.markdown("""
**対応する入力形式：**
- 📄 **PDF URL**: `https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/pdf/`
- 🔗 **DOI**: `10.1038/nature12345`
- 📋 **PMCID**: `PMC1234567` または `1234567`
- 🔢 **PMID**: `12345678`
- 💾 **ローカルファイル**: `/path/to/file.pdf`
""")

input_text = st.text_input("論文の識別子またはURL")

if st.button("要約"):
    if not input_text:
        st.error("識別子またはURLを入力してください")
    else:
        with st.spinner("PDF URLを解決中…"):
            pdf_url = resolve_to_pdf_url(input_text)
            
            if not pdf_url:
                st.stop()
        
        with st.spinner("要約生成中…"):
            try:
                if pdf_url.startswith(('http://', 'https://')):
                    # URLからPDFをダウンロード
                    headers = {"User-Agent": "Mozilla/5.0"}
                    response = requests.get(pdf_url, headers=headers)
                    response.raise_for_status()
                    
                    # レスポンスがPDFかどうかチェック
                    content_type = response.headers.get('content-type', '')
                    if 'pdf' not in content_type.lower():
                        st.warning(f"PDFファイルではない可能性があります (Content-Type: {content_type})")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(response.content)
                        tmp_pdf_path = tmp_file.name
                else:
                    # ローカルファイル
                    tmp_pdf_path = pdf_url

                result = summarize_pdf(tmp_pdf_path)
                st.success("要約結果：")
                st.write(result)
                
                # 一時ファイルをクリーンアップ
                if pdf_url.startswith(('http://', 'https://')) and os.path.exists(tmp_pdf_path):
                    os.unlink(tmp_pdf_path)
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    st.error("PDFが見つかりません。オープンアクセス論文でない可能性があります。")
                else:
                    st.error(f"PDFの取得に失敗しました: {e}")
            except Exception as e:
                import traceback
                st.error(f"エラーが発生しました: {e}\n\n{traceback.format_exc()}")

# サイドバーに使用例を表示
with st.sidebar:
    st.header("使用例")
    st.markdown("""
    **DOI例:**
    ```
    10.1038/nature12345
    10.1016/j.cell.2023.01.001
    ```
    
    **PMCID例:**
    ```
    PMC1234567
    1234567
    ```
    
    **PMID例:**
    ```
    12345678
    ```
    
    **PDF URL例:**
    ```
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/pdf/
    ```
    """)
    
    st.info("💡 オープンアクセス論文のみPDF取得可能です")