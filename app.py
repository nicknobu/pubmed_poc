# -------------------------------
# file: app.py (PMC HTML/XML本文抽出機能付き)
# -------------------------------
"""Streamlit UI ― URL/DOI を入力すると 3 点要約を表示（PDF + HTML対応）"""
import os
import re
import tempfile
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import streamlit as st
from services.summarize import summarize_pdf, summarize_text
from dotenv import load_dotenv
from services.evaluation import evaluate_summary
load_dotenv()

import openai  # 必要に応じて先頭で import

# .env 読み込み済みの前提
openai_key = os.environ.get("OPENAI_API_KEY")

if not openai_key:
    raise ValueError("環境変数 OPENAI_API_KEY が見つかりません。Hugging Face の Secrets または .env を確認してください。")

openai.api_key = openai_key

def extract_pmc_html_content(pmcid):
    """PMC記事のHTMLページから本文を抽出する"""
    try:
        # PMCIDを正規化
        if not pmcid.upper().startswith('PMC'):
            pmcid = f"PMC{pmcid}"
        
        pmc_number = pmcid.replace('PMC', '')
        html_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_number}/"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(html_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # PMC記事の本文構造を解析
        content_parts = []
        
        # タイトル
        title_elem = soup.find('h1', class_='content-title') or soup.find('title')
        if title_elem:
            content_parts.append(f"Title: {title_elem.get_text(strip=True)}\n")
        
        # 著者情報
        authors_elem = soup.find('div', class_='contrib-group')
        if authors_elem:
            authors = [a.get_text(strip=True) for a in authors_elem.find_all('a')]
            if authors:
                content_parts.append(f"Authors: {', '.join(authors)}\n")
        
        # アブストラクト
        abstract_elem = soup.find('div', {'class': 'abstract'}) or soup.find('div', id='abstract')
        if abstract_elem:
            abstract_text = abstract_elem.get_text(separator=' ', strip=True)
            content_parts.append(f"Abstract:\n{abstract_text}\n\n")
        
        # メイン本文 - 複数のセレクタを試行
        main_content = None
        
        # PMCの一般的な本文セレクタ
        content_selectors = [
            'div.article-content',
            'div.fulltext-view',
            'div.tsec',
            'article',
            'div.content',
            'main'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                # 最大の要素を選択
                main_content = max(elements, key=lambda x: len(x.get_text()))
                break
        
        if main_content:
            # 不要な要素を除去
            for unwanted in main_content.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                unwanted.decompose()
            
            # セクションごとに整理
            sections = main_content.find_all(['h1', 'h2', 'h3', 'p', 'div'])
            
            for section in sections:
                text = section.get_text(separator=' ', strip=True)
                if text and len(text) > 20:  # 短すぎるテキストは除外
                    # ヘッダータグの場合
                    if section.name in ['h1', 'h2', 'h3']:
                        content_parts.append(f"\n## {text}\n")
                    else:
                        content_parts.append(f"{text}\n")
        
        # 結論・まとめ
        conclusion_elem = soup.find('div', class_='sec') or soup.find('section', id='conclusion')
        if conclusion_elem and 'conclusion' in conclusion_elem.get_text().lower():
            conclusion_text = conclusion_elem.get_text(separator=' ', strip=True)
            content_parts.append(f"\nConclusion:\n{conclusion_text}\n")
        
        full_text = '\n'.join(content_parts)
        
        # 最小限の文字数チェック
        if len(full_text.strip()) < 200:
            st.warning("抽出されたテキストが短すぎます。PDF版の利用を検討してください。")
            return None
        
        return full_text.strip()
        
    except Exception as e:
        st.error(f"PMC HTML抽出エラー: {e}")
        return None

def extract_pmc_xml_content(pmcid):
    """PMC記事のXMLから本文を抽出する（更に高精度）"""
    try:
        # PMCIDを正規化
        if not pmcid.upper().startswith('PMC'):
            pmcid = f"PMC{pmcid}"
        
        # PMC OA APIを使用してXMLを取得
        xml_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            'db': 'pmc',
            'id': pmcid,
            'retmode': 'xml',
            'rettype': 'full'
        }
        
        response = requests.get(xml_url, params=params, timeout=15)
        response.raise_for_status()
        
        # XMLをパース
        root = ET.fromstring(response.text)
        
        content_parts = []
        
        # タイトル
        title_elem = root.find('.//article-title')
        if title_elem is not None:
            content_parts.append(f"Title: {title_elem.text}\n")
        
        # アブストラクト
        abstract_elem = root.find('.//abstract')
        if abstract_elem is not None:
            abstract_text = ' '.join(abstract_elem.itertext())
            content_parts.append(f"Abstract:\n{abstract_text}\n\n")
        
        # 本文セクション
        body = root.find('.//body')
        if body is not None:
            for sec in body.findall('.//sec'):
                # セクションタイトル
                title_elem = sec.find('title')
                if title_elem is not None:
                    content_parts.append(f"\n## {title_elem.text}\n")
                
                # セクション内容
                for p in sec.findall('.//p'):
                    p_text = ' '.join(p.itertext())
                    if p_text.strip():
                        content_parts.append(f"{p_text.strip()}\n")
        
        full_text = '\n'.join(content_parts)
        
        if len(full_text.strip()) < 200:
            st.warning("XMLから十分なテキストを抽出できませんでした。")
            return None
        
        return full_text.strip()
        
    except Exception as e:
        st.warning(f"PMC XML抽出試行失敗: {e}")
        return None

def check_pmc_pdf_availability(pmcid):
    """PMC OA Web Service APIを使ってPDFが利用可能か確認する"""
    try:
        if not pmcid.upper().startswith('PMC'):
            pmcid = f"PMC{pmcid}"
        
        api_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
        params = {'id': pmcid}
        
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        
        # XMLをパース
        root = ET.fromstring(response.text)
        
        # PDFリンクを探す
        pdf_links = []
        for record in root.findall('.//record'):
            for link in record.findall('link[@format="pdf"]'):
                href = link.get('href')
                if href:
                    pdf_links.append(href)
        
        if pdf_links:
            return True, pdf_links[0]
        else:
            return False, None
    
    except Exception as e:
        st.warning(f"PMC PDF確認でエラー: {e}")
        return None, None

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

def is_pdf_content(response):
    """レスポンスがPDFかどうかを判定する"""
    content_type = response.headers.get('content-type', '').lower()
    if 'pdf' in content_type:
        return True
    
    content_start = response.content[:10]
    return content_start.startswith(b'%PDF-')

def get_pdf_from_url(url):
    """URLからPDFを取得する"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        if is_pdf_content(response):
            return response.content
        else:
            return None
            
    except Exception as e:
        st.error(f"PDF取得エラー: {e}")
        return None

def get_content_from_pmcid(pmcid):
    """PMCIDから最適な形式でコンテンツを取得する（PDF優先、HTML/XMLフォールバック）"""
    if not pmcid.upper().startswith('PMC'):
        pmcid = f"PMC{pmcid}"
    
    st.info(f"📄 {pmcid} のコンテンツを取得中...")
    
    # 1. PDF存在確認
    has_pdf, ftp_pdf_url = check_pmc_pdf_availability(pmcid)
    
    if has_pdf:
        st.info("✅ PDF版が利用可能です。PDF版を取得します...")
        pdf_content = get_pdf_from_url(ftp_pdf_url)
        if pdf_content and pdf_content.startswith(b'%PDF-'):
            return 'pdf', pdf_content
        st.warning("PDF取得に失敗。テキスト版にフォールバックします...")
    
    # 2. XML版を試行
    st.info("📄 XML版から本文を抽出中...")
    xml_text = extract_pmc_xml_content(pmcid)
    if xml_text:
        st.success("✅ XML版から本文抽出成功！")
        return 'text', xml_text
    
    # 3. HTML版を試行
    st.info("🌐 HTML版から本文を抽出中...")
    html_text = extract_pmc_html_content(pmcid)
    if html_text:
        st.success("✅ HTML版から本文抽出成功！")
        return 'text', html_text
    
    # 4. すべて失敗
    st.error(f"❌ {pmcid} からコンテンツを取得できませんでした。")
    return None, None

def get_pdf_from_pmid(pmid):
    """PMIDからPMCIDを取得してコンテンツを取得する"""
    try:
        api_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
        params = {
            'dbfrom': 'pubmed',
            'db': 'pmc',
            'id': pmid,
            'retmode': 'xml'
        }
        
        response = requests.get(api_url, params=params, timeout=10)
        
        pmc_match = re.search(r'<Id>(\d+)</Id>', response.text)
        if pmc_match:
            pmc_id = pmc_match.group(1)
            pmcid = f"PMC{pmc_id}"
            st.info(f"Found PMCID: {pmcid}")
            return get_content_from_pmcid(pmcid)
        else:
            st.warning("このPMIDに対応するPMC記事が見つかりませんでした。")
            return None, None
        
    except Exception as e:
        st.error(f"PMID解決エラー: {e}")
        return None, None

def resolve_to_content(input_text):
    """入力テキストを解析してコンテンツを取得する"""
    input_text = input_text.strip()
    
    # 既にURLの場合
    if input_text.startswith(('http://', 'https://')):
        st.info(f"URL detected: {input_text}")
        pdf_content = get_pdf_from_url(input_text)
        if pdf_content:
            return 'pdf', pdf_content
        return None, None
    
    # PMCIDの場合
    if is_pmcid(input_text):
        st.info(f"PMCID detected: {input_text}")
        return get_content_from_pmcid(input_text)
    
    # PMIDの場合
    if is_pmid(input_text):
        st.info(f"PMID detected: {input_text}")
        return get_pdf_from_pmid(input_text)
    
    # DOIの場合（簡略化）
    if is_doi(input_text):
        st.info(f"DOI detected: {input_text}")
        st.warning("DOI対応は簡略実装。直接PMCIDまたはPMIDの使用を推奨します。")
        return None, None
    
    # ローカルファイルパスの場合
    if os.path.exists(input_text):
        with open(input_text, 'rb') as f:
            content = f.read()
            if content.startswith(b'%PDF-'):
                return 'pdf', content
            else:
                return 'text', content.decode('utf-8', errors='ignore')
    
    st.error("入力形式が認識できません。PMCIDまたはPMIDを推奨します。")
    return None, None

# Streamlit UI
st.set_page_config(page_title="PubMed Summarizer", page_icon="🧬")
st.title("🧬 PubMed 3 点要約 PoC (PDF + HTML対応)")

# 機能説明
st.success("🆕 **新機能**: PDF未提供の記事もHTML/XMLから本文抽出して要約可能！")

st.markdown("""
**対応する入力形式 & 取得方法：**
- 📋 **PMCID**: `PMC1234567` → PDF優先、HTML/XMLフォールバック
- 📄 **PDF URL**: 直接PDF取得
""")

input_text = st.text_input("PMCIDまたはPDFのURLを入力してください", placeholder="例: PMC12085841 または https://example.com/sample.pdf")


if st.button("要約"):
    if not input_text:
        st.error("識別子またはURLを入力してください")
    else:
        with st.spinner("コンテンツを取得中…"):
            content_type, content = resolve_to_content(input_text)
            
            if not content:
                st.error("コンテンツの取得に失敗しました。")
                st.stop()
        
        with st.spinner("要約生成中…"):
            try:
                if content_type == 'pdf':
                    # PDFファイルの妥当性をチェック
                    if not content.startswith(b'%PDF-'):
                        st.error("取得したファイルがPDF形式ではありません。")
                        st.stop()

                    # 一時ファイルに保存してPDF要約
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(content)
                        tmp_pdf_path = tmp_file.name

                    result = summarize_pdf(tmp_pdf_path)
                    os.unlink(tmp_pdf_path)
                    
                    # PDF要約では品質評価をスキップ
                    st.success("📋 要約結果：")
                    st.write(result)
                    st.info("💡 PDF形式のため品質評価は実行されません")
                
                elif content_type == 'text':
                    # テキスト直接要約
                    result = summarize_text(content)
                    
                    st.success("📋 要約結果：")
                    st.write(result)
                    
                    # 品質評価実行
                    with st.spinner("要約品質を評価中…"):
                        try:
                            evaluation_result = evaluate_summary(content, result)
                            
                            if evaluation_result["success"]:
                                st.markdown("---")
                                st.subheader("📊 要約品質評価")
                                
                                # 言語ペア情報を表示
                                is_multilingual = evaluation_result.get("is_multilingual", False)
                                evaluation_note = evaluation_result.get("evaluation_note", "標準評価")
                                
                                if is_multilingual:
                                    st.info(f"🌐 {evaluation_note} - 英語Abstract→日本語要約の専用基準を適用")
                                else:
                                    st.info(f"🔤 {evaluation_note}")
                                
                                # メイン指標表示
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    cosine_score = evaluation_result["cosine_similarity"]
                                    # 多言語基準での色分け
                                    if is_multilingual:
                                        cosine_color = "🟢" if cosine_score >= 0.65 else "🟡" if cosine_score >= 0.55 else "🔴"
                                    else:
                                        cosine_color = "🟢" if cosine_score >= 0.8 else "🟡" if cosine_score >= 0.6 else "🔴"
                                    
                                    st.metric(
                                        label="コサイン類似度", 
                                        value=f"{cosine_score:.3f}",
                                        help="Abstract vs 要約の意味的類似度（英日間は0.5+で合格）" if is_multilingual else "Abstract vs 要約の意味的類似度"
                                    )
                                    st.write(f"{cosine_color} {cosine_score:.3f}")
                                
                                with col2:
                                    overlap_score = evaluation_result["word_overlap"]
                                    # 多言語基準での色分け
                                    if is_multilingual:
                                        overlap_color = "🟢" if overlap_score >= 0.20 else "🟡" if overlap_score >= 0.15 else "🔴"
                                    else:
                                        overlap_color = "🟢" if overlap_score >= 0.4 else "🟡" if overlap_score >= 0.2 else "🔴"
                                    
                                    st.metric(
                                        label="単語重複率", 
                                        value=f"{overlap_score:.3f}",
                                        help="医学概念の翻訳一致率（英日間は0.15+で良好）" if is_multilingual else "共通する重要単語の割合"
                                    )
                                    st.write(f"{overlap_color} {overlap_score:.3f}")
                                
                                with col3:
                                    coverage_score = evaluation_result["content_coverage"]
                                    coverage_color = "🟢" if coverage_score >= 0.5 else "🟡" if coverage_score >= 0.3 else "🔴"
                                    st.metric(
                                        label="重要概念カバー率", 
                                        value=f"{coverage_score:.3f}",
                                        help="数値データ・重要概念の包含率"
                                    )
                                    st.write(f"{coverage_color} {coverage_score:.3f}")
                                
                                with col4:
                                    overall_score = evaluation_result["overall_score"]
                                    # 多言語基準での色分け
                                    if is_multilingual:
                                        overall_color = "🟢" if overall_score >= 0.65 else "🟡" if overall_score >= 0.55 else "🔴"
                                    else:
                                        overall_color = "🟢" if overall_score >= 0.8 else "🟡" if overall_score >= 0.6 else "🔴"
                                    
                                    st.metric(
                                        label="総合スコア", 
                                        value=f"{overall_score:.3f}",
                                        help="英日間調整済み総合評価" if is_multilingual else "全指標の重み付き平均"
                                    )
                                    st.write(f"{overall_color} {overall_score:.3f}")
                                
                                # 合格判定（修正版）
                                st.markdown("### 🎯 品質判定")
                                if evaluation_result["pass_threshold"]:
                                    if is_multilingual:
                                        st.success(f"✅ **高品質要約** (英日間類似度 {cosine_score:.3f} ≥ 0.50)")
                                    else:
                                        st.success(f"✅ **高品質要約** (類似度 {cosine_score:.3f} ≥ 0.80)")
                                    st.success(f"🏆 品質レベル: **{evaluation_result['quality_level']}**")
                                else:
                                    if is_multilingual:
                                        st.warning(f"⚠️ **要改善** (英日間類似度 {cosine_score:.3f} < 0.50)")
                                    else:
                                        st.warning(f"⚠️ **要改善** (類似度 {cosine_score:.3f} < 0.80)")
                                    st.info(f"📈 品質レベル: **{evaluation_result['quality_level']}**")
                                
                                # 改善提案
                                if evaluation_result["feedback"]:
                                    st.markdown("### 💡 改善提案")
                                    st.info(evaluation_result["feedback"])
                                
                                # 詳細情報（展開可能）
                                with st.expander("🔍 評価詳細情報"):
                                    st.markdown("**抽出されたAbstract:**")
                                    
                                    # 正しいAbstractを表示
                                    if "full_abstract" in evaluation_result:
                                        actual_abstract = evaluation_result["full_abstract"]
                                    else:
                                        actual_abstract = evaluation_result.get("abstract_text", "抽出失敗")
                                    
                                    st.text_area(
                                        "Abstract内容", 
                                        actual_abstract, 
                                        height=150,
                                        disabled=True
                                    )
                                    st.write(f"**文字数**: {len(actual_abstract)} 文字")
                                    
                                    st.markdown("**処理された要約:**")
                                    st.text_area(
                                        "要約内容", 
                                        evaluation_result.get("summary_text", result), 
                                        height=100,
                                        disabled=True
                                    )
                                    
                                    st.markdown("**評価アルゴリズム:**")
                                    if is_multilingual:
                                        st.markdown("""
                                        - **コサイン類似度**: OpenAI text-embedding-3-small使用（英日間基準）
                                        - **単語重複率**: 医学概念辞書による翻訳一致評価
                                        - **概念カバー率**: 英語数値→日本語数値の保持率評価
                                        - **総合スコア**: 英日間重み付き平均 (0.45:0.15:0.40)
                                        - **判定基準**: コサイン類似度≥0.50で合格、≥0.65で優秀
                                        """)
                                    else:
                                        st.markdown("""
                                        - **コサイン類似度**: OpenAI text-embedding-3-small使用
                                        - **単語重複率**: Jaccard係数による共通語彙計測
                                        - **概念カバー率**: 数値データ・重要概念の保持率
                                        - **総合スコア**: 重み付き平均 (0.6:0.25:0.15)
                                        - **判定基準**: コサイン類似度≥0.80で合格、≥0.85で優秀
                                        """)
                                    
                                    # 成功のお祝いメッセージ
                                    if evaluation_result["pass_threshold"]:
                                        st.balloons()
                                        if is_multilingual:
                                            st.success("🎉 英日間翻訳として高品質な要約を達成しました！LinkedIn公開レベルです！")
                                        else:
                                            st.success("🎉 LinkedIn公開レベルの高品質要約を達成しました！")
                            
                            else:
                                st.error(f"❌ 品質評価エラー: {evaluation_result['error']}")
                                st.info("💡 品質評価なしでも要約は正常に生成されています")
                        
                        except Exception as eval_error:
                            st.error(f"品質評価処理でエラーが発生: {eval_error}")
                            st.info("💡 品質評価なしでも要約は正常に生成されています")
                            
                            # エラー詳細をデバッグ情報として表示
                            import traceback
                            with st.expander("🔧 エラー詳細（デバッグ用）"):
                                st.text(traceback.format_exc())
                
                # 取得方法を表示（try文内に移動）
                st.markdown("---")
                st.info(f"📊 取得方法: {content_type.upper()}形式")
                    
            except Exception as e:
                import traceback
                st.error(f"要約処理でエラーが発生しました: {e}")
                st.text(traceback.format_exc())
# デバッグ機能追加（要約品質改善用）
st.markdown("---")
st.subheader("🔧 デバッグ機能")

if st.button("🔍 RAG取得内容を確認"):
    if not input_text:
        st.error("まず識別子またはURLを入力してください")
    else:
        with st.spinner("デバッグ情報を取得中..."):
            try:
                # コンテンツ取得
                content_type, content = resolve_to_content(input_text)
                
                if not content:
                    st.error("コンテンツの取得に失敗しました。")
                    st.stop()
                
                if content_type == 'text':
                    # RAGの動作を詳しく確認
                    from services.summarize import text_to_documents_improved, extract_key_sections
                    from infra.vector_store import get_vector_store
                    
                    st.write("### 📄 取得したコンテンツの概要")
                    st.write(f"文字数: {len(content):,} 文字")
                    st.write(f"最初の200文字: {content[:200]}...")
                    
                    st.write("### 🎯 重要セクション抽出結果")
                    key_sections = extract_key_sections(content)
                    
                    for section_name, section_text in key_sections.items():
                        if section_text.strip():
                            st.write(f"**{section_name.upper()}セクション** ({len(section_text)}文字):")
                            st.write(f"{section_text[:300]}...")
                            st.write("---")
                    
                    st.write("### 🔍 RAGチャンク分割結果")
                    docs = text_to_documents_improved(content, key_sections)
                    st.write(f"総チャンク数: {len(docs)}")
                    
                    # 重要度順にソート
                    docs_sorted = sorted(docs, key=lambda x: x.metadata.get('importance', 1.0), reverse=True)
                    
                    for i, doc in enumerate(docs_sorted[:5]):
                        importance = doc.metadata.get('importance', 1.0)
                        section = doc.metadata.get('section', 'general')
                        st.write(f"**チャンク {i+1}** (重要度: {importance}, セクション: {section}):")
                        st.write(f"{doc.page_content[:400]}...")
                        st.write("---")
                    
                    st.write("### 🔎 RAG検索結果")
                    vs = get_vector_store()
                    vs.add_documents(docs)
                    
                    # 複数のクエリで検索テスト
                    test_queries = [
                        "研究の目的と背景",
                        "主要な結果と効果",
                        "結論と臨床的意義",
                        "治療効果と安全性"
                    ]
                    
                    retriever = vs.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7}
                    )
                    
                    for query in test_queries:
                        st.write(f"**クエリ: '{query}'**")
                        retrieved_docs = retriever.get_relevant_documents(query)
                        
                        for j, rdoc in enumerate(retrieved_docs[:2]):
                            st.write(f"取得文書 {j+1}: {rdoc.page_content[:200]}...")
                        st.write("---")
                    
                else:
                    st.info("PDF形式のため、テキスト抽出後にデバッグ可能です")
                    
            except Exception as e:
                import traceback
                st.error(f"デバッグ処理でエラー: {e}")
                st.text(traceback.format_exc())
# 既存のデバッグコード（RAG確認）の後に追加
if st.button("🔍 Abstract抽出デバッグ"):
    if not input_text:
        st.error("まず識別子またはURLを入力してください")
    else:
        with st.spinner("デバッグ情報を取得中..."):
            try:
                # コンテンツ取得
                content_type, content = resolve_to_content(input_text)
                
                if not content:
                    st.error("コンテンツの取得に失敗しました。")
                    st.stop()
                
                if content_type == 'text':
                    # デバッグ用評価実行
                    from services.evaluation import evaluate_summary_debug
                    
                    st.write("### 📊 入力テキスト基本情報")
                    st.write(f"**文字数**: {len(content):,} 文字")
                    st.write(f"**行数**: {len(content.split(chr(10))):,} 行")
                    
                    # 最初の1000文字を表示
                    st.write("### 📄 入力テキスト（最初の1000文字）")
                    st.text_area("コンテンツ", content[:1000], height=200, disabled=True)
                    
                    # 行番号付きで最初の30行を表示
                    st.write("### 📝 行別表示（最初の30行）")
                    lines = content.split('\n')
                    line_display = []
                    for i, line in enumerate(lines[:30]):
                        line_display.append(f"{i+1:2d}: {line}")
                    st.text_area("行別内容", '\n'.join(line_display), height=300, disabled=True)
                    
                    # Abstract関連キーワード検索
                    st.write("### 🔍 Abstract関連キーワード検索")
                    keywords = ['abstract', 'background', 'methods', 'results', 'conclusion', 'keywords']
                    keyword_lines = {}
                    
                    for keyword in keywords:
                        found_lines = []
                        for i, line in enumerate(lines):
                            if keyword.lower() in line.lower():
                                found_lines.append(f"行{i+1}: {line[:100]}")
                        keyword_lines[keyword] = found_lines
                    
                    for keyword, found in keyword_lines.items():
                        if found:
                            st.write(f"**'{keyword}' 発見箇所:**")
                            for line in found[:5]:  # 最初の5個まで
                                st.write(f"  {line}")
                        else:
                            st.write(f"**'{keyword}' 発見箇所:** なし")
                    
                    # ダミー要約を使ってAbstract抽出をテスト
                    dummy_summary = "テスト要約です。"
                    
                    st.write("### 🧪 Abstract抽出テスト")
                    with st.spinner("Abstract抽出実行中..."):
                        
                        # デバッグモードで評価実行（コンソール出力をキャプチャ）
                        import io
                        import sys
                        
                        # 標準出力をキャプチャ
                        old_stdout = sys.stdout
                        sys.stdout = captured_output = io.StringIO()
                        
                        try:
                            debug_result = evaluate_summary_debug(content, dummy_summary)
                            
                            # 標準出力を元に戻す
                            sys.stdout = old_stdout
                            debug_logs = captured_output.getvalue()
                            
                            # デバッグログを表示
                            if debug_logs:
                                st.write("**🔧 抽出処理ログ:**")
                                st.text_area("デバッグログ", debug_logs, height=300)
                            
                            # 抽出結果表示
                            if debug_result["success"]:
                                st.success("✅ Abstract抽出成功！")
                                
                                # 完全なAbstractを表示
                                if "full_abstract" in debug_result:
                                    st.write("**📄 抽出されたAbstract（完全版）:**")
                                    st.text_area(
                                        "Abstract内容", 
                                        debug_result["full_abstract"], 
                                        height=200,
                                        disabled=True
                                    )
                                    st.write(f"**文字数**: {len(debug_result['full_abstract'])} 文字")
                                
                                # 品質評価結果も表示
                                st.write("**📊 品質評価結果:**")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("コサイン類似度", f"{debug_result['cosine_similarity']:.3f}")
                                with col2:
                                    st.metric("単語重複率", f"{debug_result['word_overlap']:.3f}")
                                with col3:
                                    st.metric("総合スコア", f"{debug_result['overall_score']:.3f}")
                                
                            else:
                                st.error(f"❌ Abstract抽出失敗: {debug_result['error']}")
                            
                            # デバッグ情報詳細
                            if debug_result.get("debug_info"):
                                st.write("### 📈 詳細デバッグ情報")
                                debug_info = debug_result["debug_info"]
                                
                                st.write(f"**総行数**: {debug_info['total_lines']}")
                                st.write(f"**総文字数**: {debug_info['total_chars']}")
                                
                                sections_found = {
                                    "Abstract": debug_info['lines_with_abstract'],
                                    "Background": debug_info['lines_with_background'],
                                    "Methods": debug_info['lines_with_methods'],
                                    "Results": debug_info['lines_with_results'], 
                                    "Conclusion": debug_info['lines_with_conclusion']
                                }
                                
                                for section, line_numbers in sections_found.items():
                                    if line_numbers:
                                        st.write(f"**{section}キーワード発見行**: {line_numbers}")
                                    else:
                                        st.write(f"**{section}キーワード発見行**: なし")
                        
                        except Exception as eval_error:
                            # 標準出力を元に戻す
                            sys.stdout = old_stdout
                            debug_logs = captured_output.getvalue()
                            
                            st.error(f"デバッグ実行エラー: {eval_error}")
                            if debug_logs:
                                st.text_area("エラー時のログ", debug_logs, height=200)
                    
                    # 改善提案
                    st.write("### 💡 抽出改善のヒント")
                    
                    # テキスト構造の分析
                    abstract_keyword_count = len([line for line in lines if 'abstract' in line.lower()])
                    background_keyword_count = len([line for line in lines if 'background' in line.lower()])
                    
                    if abstract_keyword_count == 0 and background_keyword_count == 0:
                        st.warning("⚠️ AbstractやBackgroundキーワードが見つかりません。XMLの構造が予想と異なる可能性があります。")
                        st.info("💡 対策: 冒頭部分抽出ロジックの改良、または文書構造の手動確認が必要です。")
                    
                    elif abstract_keyword_count > 0 and background_keyword_count == 0:
                        st.info("💡 Abstractキーワードは見つかりましたが、Background以下の構造化セクションが見つかりません。")
                        st.info("💡 対策: 従来型Abstract抽出ロジックの改良が必要です。")
                    
                    elif background_keyword_count > 0:
                        st.success("✅ 構造化Abstract（Background以下）のキーワードが見つかりました。")
                        st.info("💡 構造化Abstract抽出ロジックの調整で改善できる可能性があります。")
                
                else:
                    st.info("PDF形式のため、テキスト抽出後にデバッグ可能です")
                    
            except Exception as e:
                import traceback
                st.error(f"デバッグ処理でエラー: {e}")
                st.text(traceback.format_exc())
# サイドバーに統計と使用例
with st.sidebar:
    st.header("📊 対応状況")
    st.metric("PMC総記事数", "3,454,737")
    st.metric("PDF提供", "816,971 (24%)")
    st.metric("HTML/XML対応", "ほぼ全記事")
    
    st.success("✅ HTML/XML対応により大幅に対象拡大！")
    
    # 🆕 多言語対応品質評価情報
    st.header("🎯 多言語品質評価")
    st.markdown("""
    **自動品質評価システム:**
    - Abstract vs 要約の意味的類似度
    - 医学概念の翻訳一致率  
    - 重要数値データの保持率
    
    **言語ペア別基準:**
    - 🌐 **英日間**: コサイン類似度≥0.50で合格
    - 🔤 **同言語**: コサイン類似度≥0.80で合格
    - 📊 **概念重視**: 翻訳品質で重み調整
    """)
    
    st.header("🔥 推奨使用例")
    st.markdown("""
    **PMCIDで試してみる:**
    ```
    PMC12085841  (英→日 高品質)
    PMC5334499   (PDF対応)
    PMC8790252   (XML対応)
    ```
    """)
    
    st.header("💡 取得優先順位")
    st.markdown("""
    1. **PDF版** (最高品質)
    2. **XML版** (構造化データ)
    3. **HTML版** (フォールバック)
    """)
    
    # 🆕 多言語品質スコア説明
    st.header("📈 品質スコア説明")
    st.markdown("""
    **🌐 英日間評価基準:**
    - 🟢 **優秀**: 0.65以上
    - 🟡 **良好**: 0.55-0.64  
    - 🔴 **要改善**: 0.55未満
    
    **🔤 同言語評価基準:**
    - 🟢 **優秀**: 0.85以上
    - 🟡 **良好**: 0.75-0.84  
    - 🔴 **要改善**: 0.75未満
    
    **🎯 多言語対応の特徴:**
    - 英語Abstract→日本語要約の現実的基準
    - 医学専門用語の概念レベル評価
    - 翻訳品質に応じた動的重み調整
    """)