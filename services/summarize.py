# 緊急品質改善 - services/summarize.py の改良版

from pathlib import Path
from infra.pdf_fetcher import fetch_pdf
from infra.vector_store import get_vector_store
from domain.text_ops import load_and_split
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate

def summarize_text_improved(text: str) -> str:
    """大幅改良された要約機能"""
    try:
        if len(text.strip()) < 100:
            return "テキストが短すぎて要約できません。"
        
        # 重要：AbstractとConclusionを優先的に抽出
        important_sections = extract_key_sections(text)
        
        # 改良されたチャンク分割
        docs = text_to_documents_improved(text, important_sections)
        
        # ベクトルストア構築
        vs = get_vector_store()
        vs.add_documents(docs)

        # 検索戦略改善：より多くの関連文書を取得
        retriever = vs.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance で多様性確保
            search_kwargs={
                "k": 10,          # より多くのチャンクを取得
                "fetch_k": 20,    # 候補をより多く取得
                "lambda_mult": 0.7  # 関連性と多様性のバランス
            }
        )

        # 医学論文専用LLM設定
        llm = ChatOpenAI(
            model_name="gpt-4o-mini", 
            temperature=0.05,  # より一貫性のある出力
            max_tokens=1200   # 十分な要約長を確保
        )

        # 医学論文専用プロンプトテンプレート
        medical_prompt = PromptTemplate(
            template="""
あなたは医学論文の専門要約者です。提供された論文内容から、以下の3点構成で正確な要約を作成してください。

論文の関連部分:
{context}

要約要件：
1. 【研究背景・目的】
   - この研究がなぜ重要か、解決したい医学的課題は何か
   - 対象疾患の特徴や既存治療の課題
   
2. 【方法・主要結果】  
   - 研究デザインと対象（症例数、期間等）
   - 主要な治療法や介入内容
   - 重要な有効性・安全性データ（数値含む）
   
3. 【結論・臨床的意義】
   - この研究から得られた重要な知見
   - 臨床現場への影響や今後の方向性

注意事項：
- 各点は2-3文で簡潔に
- 重要な数値データは必ず含める
- 文献検索手順ではなく研究内容を要約する
- 医学的に正確な表現を使用

質問: {question}

医学論文要約:""",
            input_variables=["context", "question"]
        )

        # RAGチェーン構築
        chain = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=retriever,
            chain_type="stuff",  # より多くの文脈を含める
            chain_type_kwargs={"prompt": medical_prompt}
        )

        question = "上記の医学論文について、研究背景・方法結果・結論の3点で要約してください。"
        answer = chain.run(question)
        
        # 品質チェック
        if is_poor_quality_summary(answer, text):
            # フォールバック：直接LLM要約
            return direct_llm_summary(text)
        
        return answer.strip()
        
    except Exception as e:
        print(f"改良要約エラー: {e}")
        # フォールバック処理
        return direct_llm_summary(text)

def extract_key_sections(text: str) -> dict:
    """重要セクション（Abstract, Results, Conclusion）を抽出"""
    sections = {
        'abstract': '',
        'results': '',
        'conclusion': '',
        'methods': ''
    }
    
    lines = text.split('\n')
    current_section = None
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # セクション判定
        if any(word in line_lower for word in ['abstract', 'summary']) and len(line) < 50:
            current_section = 'abstract'
            continue
        elif any(word in line_lower for word in ['result', 'finding']) and len(line) < 50:
            current_section = 'results'
            continue
        elif any(word in line_lower for word in ['conclusion', 'discussion']) and len(line) < 50:
            current_section = 'conclusion'
            continue
        elif any(word in line_lower for word in ['method', 'material']) and len(line) < 50:
            current_section = 'methods'
            continue
        elif line_lower.startswith('##') or line_lower.startswith('#'):
            # 新しいセクションが始まった
            current_section = None
            continue
            
        # 内容を該当セクションに追加
        if current_section and line.strip():
            sections[current_section] += line + ' '
    
    return sections

def text_to_documents_improved(text: str, key_sections: dict) -> list[Document]:
    """医学論文用改良チャンク分割"""
    docs = []
    
    # 重要セクションを優先的にドキュメント化（重み付け）
    for section_name, section_text in key_sections.items():
        if section_text.strip():
            # 重要セクションには高い重みを付与
            weight = {
                'abstract': 3.0,
                'conclusion': 3.0, 
                'results': 2.5,
                'methods': 1.0
            }.get(section_name, 1.0)
            
            doc = Document(
                page_content=section_text.strip(),
                metadata={
                    "section": section_name,
                    "importance": weight,
                    "source": "key_section"
                }
            )
            docs.append(doc)
    
    # 通常のチャンク分割も併用
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n## ", "\n\n", "\n", "。", ". ", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) > 50:  # 短すぎるチャンクは除外
            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "source": "general_chunk",
                    "importance": 1.0
                }
            )
            docs.append(doc)
    
    return docs

def is_poor_quality_summary(summary: str, original_text: str) -> bool:
    """要約品質の簡易チェック"""
    summary_lower = summary.lower()
    
    # 悪い要約の特徴
    bad_indicators = [
        'システマティックレビュー', 'スクリーニング', '論文が選定',
        '文献検索', 'データベース', '検索され', '調査者',
        '重複を除いた', 'フルテキスト評価'
    ]
    
    # これらのキーワードが多く含まれていたら品質が悪い
    bad_score = sum(1 for indicator in bad_indicators if indicator in summary_lower)
    
    # 重要な医学的要素が含まれているかチェック
    good_indicators = [
        '治療', '患者', '効果', '安全性', '症例', '結果',
        '改善', '有効性', '臨床', '疾患', '研究'
    ]
    
    good_score = sum(1 for indicator in good_indicators if indicator in summary_lower)
    
    return bad_score >= 3 and good_score <= 2

def direct_llm_summary(text: str) -> str:
    """RAGを使わない直接LLM要約（フォールバック）"""
    try:
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1, max_tokens=1000)
        
        prompt = f"""
以下の医学論文を読んで、3点で要約してください：

1. 【研究背景・目的】：なぜこの研究が重要か、解決したい医学的課題
2. 【方法・主要結果】：研究デザイン、対象、重要な結果（数値含む）
3. 【結論・臨床的意義】：得られた知見と臨床への影響

注意：文献検索手順ではなく、研究の医学的内容を要約してください。

論文内容：
{text[:4000]}  # トークン制限考慮

要約：
"""
        
        response = llm.invoke(prompt)
        return response.content.strip()
        
    except Exception as e:
        return f"要約生成エラー: {e}"

# 既存の関数も改良版に置き換え
def summarize_text(text: str) -> str:
    """既存関数の置き換え"""
    return summarize_text_improved(text)

def summarize_pdf(pdf_path: str) -> str:
    """PDF要約も改良版を使用"""
    docs = load_and_split(Path(pdf_path))
    
    # PDFからテキストを結合
    full_text = "\n".join([doc.page_content for doc in docs])
    
    return summarize_text_improved(full_text)