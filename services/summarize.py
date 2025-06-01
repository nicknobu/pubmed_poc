# 緊急修正：services/summarize.py
# ベクトルストアをクリーンにして新しいセッションで実行

import tempfile
import shutil
from pathlib import Path
from infra.pdf_fetcher import fetch_pdf
from infra.vector_store import get_vector_store
from domain.text_ops import load_and_split
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
import os

def create_clean_vector_store():
    """毎回新しいベクトルストアを作成（古いデータ混入を防ぐ）"""
    # 一時ディレクトリで新しいベクトルストアを作成
    temp_dir = tempfile.mkdtemp(prefix="pubmed_vs_")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY環境変数が設定されていません")
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vs = Chroma(persist_directory=temp_dir, embedding_function=embeddings)
    
    return vs, temp_dir

def summarize_text_fixed(text: str) -> str:
    """ベクトルストア問題を修正した要約機能"""
    try:
        if len(text.strip()) < 100:
            return "テキストが短すぎて要約できません。"
        
        print("🔧 新しいベクトルストアを作成中...")
        
        # クリーンなベクトルストアを作成
        vs, temp_dir = create_clean_vector_store()
        
        try:
            # 重要セクション抽出
            important_sections = extract_key_sections(text)
            print(f"📋 抽出されたセクション: {list(important_sections.keys())}")
            
            # 改良されたチャンク分割
            docs = text_to_documents_improved(text, important_sections)
            print(f"📦 作成されたチャンク数: {len(docs)}")
            
            # 新しいベクトルストアに追加
            vs.add_documents(docs)
            print("✅ ベクトルストアに文書を追加完了")
            
            # RAG検索設定
            retriever = vs.as_retriever(
                search_type="similarity",  # まずシンプルに
                search_kwargs={"k": 12}     # 適度な数のチャンクを取得
            )
            
            # 検索テスト（デバッグ用）
            test_query = "骨粗しょう症 コンプライアンス 結果"
            test_docs = retriever.get_relevant_documents(test_query)
            print(f"🔍 検索テスト結果: {len(test_docs)} 文書取得")
            if test_docs:
                print(f"🎯 最初の文書: {test_docs[0].page_content[:100]}...")
            
            # LLM設定
            llm = ChatOpenAI(
                model_name="gpt-4o-mini", 
                temperature=0.1,
                max_tokens=2500
            )
            
            # 医学論文専用プロンプト
            medical_prompt = PromptTemplate(
                template="""
あなたは医学論文の専門要約者です。以下の論文の関連部分を読んで、正確な3点要約を日本語で作成してください。

論文の関連内容:
{context}

要約要件：
1. 【研究背景・目的】
   - 研究対象疾患の重要性と既存治療の課題（2-3文）
   - この研究の具体的な目的と仮説（1-2文）

2. 【方法・主要結果】
   - 研究デザイン、対象者数、期間、設定（1-2文）
   - 重要な結果を具体的な数値とともに詳述（3-4文）
   - 統計的有意性や信頼区間があれば含める

3. 【結論・臨床的意義】
   - 研究から得られた主要な知見（2-3文）
   - 臨床現場への具体的な影響と推奨事項（1-2文）
   - 研究の限界と今後の展望（1文）

重要事項：
- 各セクション合計150-250文字を目安に詳しく記述
- 重要な数値データ、パーセンテージ、信頼区間は必ず含める
- 医学的専門用語を適切に使用し、信頼性の高い要約とする
- 結果の臨床的意義を具体的に説明する

質問: {question}

医学論文要約:""",
                input_variables=["context", "question"]
            )
            
            # RAGチェーン構築
            chain = RetrievalQA.from_chain_type(
                llm=llm, 
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": medical_prompt}
            )
            
            question = "この医学論文について、研究背景・方法結果・結論の3点で要約してください。"
            print("🤖 LLMによる要約生成中...")
            answer = chain.run(question)
            
            return answer.strip()
            
        finally:
            # 一時ディレクトリをクリーンアップ
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("🧹 一時ファイルをクリーンアップ")
            
    except Exception as e:
        print(f"❌ 修正要約エラー: {e}")
        # フォールバック：直接LLM要約
        return direct_llm_summary_simple(text)

def text_to_documents_improved_extended(text: str, key_sections: dict) -> list[Document]:
    """より多くの情報を含むチャンク分割"""
    docs = []
    
    # 重要セクションを優先的にドキュメント化（文字数制限を緩和）
    for section_name, section_text in key_sections.items():
        if section_text.strip() and len(section_text.strip()) > 50:
            weight = {
                'abstract': 3.0,
                'conclusion': 3.0, 
                'results': 2.5,
                'methods': 2.0  # Methodsの重要度も上げる
            }.get(section_name, 1.0)
            
            # 長いセクションは分割して複数のチャンクに
            if len(section_text) > 1500:
                # 長いセクションを複数に分割
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200
                )
                sub_chunks = splitter.split_text(section_text)
                
                for i, sub_chunk in enumerate(sub_chunks):
                    doc = Document(
                        page_content=sub_chunk.strip(),
                        metadata={
                            "section": f"{section_name}_part_{i+1}",
                            "importance": weight,
                            "source": "key_section_split"
                        }
                    )
                    docs.append(doc)
            else:
                doc = Document(
                    page_content=section_text.strip(),
                    metadata={
                        "section": section_name,
                        "importance": weight,
                        "source": "key_section"
                    }
                )
                docs.append(doc)
    
    # 通常のチャンク分割も併用（チャンクサイズを増加）
    used_content = set()
    for doc in docs:
        used_content.add(doc.page_content[:100])
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # 1000 → 1500 に増加
        chunk_overlap=200,  # 100 → 200 に増加
        separators=["\n\n## ", "\n\n", "\n", "。", ". ", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    
    for i, chunk in enumerate(chunks):
        chunk_preview = chunk[:100]
        if (len(chunk.strip()) > 50 and 
            chunk_preview not in used_content and
            not any(preview in chunk_preview for preview in used_content)):
            
            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "source": "general_chunk",
                    "importance": 1.0
                }
            )
            docs.append(doc)
            used_content.add(chunk_preview)
    
    return docs

def direct_llm_summary_detailed(text: str) -> str:
    """RAGを使わない詳細LLM要約（フォールバック）"""
    try:
        llm = ChatOpenAI(
            model_name="gpt-4o-mini", 
            temperature=0.1, 
            max_tokens=2500  # 大幅に増加
        )
        
        # 重要部分を抽出
        sections = extract_key_sections(text)
        
        # より多くの内容を含める
        key_content = ""
        if sections['abstract']:
            key_content += f"Abstract: {sections['abstract'][:2000]}\n\n"
        if sections['results']:
            key_content += f"Results: {sections['results'][:2000]}\n\n"  
        if sections['conclusion']:
            key_content += f"Conclusion: {sections['conclusion'][:2000]}\n\n"
        if sections['methods']:
            key_content += f"Methods: {sections['methods'][:1500]}\n\n"
        
        # 重要部分がない場合は全文のより多くの部分を使用
        if not key_content.strip():
            key_content = text[:6000]  # 3000 → 6000 に増加
        
        prompt = f"""
以下の医学論文を読んで、詳細で有用な3点要約を日本語で作成してください：

{key_content}

要約要件：
1. 【研究背景・目的】（150-200文字程度）
   - 対象疾患の重要性と既存治療の課題を詳述
   - 本研究の具体的な目的と意義を明確に

2. 【方法・主要結果】（200-300文字程度）
   - 研究デザイン、対象者、期間を具体的に
   - 重要な結果を数値データとともに詳細に記述
   - 統計的有意性や信頼区間があれば含める

3. 【結論・臨床的意義】（150-200文字程度）
   - 研究の主要な知見と臨床への具体的影響
   - 研究の限界と今後の展望も含める

注意：
- 各点をしっかりとした分量で記述する
- 数値データは必ず含める
- 医学的に正確で詳細な表現を使用

詳細要約：
"""
        
        response = llm.invoke(prompt)
        return response.content.strip()
        
    except Exception as e:
        return f"詳細要約生成エラー: {e}"

# extract_key_sections と text_to_documents_improved 関数は前回と同じものを使用

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
        
        # セクション判定（より正確に）
        if (any(word in line_lower for word in ['abstract', 'summary']) and 
            len(line) < 50 and 
            not line_lower.startswith('background')):
            current_section = 'abstract'
            continue
        elif (any(word in line_lower for word in ['result', 'finding']) and 
              len(line) < 50):
            current_section = 'results'
            continue
        elif (any(word in line_lower for word in ['conclusion', 'discussion']) and 
              len(line) < 50):
            current_section = 'conclusion'
            continue
        elif (any(word in line_lower for word in ['method', 'material']) and 
              len(line) < 50):
            current_section = 'methods'
            continue
        elif line_lower.startswith('##') or (line_lower.startswith('#') and len(line) < 100):
            # 新しいセクションが始まった可能性
            current_section = None
            continue
            
        # 内容を該当セクションに追加
        if current_section and line.strip() and len(line.strip()) > 10:
            sections[current_section] += line + ' '
    
    return sections

def text_to_documents_improved(text: str, key_sections: dict) -> list[Document]:
    """医学論文用改良チャンク分割"""
    docs = []
    
    # 重要セクションを優先的にドキュメント化
    for section_name, section_text in key_sections.items():
        if section_text.strip() and len(section_text.strip()) > 50:
            weight = {
                'abstract': 3.0,
                'conclusion': 3.0, 
                'results': 2.5,
                'methods': 1.5
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
    
    # 通常のチャンク分割も併用（重複除去）
    used_content = set()
    for doc in docs:
        used_content.add(doc.page_content[:100])  # 重複チェック用
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n## ", "\n\n", "\n", "。", ". ", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    
    for i, chunk in enumerate(chunks):
        chunk_preview = chunk[:100]
        if (len(chunk.strip()) > 50 and 
            chunk_preview not in used_content and
            not any(preview in chunk_preview for preview in used_content)):
            
            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "source": "general_chunk",
                    "importance": 1.0
                }
            )
            docs.append(doc)
            used_content.add(chunk_preview)
    
    return docs

# 既存関数の置き換え
def summarize_text(text: str) -> str:
    """メイン要約関数(詳細版)"""
    return summarize_text_fixed(text)

def summarize_pdf(pdf_path: str) -> str:
    """PDF要約も修正版を使用"""
    docs = load_and_split(Path(pdf_path))
    full_text = "\n".join([doc.page_content for doc in docs])
    return summarize_text_fixed(full_text)