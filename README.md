# 🧬 PubMed 医学論文要約システム

**AI駆動開発による医学論文の3点要約PoC**

[![Demo](https://img.shields.io/badge/🚀_Live_Demo-HuggingFace-yellow)](https://huggingface.co/spaces/nicknobu/pubmed-summary-demo)
[![Code](https://img.shields.io/badge/📂_Source_Code-GitHub-blue)](https://github.com/nicknobu/pubmed_poc)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-red)](https://streamlit.io)

## 🎯 プロジェクト概要

医学論文（PubMed）を対象とした**AI協働開発**による要約システムです。RAG（Retrieval-Augmented Generation）アーキテクチャを採用し、英語Abstract→日本語要約の高品質な変換を実現します。

### ✨ 主要機能

- 📄 **多形式対応**: PDF、HTML、XML から本文抽出
- 🔍 **柔軟な入力**: PMCID、URL に対応
- 🤖 **RAG要約**: LangChain + OpenAI による文脈理解要約
- 📊 **品質評価**: 英日間翻訳に最適化された自動評価システム
- 🌐 **Webアプリ**: Streamlit による直感的なUI

## 🏗️ 技術アーキテクチャ

### **Core Stack**
```
Frontend:  Streamlit Web UI
Backend:   LangChain + OpenAI GPT-4o-mini
Vector DB: Chroma (RAG検索)
Embedding: OpenAI text-embedding-3-small
```

### **AI協働開発手法**
- **ペアコーディング**: ChatGPT-4・Claude-4 との効果的協働
- **プロンプトエンジニアリング**: 医学論文専用プロンプト設計
- **RAG最適化**: 医学文書向けチャンク分割・検索戦略

## 🚀 セットアップ & 実行

### 1. 環境構築
```bash
# リポジトリクローン
git clone https://github.com/nicknobu/pubmed_poc.git
cd pubmed_poc

# 依存関係インストール
pip install -r requirements.txt
```

### 2. 環境変数設定
```bash
# .env ファイル作成
echo "OPENAI_API_KEY=your-openai-api-key" > .env
```

### 3. アプリケーション起動
```bash
# Streamlit Web UI 起動
streamlit run app.py
```

## 📋 使用方法

### **入力例**
```bash
# PMCID形式
PMC12085841

# PDF URL
https://example.com/paper.pdf
```

### **出力例**
```
【研究背景・目的】
骨粗鬆症治療において服薬遵守率の低下が課題となっている中、
本研究では実臨床における遵守率改善要因を24ヶ月間の観察で検討...

【方法・主要結果】
後ろ向きコホート研究として842名を対象に解析。遵守群では
非遵守群比で治療継続率が54%向上（p<0.001）...

【結論・臨床的意義】
定期的な患者フォローアップが遵守率改善の鍵となることが示され、
臨床現場での実践的介入指針を提供する重要な知見である...
```

## 🔬 技術仕様詳細

### **RAG パイプライン**
```python
# 1. 文書取得・前処理
content = extract_pmc_content(pmcid)
sections = extract_key_sections(content)

# 2. チャンク分割（重要度重み付き）
docs = text_to_documents_improved(content, sections)

# 3. ベクトル化・検索
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_type="mmr", k=12)

# 4. 文脈付き要約生成
chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    retriever=retriever,
    chain_type="stuff",
    prompt=medical_prompt_template
)
```

### **多言語品質評価**
- **コサイン類似度**: OpenAI Embeddings による意味的類似度
- **概念一致率**: 医学専門用語の英日対応辞書活用  
- **データ保持率**: 数値・統計情報の翻訳精度評価
- **動的基準**: 英日間（≥0.50）vs 同言語（≥0.80）で閾値調整

## 📊 品質評価システム

### **多言語対応基準**
| 言語ペア | 合格基準 | 優秀基準 | 重み配分 |
|----------|----------|----------|----------|
| 🌐 英→日 | ≥0.50 | ≥0.65 | 概念重視 |
| 🔤 同言語 | ≥0.80 | ≥0.85 | 類似度重視 |

### **評価指標**
```python
overall_score = (
    cosine_similarity * 0.45 +    # 意味的類似度
    word_overlap * 0.15 +         # 概念一致率  
    content_coverage * 0.40       # データ保持率
)
```

## 🧪 テスト・デバッグ

### **単体テスト実行**
```bash
pytest tests/test_summarize.py -v
```

### **デバッグ機能**
- 📊 RAG検索内容確認
- 🔍 Abstract抽出プロセス可視化
- 📈 品質評価詳細ログ

## 📁 プロジェクト構造

```
pubmed_poc/
├── app.py                    # Streamlit Web UI
├── services/
│   ├── summarize.py         # RAG要約エンジン
│   └── evaluation.py        # 多言語品質評価
├── domain/
│   └── text_ops.py          # PDF処理・テキスト分割
├── infra/
│   ├── pdf_fetcher.py       # PDF取得
│   └── vector_store.py      # Chroma VectorStore
├── tests/
│   └── test_summarize.py    # 単体テスト
├── requirements.txt         # 依存関係
└── .env.example            # 環境変数テンプレート
```

## 🌟 開発実績・学習成果

### **AI駆動開発スキル**
- **ChatGPT-4/Claude-4**: 効果的なペアコーディング手法習得
- **プロンプトエンジニアリング**: 医学論文専用プロンプト設計
- **コード生成・最適化**: AI支援による開発効率向上

### **技術スタック習得**
- **Python**: LangChain, Streamlit, OpenAI API
- **RAG実装**: ベクトル検索, チャンク分割戦略
- **品質評価**: 多言語対応評価システム設計
- **Web開発**: Streamlit による直感的UI構築

### **開発環境・ツール**
- **VSCode**: AI拡張機能活用
- **Git/GitHub**: バージョン管理・コード公開
- **HuggingFace**: Spaces でのデプロイ・デモ公開

## 🚀 Live Demo

**🌐 HuggingFace Spaces**: [pubmed-summary-demo](https://huggingface.co/spaces/nicknobu/pubmed-summary-demo)

推奨テスト用PMCID:
- `PMC12085841` - 英日高品質要約例
- `PMC5334499` - PDF対応例  
- `PMC8790252` - XML対応例

## 🔗 関連リンク

- 📂 **GitHub Repository**: https://github.com/nicknobu/pubmed_poc
- 🚀 **Live Demo**: https://huggingface.co/spaces/nicknobu/pubmed-summary-demo
- 📚 **PubMed Central**: https://www.ncbi.nlm.nih.gov/pmc/

## 📝 ライセンス

MIT License - 個人学習・研究目的での開発プロジェクト

---

**Note**: これは個人的な学習・研究目的で開発されたプロジェクトです。医療判断には使用せず、研究参考資料としてご利用ください。