---
title: PubMed Summary Demo
emoji: 📄
colorFrom: cyan
colorTo: indigo
sdk: streamlit
sdk_version: "1.34.0"
app_file: app.py
pinned: false
---

# PubMed 要約デモ（3 点要約）

この Hugging Face Space は、PubMed 論文の PDF を指定することで、その内容を自動的に 3 点で要約する Streamlit アプリです。

## 特徴

- PubMed Central（PMC）の論文 PDF URL を入力して要約可能
- OpenAI API を使った自然言語要約
- Streamlit によるシンプルな Web UI

## 使い方

1. 「PubMed PDF URL」欄に、PMCID 形式の**フルリンク（例：https://www.ncbi.nlm.nih.gov/pmc/articles/PMCxxxxxxx/pdf/）**を入力します。
2. 「要約」ボタンをクリックすると、3 点要約が生成されます。
3. PDF は自動でダウンロードされ、一時保存されます。

## 技術スタック

- Python
- Streamlit
- OpenAI API
- dotenv

---

# PubMed Summary Demo (3-Point Summarizer)

This Hugging Face Space is a Streamlit app that generates a 3-point summary from a PubMed article PDF.

## Features

- Accepts full PMC PDF URLs (e.g., `https://www.ncbi.nlm.nih.gov/pmc/articles/PMCxxxxxxx/pdf/`)
- Uses OpenAI API for summarization
- Clean and minimal Streamlit interface

## How to Use

1. Enter a full PubMed Central PDF URL into the input box.
2. Click the **Summarize** button.
3. The PDF is downloaded, and a 3-point summary is shown.

## Tech Stack

- Python
- Streamlit
- OpenAI API
- dotenv
