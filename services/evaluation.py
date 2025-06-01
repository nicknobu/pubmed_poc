# 修正版: services/evaluation.py
"""論文要約の品質評価機能（文法エラー修正版）"""

import numpy as np
import openai
import os
import re
from typing import Dict, Optional

class SummaryEvaluator:
    """要約品質評価クラス"""
    
    def __init__(self):
        """OpenAI APIキーの設定"""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY環境変数が設定されていません")
        
        # OpenAI クライアント初期化
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def evaluate_summary_quality(self, original_text: str, summary: str) -> Dict:
        """
        論文Abstract vs 要約の品質評価
        
        Args:
            original_text: 論文全文
            summary: 生成された3点要約
            
        Returns:
            評価結果の辞書
        """
        try:
            # 1. Abstract抽出
            abstract = self.extract_abstract(original_text)
            
            if not abstract or len(abstract.strip()) < 50:
                return self._create_error_result("Abstractが見つからないか短すぎます")
            
            # 2. 要約の前処理（3点要約を統合）
            processed_summary = self.preprocess_summary(summary)
            
            if not processed_summary or len(processed_summary.strip()) < 30:
                return self._create_error_result("要約が短すぎるか無効です")
            
            # 3. コサイン類似度計算
            cosine_similarity = self.calculate_cosine_similarity(abstract, processed_summary)
            
            # 4. 追加の品質指標
            word_overlap = self.calculate_word_overlap(abstract, processed_summary)
            content_coverage = self.calculate_content_coverage(abstract, processed_summary)
            
            # 5. 総合評価
            overall_score = self._calculate_overall_score(
                cosine_similarity, word_overlap, content_coverage
            )
            
            return {
                "success": True,
                "abstract_text": abstract[:200] + "..." if len(abstract) > 200 else abstract,
                "summary_text": processed_summary,
                "cosine_similarity": round(cosine_similarity, 3),
                "word_overlap": round(word_overlap, 3),
                "content_coverage": round(content_coverage, 3),
                "overall_score": round(overall_score, 3),
                "pass_threshold": cosine_similarity >= 0.8,
                "quality_level": self._get_quality_level(overall_score),
                "feedback": self._generate_feedback(cosine_similarity, word_overlap, content_coverage)
            }
            
        except Exception as e:
            return self._create_error_result(f"評価処理エラー: {str(e)}")
    
    def extract_abstract(self, text: str) -> Optional[str]:
        """論文からAbstractを抽出（構造化Abstract対応）"""
        lines = text.split('\n')
        
        # Method 1: 構造化Abstract (Background → Methods → Results → Conclusion)
        abstract_sections = self._extract_structured_abstract(lines)
        if abstract_sections:
            return abstract_sections
        
        # Method 2: 従来のAbstract抽出
        abstract_content = self._extract_traditional_abstract(lines)
        if abstract_content:
            return abstract_content
        
        # Method 3: 冒頭部分からの推定
        return self._extract_from_beginning(lines)
    
    def _extract_structured_abstract(self, lines: list) -> Optional[str]:
        """構造化Abstract (Background/Methods/Results/Conclusion) を抽出"""
        abstract_parts = []
        current_section = None
        in_abstract_area = False
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            line_lower = line_clean.lower()
            
            # Abstract領域の開始を検出
            if ('abstract' in line_lower and len(line_clean) < 50 and 
                not line_lower.startswith('background')):
                in_abstract_area = True
                continue
            
            # Background以降のセクション検出
            if in_abstract_area or any(section in line_lower for section in ['background', 'objective', 'purpose']):
                in_abstract_area = True
                
                # セクション見出しの検出
                if line_lower in ['background', 'methods', 'results', 'conclusion', 'conclusions']:
                    current_section = line_lower
                    continue
                elif (line_lower.startswith('background') or line_lower.startswith('objective') or 
                      line_lower.startswith('purpose')):
                    current_section = 'background'
                    # 見出し行に内容が含まれている場合は処理
                    if len(line_clean) > len(line_lower.split()[0]) + 5:
                        content = line_clean.split(' ', 1)[1] if ' ' in line_clean else ''
                        if content:
                            abstract_parts.append(content)
                    continue
                elif line_lower.startswith('method'):
                    current_section = 'methods'
                    continue
                elif line_lower.startswith('result'):
                    current_section = 'results'
                    continue
                elif line_lower.startswith('conclusion'):
                    current_section = 'conclusion'
                    continue
                
                # Abstract終了の判定
                if (line_lower in ['keywords', 'keyword'] or 
                    line_lower.startswith('keywords:') or
                    line_lower.startswith('keyword:') or
                    any(keyword in line_lower for keyword in ['introduction', '## introduction', 'citation'])):
                    break
                
                # 内容の追加
                if current_section and line_clean and len(line_clean) > 10:
                    # 箇条書きや番号を除去
                    cleaned_line = re.sub(r'^[•·-]\s*', '', line_clean)
                    cleaned_line = re.sub(r'^\d+[\.)]\s*', '', cleaned_line)
                    
                    if cleaned_line and not cleaned_line.lower().startswith(('abstract', 'background', 'method', 'result', 'conclusion')):
                        abstract_parts.append(cleaned_line)
        
        # 十分な内容が抽出された場合のみ返す
        if abstract_parts and len(' '.join(abstract_parts)) > 200:
            return ' '.join(abstract_parts)
        
        return None
    
    def _extract_traditional_abstract(self, lines: list) -> Optional[str]:
        """従来型のAbstract抽出"""
        abstract_lines = []
        in_abstract = False
        
        for line in lines:
            line_clean = line.strip()
            line_lower = line_clean.lower()
            
            # Abstract開始の判定
            if ('abstract' in line_lower and len(line_clean) < 50):
                in_abstract = True
                continue
            
            # Abstract終了の判定
            if in_abstract:
                if (any(keyword in line_lower for keyword in 
                       ['introduction', 'keywords', '## ', 'citation', 'references']) and 
                    len(line_clean) < 100):
                    break
                
                # 空行は無視
                if not line_clean:
                    continue
                
                # 有効な内容行を追加
                if len(line_clean) > 10:
                    abstract_lines.append(line_clean)
        
        if abstract_lines and len(' '.join(abstract_lines)) > 100:
            return ' '.join(abstract_lines)
        
        return None
    
    def _extract_from_beginning(self, lines: list) -> Optional[str]:
        """冒頭部分からAbstractを推定"""
        early_lines = []
        
        for line in lines[:30]:  # 最初の30行をチェック
            line_clean = line.strip()
            
            # タイトルやメタデータをスキップ
            if (len(line_clean) > 30 and 
                not line_clean.lower().startswith(('title:', 'author', 'doi:', 'pmid:'))):
                early_lines.append(line_clean)
                
                # 適度な長さで切る
                if len(' '.join(early_lines)) > 800:
                    break
        
        if early_lines and len(' '.join(early_lines)) > 200:
            return ' '.join(early_lines)
        
        return None
    
    def preprocess_summary(self, summary: str) -> str:
        """3点要約を統合してテキスト化"""
        # 番号や記号を除去
        cleaned = re.sub(r'^[0-9]+[.．)]?\s*', '', summary, flags=re.MULTILINE)
        cleaned = re.sub(r'^[・●○]\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'【[^】]*】', '', cleaned)  # 【】内を除去
        
        # 改行を統合
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """OpenAI Embeddingsを使用したコサイン類似度計算"""
        try:
            # テキストの前処理
            text1_clean = text1[:8000]  # トークン制限考慮
            text2_clean = text2[:8000]
            
            # Embeddings生成
            response1 = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text1_clean,
                encoding_format="float"
            )
            
            response2 = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text2_clean,
                encoding_format="float"
            )
            
            # ベクトル取得
            vec1 = np.array(response1.data[0].embedding)
            vec2 = np.array(response2.data[0].embedding)
            
            # コサイン類似度計算
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_sim = dot_product / (norm1 * norm2)
            
            return float(cosine_sim)
            
        except Exception as e:
            print(f"コサイン類似度計算エラー: {e}")
            return 0.0
    
    def calculate_word_overlap(self, text1: str, text2: str) -> float:
        """単語重複率計算"""
        # 単語抽出（英語・日本語対応）
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        # Jaccard係数
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_content_coverage(self, abstract: str, summary: str) -> float:
        """重要概念のカバー率計算"""
        # 医学論文で重要なキーワードを抽出
        important_patterns = [
            r'\b\d+(?:\.\d+)?%',  # パーセンテージ
            r'\bp\s*[<>=]\s*0\.\d+',  # p値
            r'\b\d+(?:\.\d+)?\s*(?:months?|years?|days?)',  # 期間
            r'\b\d+(?:\.\d+)?\s*(?:mg|g|ml|patients?|cases?)',  # 数量・対象
            r'\b(?:significant|effective|improvement|increase|decrease)\b',  # 効果表現
        ]
        
        abstract_matches = set()
        summary_matches = set()
        
        for pattern in important_patterns:
            abstract_matches.update(re.findall(pattern, abstract.lower()))
            summary_matches.update(re.findall(pattern, summary.lower()))
        
        if len(abstract_matches) == 0:
            return 0.5  # デフォルト値
        
        coverage = len(summary_matches.intersection(abstract_matches)) / len(abstract_matches)
        return min(coverage, 1.0)
    
    def _calculate_overall_score(self, cosine_sim: float, word_overlap: float, coverage: float) -> float:
        """総合スコア計算"""
        # 重み付き平均
        weights = {
            'cosine': 0.6,    # コサイン類似度を最重視
            'overlap': 0.25,  # 単語重複
            'coverage': 0.15  # 重要概念カバー率
        }
        
        overall = (cosine_sim * weights['cosine'] + 
                  word_overlap * weights['overlap'] + 
                  coverage * weights['coverage'])
        
        return overall
    
    def _get_quality_level(self, score: float) -> str:
        """品質レベル判定"""
        if score >= 0.85:
            return "優秀"
        elif score >= 0.75:
            return "良好"
        elif score >= 0.65:
            return "標準"
        elif score >= 0.50:
            return "要改善"
        else:
            return "不十分"
    
    def _generate_feedback(self, cosine_sim: float, word_overlap: float, coverage: float) -> str:
        """改善提案生成"""
        feedback = []
        
        if cosine_sim < 0.7:
            feedback.append("要約内容がAbstractと大きく異なっています")
        elif cosine_sim >= 0.8:
            feedback.append("要約内容がAbstractと良く一致しています")
        
        if word_overlap < 0.3:
            feedback.append("重要な用語が不足している可能性があります")
        
        if coverage < 0.3:
            feedback.append("数値データや重要概念の記載が不足しています")
        elif coverage >= 0.6:
            feedback.append("重要な数値データが適切に含まれています")
        
        return "。".join(feedback) if feedback else "要約品質は標準的です"
    
    def _create_error_result(self, error_message: str) -> Dict:
        """エラー結果作成"""
        return {
            "success": False,
            "error": error_message,
            "cosine_similarity": 0.0,
            "word_overlap": 0.0,
            "content_coverage": 0.0,
            "overall_score": 0.0,
            "pass_threshold": False,
            "quality_level": "エラー",
            "feedback": error_message
        }

# 使いやすいラッパー関数
def evaluate_summary(original_text: str, summary: str) -> Dict:
    """要約品質評価のメイン関数"""
    evaluator = SummaryEvaluator()
    return evaluator.evaluate_summary_quality(original_text, summary)