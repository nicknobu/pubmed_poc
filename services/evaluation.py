# services/evaluation.py にデバッグ機能を追加した版

import numpy as np
import openai
import os
import re
from typing import Dict, Optional

class SummaryEvaluator:
    """要約品質評価クラス（デバッグ機能付き）"""
    
    def __init__(self, debug_mode=False):
        """OpenAI APIキーの設定"""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY環境変数が設定されていません")
        
        # OpenAI クライアント初期化
        self.client = openai.OpenAI(api_key=self.api_key)
        self.debug_mode = debug_mode
    
    def evaluate_summary_quality(self, original_text: str, summary: str) -> Dict:
        """論文Abstract vs 要約の品質評価（デバッグ情報付き）"""
        try:
            if self.debug_mode:
                print(f"🔍 入力テキスト長: {len(original_text)} 文字")
                print(f"🔍 入力テキスト最初の500文字:\n{original_text[:500]}\n")
            
            # 1. Abstract抽出（デバッグ情報付き）
            abstract = self.extract_abstract_with_debug(original_text)
            
            if not abstract or len(abstract.strip()) < 50:
                return self._create_error_result("Abstractが見つからないか短すぎます")
            
            if self.debug_mode:
                print(f"✅ Abstract抽出成功: {len(abstract)} 文字")
                print(f"📄 抽出されたAbstract:\n{abstract[:300]}...\n")
            
            # 2. 要約の前処理
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
                "abstract_text": abstract[:300] + "..." if len(abstract) > 300 else abstract,
                "full_abstract": abstract,  # デバッグ用：完全なAbstract
                "summary_text": processed_summary,
                "cosine_similarity": round(cosine_similarity, 3),
                "word_overlap": round(word_overlap, 3),
                "content_coverage": round(content_coverage, 3),
                "overall_score": round(overall_score, 3),
                "pass_threshold": cosine_similarity >= 0.8,
                "quality_level": self._get_quality_level(overall_score),
                "feedback": self._generate_feedback(cosine_similarity, word_overlap, content_coverage),
                "debug_info": self._get_debug_info(original_text) if self.debug_mode else None
            }
            
        except Exception as e:
            return self._create_error_result(f"評価処理エラー: {str(e)}")
    
    def extract_abstract_with_debug(self, text: str) -> Optional[str]:
        """デバッグ情報付きAbstract抽出"""
        lines = text.split('\n')
        
        if self.debug_mode:
            print(f"🔍 テキストを{len(lines)}行に分割")
            print("🔍 最初の20行:")
            for i, line in enumerate(lines[:20]):
                print(f"  {i+1:2d}: {line[:80]}")
            print()
        
        # Method 1: 構造化Abstract
        abstract_sections = self._extract_structured_abstract_debug(lines)
        if abstract_sections:
            if self.debug_mode:
                print("✅ 構造化Abstract抽出成功")
            return abstract_sections
        
        if self.debug_mode:
            print("❌ 構造化Abstract抽出失敗")
        
        # Method 2: 従来型Abstract
        abstract_content = self._extract_traditional_abstract_debug(lines)
        if abstract_content:
            if self.debug_mode:
                print("✅ 従来型Abstract抽出成功")
            return abstract_content
        
        if self.debug_mode:
            print("❌ 従来型Abstract抽出失敗")
        
        # Method 3: 冒頭部分推定
        beginning_content = self._extract_from_beginning_debug(lines)
        if beginning_content:
            if self.debug_mode:
                print("✅ 冒頭部分推定成功")
            return beginning_content
        
        if self.debug_mode:
            print("❌ 全ての抽出方法が失敗")
        
        return None
    
    def _extract_structured_abstract_debug(self, lines: list) -> Optional[str]:
        """構造化Abstract抽出（デバッグ版）"""
        abstract_parts = []
        current_section = None
        in_abstract_area = False
        found_sections = []
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            line_lower = line_clean.lower()
            
            # Abstract領域の開始を検出
            if ('abstract' in line_lower and len(line_clean) < 50 and 
                not line_lower.startswith('background')):
                in_abstract_area = True
                if self.debug_mode:
                    print(f"🎯 Abstract開始検出: 行{i+1}")
                continue
            
            # Background以降のセクション検出
            if in_abstract_area or any(section in line_lower for section in ['background', 'objective', 'purpose']):
                in_abstract_area = True
                
                # セクション見出しの検出
                if line_lower in ['background', 'methods', 'results', 'conclusion', 'conclusions']:
                    current_section = line_lower
                    found_sections.append(line_lower)
                    if self.debug_mode:
                        print(f"📍 セクション検出: {line_lower} (行{i+1})")
                    continue
                elif line_lower.startswith('background'):
                    current_section = 'background'
                    found_sections.append('background')
                    if self.debug_mode:
                        print(f"📍 Background開始: 行{i+1}")
                    # 見出し行に内容が含まれている場合
                    if len(line_clean) > 15:
                        content = line_clean.split(' ', 1)[1] if ' ' in line_clean else line_clean
                        if content and len(content) > 10:
                            abstract_parts.append(content)
                            if self.debug_mode:
                                print(f"  内容追加: {content[:50]}...")
                    continue
                elif line_lower.startswith('method'):
                    current_section = 'methods'
                    found_sections.append('methods')
                    if self.debug_mode:
                        print(f"📍 Methods開始: 行{i+1}")
                    continue
                elif line_lower.startswith('result'):
                    current_section = 'results'
                    found_sections.append('results')
                    if self.debug_mode:
                        print(f"📍 Results開始: 行{i+1}")
                    continue
                elif line_lower.startswith('conclusion'):
                    current_section = 'conclusion'
                    found_sections.append('conclusion')
                    if self.debug_mode:
                        print(f"📍 Conclusion開始: 行{i+1}")
                    continue
                
                # Abstract終了の判定
                if (line_lower in ['keywords', 'keyword'] or 
                    line_lower.startswith('keywords:') or
                    line_lower.startswith('keyword:') or
                    any(keyword in line_lower for keyword in ['introduction', '## introduction', 'citation'])):
                    if self.debug_mode:
                        print(f"🛑 Abstract終了検出: 行{i+1} ({line_lower})")
                    break
                
                # 内容の追加
                if current_section and line_clean and len(line_clean) > 10:
                    # 箇条書きや番号を除去
                    cleaned_line = re.sub(r'^[•·-]\s*', '', line_clean)
                    cleaned_line = re.sub(r'^\d+[\.)]\s*', '', cleaned_line)
                    
                    if (cleaned_line and 
                        not cleaned_line.lower().startswith(('abstract', 'background', 'method', 'result', 'conclusion')) and
                        len(cleaned_line) > 5):
                        abstract_parts.append(cleaned_line)
                        if self.debug_mode:
                            print(f"  {current_section}内容: {cleaned_line[:50]}...")
        
        if self.debug_mode:
            print(f"🔍 発見されたセクション: {found_sections}")
            print(f"🔍 抽出された部分数: {len(abstract_parts)}")
            if abstract_parts:
                print(f"🔍 統合長: {len(' '.join(abstract_parts))} 文字")
        
        # 十分な内容が抽出された場合のみ返す
        if abstract_parts and len(' '.join(abstract_parts)) > 200:
            return ' '.join(abstract_parts)
        
        return None
    
    def _extract_traditional_abstract_debug(self, lines: list) -> Optional[str]:
        """従来型Abstract抽出（デバッグ版）"""
        abstract_lines = []
        in_abstract = False
        abstract_start_line = -1
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            line_lower = line_clean.lower()
            
            # Abstract開始の判定
            if ('abstract' in line_lower and len(line_clean) < 50):
                in_abstract = True
                abstract_start_line = i + 1
                if self.debug_mode:
                    print(f"📍 従来型Abstract開始: 行{i+1}")
                continue
            
            if in_abstract:
                # Abstract終了の判定
                if (any(keyword in line_lower for keyword in 
                       ['introduction', 'keywords', '## ', 'citation', 'references']) and 
                    len(line_clean) < 100):
                    if self.debug_mode:
                        print(f"🛑 従来型Abstract終了: 行{i+1}")
                    break
                
                # 空行は無視
                if not line_clean:
                    continue
                
                # 有効な内容行を追加
                if len(line_clean) > 10:
                    abstract_lines.append(line_clean)
                    if self.debug_mode:
                        print(f"  内容追加: {line_clean[:50]}...")
        
        if self.debug_mode:
            print(f"🔍 従来型抽出結果: {len(abstract_lines)}行, {len(' '.join(abstract_lines))}文字")
        
        if abstract_lines and len(' '.join(abstract_lines)) > 100:
            return ' '.join(abstract_lines)
        
        return None
    
    def _extract_from_beginning_debug(self, lines: list) -> Optional[str]:
        """冒頭部分推定（デバッグ版）"""
        early_lines = []
        
        for i, line in enumerate(lines[:30]):
            line_clean = line.strip()
            
            # タイトルやメタデータをスキップ
            if (len(line_clean) > 30 and 
                not line_clean.lower().startswith(('title:', 'author', 'doi:', 'pmid:'))):
                early_lines.append(line_clean)
                if self.debug_mode:
                    print(f"  冒頭部分追加: {line_clean[:50]}...")
                
                # 適度な長さで切る
                if len(' '.join(early_lines)) > 800:
                    break
        
        if self.debug_mode:
            print(f"🔍 冒頭部分推定: {len(early_lines)}行, {len(' '.join(early_lines))}文字")
        
        if early_lines and len(' '.join(early_lines)) > 200:
            return ' '.join(early_lines)
        
        return None
    
    def _get_debug_info(self, text: str) -> Dict:
        """デバッグ情報の収集"""
        lines = text.split('\n')
        return {
            "total_lines": len(lines),
            "total_chars": len(text),
            "first_20_lines": lines[:20],
            "lines_with_abstract": [i for i, line in enumerate(lines) if 'abstract' in line.lower()],
            "lines_with_background": [i for i, line in enumerate(lines) if 'background' in line.lower()],
            "lines_with_methods": [i for i, line in enumerate(lines) if 'method' in line.lower()],
            "lines_with_results": [i for i, line in enumerate(lines) if 'result' in line.lower()],
            "lines_with_conclusion": [i for i, line in enumerate(lines) if 'conclusion' in line.lower()],
        }
    
    # その他のメソッドは前回と同じ
    def preprocess_summary(self, summary: str) -> str:
        """3点要約を統合してテキスト化"""
        cleaned = re.sub(r'^[0-9]+[.．)]?\s*', '', summary, flags=re.MULTILINE)
        cleaned = re.sub(r'^[・●○]\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'【[^】]*】', '', cleaned)
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()
    
    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """OpenAI Embeddingsを使用したコサイン類似度計算"""
        try:
            text1_clean = text1[:8000]
            text2_clean = text2[:8000]
            
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
            
            vec1 = np.array(response1.data[0].embedding)
            vec2 = np.array(response2.data[0].embedding)
            
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
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0
    
    def calculate_content_coverage(self, abstract: str, summary: str) -> float:
        important_patterns = [
            r'\b\d+(?:\.\d+)?%',
            r'\bp\s*[<>=]\s*0\.\d+',
            r'\b\d+(?:\.\d+)?\s*(?:months?|years?|days?)',
            r'\b\d+(?:\.\d+)?\s*(?:mg|g|ml|patients?|cases?)',
            r'\b(?:significant|effective|improvement|increase|decrease)\b',
        ]
        
        abstract_matches = set()
        summary_matches = set()
        
        for pattern in important_patterns:
            abstract_matches.update(re.findall(pattern, abstract.lower()))
            summary_matches.update(re.findall(pattern, summary.lower()))
        
        if len(abstract_matches) == 0:
            return 0.5
        
        coverage = len(summary_matches.intersection(abstract_matches)) / len(abstract_matches)
        return min(coverage, 1.0)
    
    def _calculate_overall_score(self, cosine_sim: float, word_overlap: float, coverage: float) -> float:
        weights = {'cosine': 0.6, 'overlap': 0.25, 'coverage': 0.15}
        return (cosine_sim * weights['cosine'] + 
                word_overlap * weights['overlap'] + 
                coverage * weights['coverage'])
    
    def _get_quality_level(self, score: float) -> str:
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

# デバッグ用ラッパー関数
def evaluate_summary_debug(original_text: str, summary: str) -> Dict:
    """デバッグモード付き要約品質評価"""
    evaluator = SummaryEvaluator(debug_mode=True)
    return evaluator.evaluate_summary_quality(original_text, summary)

def evaluate_summary(original_text: str, summary: str) -> Dict:
    """通常の要約品質評価"""
    evaluator = SummaryEvaluator(debug_mode=False)
    return evaluator.evaluate_summary_quality(original_text, summary)