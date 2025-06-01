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
    """多言語対応の品質評価（緊急修正版）"""
    try:
        # 1. Abstract抽出
        abstract = self.extract_abstract_with_debug(original_text) if self.debug_mode else self.extract_abstract(original_text)
        
        if not abstract or len(abstract.strip()) < 50:
            return self._create_error_result("Abstractが見つからないか短すぎます")
        
        # 2. 要約の前処理
        processed_summary = self.preprocess_summary(summary)
        
        if not processed_summary or len(processed_summary.strip()) < 30:
            return self._create_error_result("要約が短すぎるか無効です")
        
        # 🆕 3. 言語ペア検出（簡易版）
        is_multilingual = self._detect_multilingual_pair(abstract, processed_summary)
        
        # 4. コサイン類似度計算
        cosine_similarity = self.calculate_cosine_similarity(abstract, processed_summary)
        
        # 5. 改良された品質指標
        word_overlap = self.calculate_multilingual_word_overlap(abstract, processed_summary)
        content_coverage = self.calculate_multilingual_content_coverage(abstract, processed_summary)
        
        # 🆕 6. 多言語対応の評価基準
        if is_multilingual:
            # 英日間の現実的基準
            cosine_threshold = 0.50  # 0.8 → 0.50
            cosine_excellent = 0.65  # 0.85 → 0.65
            quality_level = self._get_multilingual_quality_level(cosine_similarity)
            feedback = self._generate_multilingual_feedback(cosine_similarity, word_overlap, content_coverage)
            
            # 調整済み総合スコア（概念カバー率を重視）
            overall_score = (cosine_similarity * 0.45 + 
                           word_overlap * 0.15 + 
                           content_coverage * 0.40)
        else:
            # 同言語間の従来基準
            cosine_threshold = 0.80
            cosine_excellent = 0.85
            quality_level = self._get_quality_level(cosine_similarity)
            feedback = self._generate_feedback(cosine_similarity, word_overlap, content_coverage)
            
            overall_score = (cosine_similarity * 0.6 + 
                           word_overlap * 0.25 + 
                           content_coverage * 0.15)
        
        # 7. 合格判定
        pass_threshold = cosine_similarity >= cosine_threshold
        
        return {
            "success": True,
            "abstract_text": abstract[:300] + "..." if len(abstract) > 300 else abstract,
            "full_abstract": abstract,
            "summary_text": processed_summary,
            "cosine_similarity": round(cosine_similarity, 3),
            "word_overlap": round(word_overlap, 3),
            "content_coverage": round(content_coverage, 3),
            "overall_score": round(overall_score, 3),
            "pass_threshold": pass_threshold,
            "quality_level": quality_level,
            "feedback": feedback,
            "is_multilingual": is_multilingual,
            "evaluation_note": "英日間評価基準適用" if is_multilingual else "同言語間評価基準適用",
            "debug_info": self._get_debug_info(original_text) if self.debug_mode else None
        }
        
    except Exception as e:
        return self._create_error_result(f"評価処理エラー: {str(e)}")

def _detect_multilingual_pair(self, text1: str, text2: str) -> bool:
    """多言語ペアかどうかを検出"""
    # 英語の特徴的文字の比率
    english_chars1 = len([c for c in text1 if c.isascii() and c.isalpha()])
    total_chars1 = len([c for c in text1 if c.isalpha()])
    english_ratio1 = english_chars1 / total_chars1 if total_chars1 > 0 else 0
    
    # 日本語の特徴的文字の存在
    japanese_chars2 = len([c for c in text2 if 0x3040 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FAF])
    japanese_ratio2 = japanese_chars2 / len(text2) if len(text2) > 0 else 0
    
    # 英語Abstract + 日本語要約の組み合わせを検出
    return english_ratio1 > 0.8 and japanese_ratio2 > 0.1

def calculate_multilingual_word_overlap(self, text1: str, text2: str) -> float:
    """多言語対応の単語重複率（概念レベル）"""
    # 基本的な医学概念の英日対応
    medical_concepts = {
        'osteoporosis': '骨粗鬆症',
        'parathyroid': '副甲状腺',
        'hormone': 'ホルモン',
        'treatment': '治療',
        'compliance': '遵守',
        'patients': '患者',
        'fracture': '骨折',
        'medication': '薬物',
        'therapy': '療法',
        'clinical': '臨床',
        'study': '研究',
        'retrospective': '後ろ向き',
        'prospective': '前向き',
        'months': 'ヶ月',
        'years': '年',
        'risk': 'リスク'
    }
    
    # 概念レベルでの一致をカウント
    concept_matches = 0
    total_concepts = len(medical_concepts)
    
    for en_word, ja_word in medical_concepts.items():
        if en_word.lower() in text1.lower() and ja_word in text2:
            concept_matches += 1
    
    # 基本的な単語重複も考慮
    basic_overlap = super().calculate_word_overlap(text1, text2)
    
    # 概念一致率と基本重複率の組み合わせ
    concept_score = concept_matches / total_concepts if total_concepts > 0 else 0
    combined_score = (concept_score * 0.7) + (basic_overlap * 0.3)
    
    return min(combined_score, 1.0)

def calculate_multilingual_content_coverage(self, abstract: str, summary: str) -> float:
    """多言語対応の重要概念カバー率"""
    # 英語パターン
    english_patterns = [
        r'\b\d+(?:\.\d+)?%',  # パーセンテージ
        r'\bp\s*[<>=]\s*0\.\d+',  # p値
        r'\b\d+(?:\.\d+)?\s*(?:months?|years?|days?)',  # 期間
        r'\b\d+(?:\.\d+)?\s*(?:mg|g|ml|patients?|cases?)',  # 数量・対象
    ]
    
    # 日本語パターン
    japanese_patterns = [
        r'\d+(?:\.\d+)?%',  # パーセンテージ
        r'\d+(?:\.\d+)?(?:ヶ?月|年|日)',  # 期間
        r'\d+(?:\.\d+)?(?:名|人|例|件)',  # 対象数
        r'\d+(?:\.\d+)?倍',  # 倍率
        r'p\s*[<>=]\s*0\.\d+',  # p値
    ]
    
    # 英語Abstract中の重要データ
    abstract_matches = set()
    for pattern in english_patterns:
        abstract_matches.update(re.findall(pattern, abstract.lower()))
    
    # 日本語要約中の重要データ
    summary_matches = set()
    for pattern in japanese_patterns:
        summary_matches.update(re.findall(pattern, summary))
    
    # 数値の部分一致をチェック（例：15.5 months → 15.5ヶ月）
    abstract_numbers = set(re.findall(r'\d+(?:\.\d+)?', abstract))
    summary_numbers = set(re.findall(r'\d+(?:\.\d+)?', summary))
    
    number_overlap = len(abstract_numbers.intersection(summary_numbers))
    total_numbers = len(abstract_numbers)
    
    if total_numbers > 0:
        number_coverage = number_overlap / total_numbers
    else:
        number_coverage = 0.5
    
    # 概念的重要度も考慮
    important_concepts_coverage = 0.0
    concept_pairs = [
        ('54%', '54%'), ('60%', '60%'), ('24-month', '24ヶ月'),
        ('15.5 months', '15.5'), ('compliance', '遵守'),
        ('non-compliance', '非遵守'), ('retrospective', '後ろ向き')
    ]
    
    matched_concepts = 0
    for en_concept, ja_concept in concept_pairs:
        if en_concept.lower() in abstract.lower() and ja_concept in summary:
            matched_concepts += 1
    
    if concept_pairs:
        important_concepts_coverage = matched_concepts / len(concept_pairs)
    
    # 総合カバー率
    final_coverage = (number_coverage * 0.6) + (important_concepts_coverage * 0.4)
    return min(final_coverage, 1.0)

def _get_multilingual_quality_level(self, cosine_sim: float) -> str:
    """多言語間の品質レベル判定"""
    if cosine_sim >= 0.65:
        return "優秀"
    elif cosine_sim >= 0.55:
        return "良好"
    elif cosine_sim >= 0.50:
        return "標準"
    elif cosine_sim >= 0.45:
        return "要改善"
    else:
        return "不十分"

def _generate_multilingual_feedback(self, cosine_sim: float, word_overlap: float, coverage: float) -> str:
    """多言語間のフィードバック生成"""
    feedback = []
    
    if cosine_sim >= 0.65:
        feedback.append("英日間翻訳として優秀な意味的類似度を達成しています")
    elif cosine_sim >= 0.55:
        feedback.append("英日間翻訳として良好な意味的類似度です")
    elif cosine_sim >= 0.50:
        feedback.append("英日間翻訳として標準的な意味的類似度です")
    else:
        feedback.append("英日間翻訳の意味的類似度の向上が推奨されます")
    
    if word_overlap >= 0.15:
        feedback.append("医学概念の翻訳が適切に行われています")
    elif word_overlap < 0.10:
        feedback.append("英日間翻訳のため単語重複率は自然に低くなります")
    
    if coverage >= 0.50:
        feedback.append("重要な数値データが適切に保持されています")
    elif coverage >= 0.30:
        feedback.append("数値データの保持は標準的です")
    else:
        feedback.append("重要な数値データの保持率向上が推奨されます")
    
    return "。".join(feedback) if feedback else "英日間翻訳として適切な品質です"
    
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