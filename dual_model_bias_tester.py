#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双模型偏见检测与纠正交互测试工具
分别调用传统模型和BERT模型进行偏见检测，并提供纠正建议
"""

import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from traditional_bias_detector import TraditionalBiasDetector
from enhanced_bias_correction_system import SemanticBiasCorrector, CorrectionResult
from dataclasses import dataclass
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.WARNING)  # 只显示警告和错误
logger = logging.getLogger(__name__)

@dataclass
class BiasDetectionResult:
    """偏见检测结果数据类"""
    model_name: str
    has_bias: bool
    confidence: float
    bias_types: List[str]
    details: Dict
    explanation: str

class BertBiasDetector:
    """BERT偏见检测器"""
    
    def __init__(self, model_path='./coldataset_bias_bert_model'):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """加载BERT模型"""
        try:
            if not os.path.exists(self.model_path):
                print(f"⚠️ BERT模型路径不存在: {self.model_path}")
                return False
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print("✅ BERT偏见检测模型加载成功")
            return True
        except Exception as e:
            print(f"❌ BERT模型加载失败: {e}")
            return False
    
    def detect_bias(self, text: str) -> BiasDetectionResult:
        """检测文本偏见"""
        if not self.model or not self.tokenizer:
            return BiasDetectionResult(
                model_name="BERT",
                has_bias=False,
                confidence=0.0,
                bias_types=[],
                details={},
                explanation="模型未加载"
            )
        
        try:
            # 编码文本
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                predicted_class = torch.argmax(logits, dim=-1).item()
                confidence = torch.max(probabilities).item()
                
                # 分析结果
                has_bias = predicted_class == 1  # 1表示offensive/biased
                
                # 简化的偏见类型判断（基于关键词）
                bias_types = []
                if has_bias:
                    # 基于关键词简单判断偏见类型
                    if any(word in text for word in ['男性', '女性', '男人', '女人', '男的', '女的']):
                        bias_types.append('gender')
                    if any(word in text for word in ['黑人', '白人', '亚洲人', '中国人', '日本人', '韩国人']):
                        bias_types.append('race')
                    if any(word in text for word in ['河南人', '东北人', '上海人', '北京人', '农村人', '城里人']):
                        bias_types.append('region')
                    if not bias_types:
                        bias_types.append('general')
                
                explanation = f"BERT模型预测: {'有偏见' if has_bias else '无偏见'} (置信度: {confidence:.3f})"
                
                return BiasDetectionResult(
                    model_name="BERT",
                    has_bias=has_bias,
                    confidence=confidence,
                    bias_types=bias_types,
                    details={
                        'predicted_class': predicted_class,
                        'raw_probabilities': probabilities.cpu().numpy().tolist(),
                        'device': str(self.device)
                    },
                    explanation=explanation
                )
                
        except Exception as e:
            return BiasDetectionResult(
                model_name="BERT",
                has_bias=False,
                confidence=0.0,
                bias_types=[],
                details={'error': str(e)},
                explanation=f"BERT检测出错: {e}"
            )

class TraditionalBiasDetectorWrapper:
    """传统偏见检测器包装类"""
    
    def __init__(self):
        self.detector = TraditionalBiasDetector()
        print("✅ 传统偏见检测模型加载成功")
    
    def detect_bias(self, text: str) -> BiasDetectionResult:
        """检测文本偏见"""
        try:
            # 调用传统检测器 - 降低阈值增加敏感性
            result = self.detector.detect_bias(text, threshold_svm=0.25, threshold_sentiment=0.15)
            
            if not result:
                return BiasDetectionResult(
                    model_name="Traditional",
                    has_bias=False,
                    confidence=0.5,
                    bias_types=[],
                    details={},
                    explanation="传统模型检测失败"
                )
            
            summary = result.get('summary', {})
            detection_results = result.get('detection_results', {})
            
            has_bias = summary.get('is_biased', False)
            confidence = summary.get('confidence', 0.0)
            bias_types = summary.get('bias_types', [])
            
            # 获取详细信息
            sensitive_words = detection_results.get('sensitive_words', {})
            svm_prediction = detection_results.get('svm_prediction', {})
            sentiment_analysis = detection_results.get('sentiment_analysis', {})
            fairness_check = detection_results.get('fairness_check', {})
            
            explanation = f"传统模型检测: {'有偏见' if has_bias else '无偏见'} (置信度: {confidence:.3f})"
            if has_bias and bias_types:
                types_str = '、'.join(bias_types)
                explanation += f"，偏见类型: {types_str}"
            
            return BiasDetectionResult(
                model_name="Traditional",
                has_bias=has_bias,
                confidence=confidence,
                bias_types=bias_types,
                details={
                    'sensitive_words_found': len(sensitive_words.get('found_words', [])),
                    'svm_probability': svm_prediction.get('probability', 0),
                    'sentiment_intensity': sentiment_analysis.get('intensity', 0),
                    'fairness_violations': len(fairness_check.get('violations', [])),
                    'detection_components': detection_results.get('flow', [])
                },
                explanation=explanation
            )
            
        except Exception as e:
            return BiasDetectionResult(
                model_name="Traditional",
                has_bias=False,
                confidence=0.0,
                bias_types=[],
                details={'error': str(e)},
                explanation=f"传统模型检测出错: {e}"
            )

class DualModelBiasTester:
    """双模型偏见检测与纠正系统"""
    
    def __init__(self):
        print("🚀 初始化双模型偏见检测与纠正系统")
        print("=" * 60)
        
        # 初始化检测器
        self.traditional_detector = TraditionalBiasDetectorWrapper()
        self.bert_detector = BertBiasDetector()
        
        # 初始化纠正器
        self.corrector = SemanticBiasCorrector()
        
        print("=" * 60)
        print("✅ 双模型系统初始化完成")
        print()
    
    def detect_with_both_models(self, text: str) -> tuple[BiasDetectionResult, BiasDetectionResult]:
        """使用两个模型分别检测偏见"""
        print(f"🔍 正在检测文本: {text}")
        print("-" * 40)
        
        # 分别调用两个检测器
        traditional_result = self.traditional_detector.detect_bias(text)
        bert_result = self.bert_detector.detect_bias(text)
        
        return traditional_result, bert_result
    
    def correct_bias(self, text: str, bias_types: List[str]) -> Optional[CorrectionResult]:
        """纠正偏见"""
        if not bias_types:
            return None
        
        try:
            return self.corrector.correct_with_semantic_preservation(text, bias_types)
        except Exception as e:
            print(f"⚠️ 偏见纠正失败: {e}")
            return None
    
    def display_results(self, traditional_result: BiasDetectionResult, bert_result: BiasDetectionResult):
        """显示检测结果"""
        print("📊 检测结果对比:")
        print()
        
        # 传统模型结果
        print("🔧 传统模型 (COLDataset训练)")
        print(f"   结果: {traditional_result.explanation}")
        if traditional_result.has_bias:
            print(f"   偏见类型: {', '.join(traditional_result.bias_types) if traditional_result.bias_types else '未知'}")
            print(f"   详细信息: 敏感词{traditional_result.details.get('sensitive_words_found', 0)}个, "
                  f"SVM概率{traditional_result.details.get('svm_probability', 0):.3f}")
        print()
        
        # BERT模型结果
        print("🧠 BERT模型 (深度学习)")
        print(f"   结果: {bert_result.explanation}")
        if bert_result.has_bias:
            print(f"   偏见类型: {', '.join(bert_result.bias_types) if bert_result.bias_types else '未知'}")
            print(f"   设备: {bert_result.details.get('device', 'CPU')}")
        print()
        
        # 综合判断
        both_detect_bias = traditional_result.has_bias and bert_result.has_bias
        any_detect_bias = traditional_result.has_bias or bert_result.has_bias
        
        if both_detect_bias:
            print("⚠️ 综合判断: 两个模型都检测到偏见，建议进行纠正")
            combined_bias_types = list(set(traditional_result.bias_types + bert_result.bias_types))
            return True, combined_bias_types
        elif any_detect_bias:
            print("⚡ 综合判断: 一个模型检测到偏见，存在潜在风险")
            combined_bias_types = list(set(traditional_result.bias_types + bert_result.bias_types))
            return True, combined_bias_types
        else:
            print("✅ 综合判断: 两个模型都未检测到明显偏见")
            return False, []
    
    def interactive_test(self):
        """交互式测试"""
        print("🧪 双模型偏见检测与纠正交互工具")
        print("=" * 60)
        print("💡 功能说明:")
        print("   - 同时使用传统模型和BERT模型检测偏见")
        print("   - 对比两个模型的检测结果")
        print("   - 提供智能偏见纠正建议")
        print("   - 输入 'quit' 退出程序")
        print("=" * 60)
        
        while True:
            try:
                print("\n" + "🔹" * 50)
                text = input("请输入要检测的文本: ").strip()
                
                if not text:
                    print("⚠️ 请输入有效文本")
                    continue
                
                if text.lower() in ['quit', 'exit', '退出', 'q']:
                    print("\n👋 感谢使用双模型偏见检测系统，再见！")
                    break
                
                print()
                
                # 双模型检测
                traditional_result, bert_result = self.detect_with_both_models(text)
                
                # 显示结果
                needs_correction, combined_bias_types = self.display_results(traditional_result, bert_result)
                
                # 如果需要纠正
                if needs_correction and combined_bias_types:
                    print("🔧 正在生成纠正建议...")
                    correction_result = self.correct_bias(text, combined_bias_types)
                    
                    if correction_result:
                        print("✨ 偏见纠正建议:")
                        print(f"   原文: {correction_result.original_text}")
                        print(f"   纠正后: {correction_result.corrected_text}")
                        print(f"   纠正方法: {correction_result.correction_method}")
                        print(f"   置信度: {correction_result.confidence:.3f}")
                        print(f"   保持原意: {'✅' if correction_result.preserved_meaning else '❌'}")
                        print(f"   说明: {correction_result.explanation}")
                    else:
                        print("⚠️ 无法生成合适的纠正建议，建议人工审查")
                
                print("\n" + "✅" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 用户中断，程序退出")
                break
            except Exception as e:
                print(f"\n❌ 程序出错: {e}")
                import traceback
                traceback.print_exc()

def main():
    """主函数"""
    try:
        tester = DualModelBiasTester()
        tester.interactive_test()
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 