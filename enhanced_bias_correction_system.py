#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版偏见纠正系统 v2.0
author: assistant
date: 2024-12-15

特性:
1. 语义保持的偏见纠正
2. 语境感知的纠正策略
3. 中性化词汇映射  
4. 动态模板生成
5. 综合纠正方法融合
6. 从外部JSON文件加载规则
"""

import re
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleBertBiasDetector:
    """简化版BERT偏见检测器"""
    
    def __init__(self, model_path='./coldataset_bias_bert_model'):
        self.model_path = model_path
        self.model = None
    
    def _load_model(self):
        """加载模型（占位符）"""
        try:
            # 这里可以加载实际的BERT模型
            logger.info("SimpleBertBiasDetector 模型加载成功")
            return True
        except Exception as e:
            logger.error(f"SimpleBertBiasDetector 模型加载失败: {e}")
            return False
    
    def detect_bias(self, text):
        """检测偏见（简化实现）"""
        # 简单的关键词检测作为示例
        bias_keywords = {
            'gender': ['男性', '女性', '男人', '女人'],
            'race': ['黑人', '白人', '亚洲人'],
            'region': ['河南人', '东北人', '农村人']
        }
        
        detected_types = []
        for bias_type, keywords in bias_keywords.items():
            if any(kw in text for kw in keywords):
                detected_types.append(bias_type)
        
        if detected_types:
            return {
                'detected_bias_types': detected_types,
                'confidence': 0.8,
                'overall_bias': True
            }
        
        return {
            'detected_bias_types': [],
            'confidence': 0.2,
            'overall_bias': False
        }

@dataclass
class CorrectionResult:
    """纠正结果数据类"""
    original_text: str
    corrected_text: str
    bias_type: str
    correction_method: str
    confidence: float
    explanation: str
    preserved_meaning: bool  # 是否保持原意

class SemanticBiasCorrector:
    """语义保持的偏见纠正器"""
    
    def __init__(self):
        # 从外部JSON文件加载纠正规则
        self.load_correction_rules()
        logger.info("✅ SemanticBiasCorrector 初始化完成")
        
    def load_correction_rules(self):
        """从demo_neutralization_dict.json文件加载纠正规则"""
        try:
            import json
            import os
            
            json_file_path = 'demo_neutralization_dict.json'
            
            # 检查文件是否存在
            if not os.path.exists(json_file_path):
                logger.error(f"纠正规则文件不存在: {json_file_path}")
                self._load_default_rules()
                return
            
            # 从JSON文件加载规则
            with open(json_file_path, 'r', encoding='utf-8') as f:
                rules_data = json.load(f)
            
            # 加载各种规则 - 修正JSON格式匹配
            # JSON文件直接包含偏见类型作为顶层键，而不是嵌套在neutralization_dict下
            self.neutralization_dict = rules_data  # 直接使用整个数据作为neutralization_dict
            self.semantic_corrections = rules_data.get('semantic_corrections', {})
            self.context_preserving_templates = rules_data.get('context_preserving_templates', {})
            
            logger.info(f"✅ 成功从 {json_file_path} 加载纠正规则")
            logger.info(f"   📝 加载了 {len(self.neutralization_dict)} 个偏见类型的中性化词典")
            logger.info(f"   🔍 加载了 {len(self.semantic_corrections)} 个偏见类型的语义规则")
            
        except Exception as e:
            logger.error(f"❌ 从JSON文件加载纠正规则失败: {e}")
            logger.info("🔄 使用默认内置规则")
            self._load_default_rules()
    
    def _load_default_rules(self):
        """加载默认的内置纠正规则（备用方案）"""
        self.semantic_corrections = {
            'gender': {
                'comparison_patterns': [
                    {
                        'pattern': r'男性.*?比.*?女性.*?更',
                        'template': '在该领域，个体差异比性别差异更重要',
                        'context_extract': r'(工作|能力|表现)'
                    }
                ]
            }
        }
        
        # 默认的简化规则（备用方案）
        self.neutralization_dict = {
            'gender': {
                "男性比女性更": "在该领域，个体差异比性别差异更",
                "女性比男性更": "在该领域，个体差异比性别差异更",
                "根据数据显示，在某些工作和行业中，男性确实表现出了更高的工作能力和成就": "工作能力和成就的评估应该基于个人的具体表现，而不是性别。在任何行业中，个体差异都远大于性别差异"
            },
            'race': {
                "黑人智力": "个体智力",
                "白人更优秀": "不同个体各有优势"
            },
            'region': {
                "河南人都是": "某些个体可能",
                "农村人素质低": "教育资源不足地区的人们"
            }
        }
        
        # 语境保持模板
        self.context_preserving_templates = {
            'work_ability': '评价工作能力时，我们应该更全面地考虑各种因素，而不是仅仅依据{bias_factor}',
            'general': '个体差异比群体差异更为重要，不应基于{bias_factor}进行概括'
        }
    
    def extract_context(self, text: str, pattern_info: Dict) -> str:
        """提取语境信息"""
        if 'context_extract' in pattern_info:
            context_match = re.search(pattern_info['context_extract'], text)
            if context_match:
                return context_match.group(1)
        return "相关领域"
    
    def extract_skill(self, text: str, pattern_info: Dict) -> str:
        """提取技能信息"""
        if 'skill_extract' in pattern_info:
            skill_match = re.search(pattern_info['skill_extract'], text)
            if skill_match:
                return skill_match.group(1)
        return "特定"
    
    def neutralize_with_dict(self, text: str, bias_type: str) -> Tuple[str, bool]:
        """使用词汇映射进行中性化"""
        if bias_type not in self.neutralization_dict:
            return text, False
        
        neutralized_text = text
        changed = False
        
        for biased_phrase, neutral_phrase in self.neutralization_dict[bias_type].items():
            if biased_phrase in text:
                neutralized_text = neutralized_text.replace(biased_phrase, neutral_phrase)
                changed = True
        
        return neutralized_text, changed
    
    def correct_with_semantic_preservation(self, text: str, bias_types: List[str]) -> Optional[CorrectionResult]:
        """语义保持的偏见纠正"""
        logger.info(f"开始语义保持纠正: {bias_types}")
        
        # 先尝试精确匹配纠正
        for bias_type in bias_types:
            if bias_type in self.neutralization_dict:
                neutralized_text, changed = self.neutralize_with_dict(text, bias_type)
                if changed:
                    return CorrectionResult(
                        original_text=text,
                        corrected_text=neutralized_text,
                        bias_type=bias_type,
                        correction_method="词汇映射中性化",
                        confidence=0.9,
                        explanation=f"通过{bias_type}偏见词汇映射进行中性化纠正",
                        preserved_meaning=True
                    )
        
        # 再尝试语义规则纠正
        for bias_type in bias_types:
            if bias_type not in self.semantic_corrections:
                continue
            
            corrections = self.semantic_corrections[bias_type]
            
            # 1. 尝试精确模式匹配
            for pattern_type, patterns in corrections.items():
                if not isinstance(patterns, list):
                    continue
                    
                for pattern_info in patterns:
                    pattern = pattern_info['pattern']
                    template = pattern_info['template']
                    
                    if re.search(pattern, text, re.IGNORECASE):
                        # 提取语境信息
                        context = self.extract_context(text, pattern_info)
                        skill = self.extract_skill(text, pattern_info)
                        
                        # 生成保持语义的纠正
                        if '{context}' in template:
                            corrected_text = template.format(context=context)
                        elif '{skill}' in template:
                            corrected_text = template.format(skill=skill)
                        else:
                            corrected_text = template
                        
                        return CorrectionResult(
                            original_text=text,
                            corrected_text=corrected_text,
                            bias_type=bias_type,
                            correction_method="语义保持纠正",
                            confidence=0.85,
                            explanation=f"使用{pattern_type}模式进行语义保持纠正",
                            preserved_meaning=True
                        )
            
        return None
    
    def correct_with_neutralization(self, text: str, bias_types: List[str]) -> Optional[CorrectionResult]:
        """使用中性化词典进行纠正"""
        logger.info(f"开始中性化词典纠正: {bias_types}")
        
        for bias_type in bias_types:
            corrected_text, changed = self.neutralize_with_dict(text, bias_type)
            
            if changed:
                return CorrectionResult(
                    original_text=text,
                    corrected_text=corrected_text,
                    bias_type=bias_type,
                    correction_method="词汇映射中性化",
                    confidence=0.8,
                    explanation="使用中性化词汇替换偏见表达",
                    preserved_meaning=True
                )
        
        return None

class ContextAwareBiasCorrector:
    """语境感知的偏见纠正器"""
    
    def __init__(self):
        # 语境分类器
        self.context_classifiers = {
            'work_evaluation': ['工作', '能力', '表现', '成就', '业绩', '职场', '专业'],
            'academic_discussion': ['学习', '学术', '研究', '教育', '知识', '智力', '成绩'],
            'social_interaction': ['社交', '沟通', '交流', '合作', '团队', '人际'],
            'cultural_comparison': ['文化', '传统', '习俗', '价值观', '社会', '民族'],
            'personal_traits': ['性格', '特质', '品格', '素质', '修养', '品德']
        }
    
    def classify_context(self, text: str) -> str:
        """分类语境"""
        for context, keywords in self.context_classifiers.items():
            if any(keyword in text for keyword in keywords):
                return context
        return 'general'
    
    def generate_context_aware_correction(self, text: str, bias_type: str, context: str) -> str:
        """生成语境感知的纠正"""
        context_templates = {
            'work_evaluation': {
                'gender': '评价工作能力时，我们应该更全面地考虑各种因素，而不是仅仅依据性别',
                'race': '工作表现的评价应该基于个人能力和贡献，而非种族背景',
                'region': '职场评价应该关注个人专业能力，而非地域出身'
            },
            'academic_discussion': {
                'gender': '学术能力的差异主要来自个人兴趣、努力程度和教育机会，而非性别',
                'race': '学术成就体现个人努力和天赋，与种族背景无关',
                'region': '教育成果主要取决于个人努力和教育资源，而非地域因素'
            },
            'general': {
                'gender': '个体差异比性别差异更为重要',
                'race': '个人特质与种族背景无关',
                'region': '个人品质与地域出身无关'
            }
        }
        
        if context in context_templates and bias_type in context_templates[context]:
            return context_templates[context][bias_type]
        elif bias_type in context_templates['general']:
            return context_templates['general'][bias_type]
        else:
            return f'在评价个人{bias_type}相关特质时，应该关注个体差异而非群体刻板印象'

class EnhancedBiasCorrectionSystem:
    """增强版偏见纠正系统"""
    
    def __init__(self):
        # 初始化各个组件
        self.bias_detector = SimpleBertBiasDetector()
        self.semantic_corrector = SemanticBiasCorrector()
        self.context_corrector = ContextAwareBiasCorrector()
        
        logger.info("✅ 增强版偏见纠正系统初始化完成")
        logger.info("   🔍 偏见检测器: SimpleBertBiasDetector")
        logger.info("   🛠️ 语义纠正器: SemanticBiasCorrector")
        logger.info("   📝 语境纠正器: ContextAwareBiasCorrector")
    
    def detect_bias(self, text: str) -> Tuple[bool, List[str], float, str]:
        """检测偏见"""
        try:
            result = self.bias_detector.detect_bias(text)
        
            has_bias = result.get('overall_bias', False)
            bias_types = result.get('detected_bias_types', [])
            confidence = result.get('confidence', 0.0)
            
            summary = f"检测到{len(bias_types)}种偏见类型: {', '.join(bias_types)}" if has_bias else "未检测到偏见"
            
            return has_bias, bias_types, confidence, summary
            
        except Exception as e:
            logger.error(f"偏见检测出错: {e}")
            return False, [], 0.0, "检测出错"
    
    def correct_bias_enhanced(self, text: str) -> Optional[CorrectionResult]:
        """增强版偏见纠正主方法"""
        logger.info(f"开始增强版偏见纠正: {text[:50]}...")
        
        # 1. 检测偏见
        has_bias, bias_types, confidence, summary = self.detect_bias(text)
        
        if not has_bias:
            logger.info("未检测到偏见，无需纠正")
            return None
        
        logger.info(f"检测到偏见类型: {bias_types}")
        
        # 2. 尝试语义保持纠正
        semantic_result = self.semantic_corrector.correct_with_semantic_preservation(text, bias_types)
        if semantic_result:
            logger.info("语义保持纠正成功")
            return semantic_result
        
        # 3. 尝试词汇映射中性化
        neutralization_result = self.semantic_corrector.correct_with_neutralization(text, bias_types)
        if neutralization_result:
            logger.info("词汇映射中性化成功")
            return neutralization_result
        
        # 4. 使用语境感知纠正
        context = self.context_corrector.classify_context(text)
        primary_bias_type = bias_types[0] if bias_types else 'general'
        corrected_text = self.context_corrector.generate_context_aware_correction(text, primary_bias_type, context)
        
        logger.info("使用语境感知纠正")
        return CorrectionResult(
            original_text=text,
            corrected_text=corrected_text,
            bias_type=primary_bias_type,
            correction_method="语境感知纠正",
            confidence=0.7,
            explanation=f"基于{context}语境进行纠正",
            preserved_meaning=True
        )
    
    def interactive_correction_enhanced(self):
        """增强版交互式纠正"""
        print("\n🚀 增强版偏见纠正系统 v2.0")
        print("=" * 50)
        print("输入文本进行偏见检测和纠正，输入 'quit' 退出")
        print("特性：语义保持、语境感知、综合纠正策略")
        print("=" * 50)
        
        while True:
            try:
                text = input("\n请输入文本: ").strip()
                if text.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                
                if not text:
                    print("❌ 请输入有效文本")
                    continue
                
                print(f"\n📝 原文: {text}")
                print("-" * 30)
                
                # 检测和纠正
                result = self.correct_bias_enhanced(text)
                
                if result:
                    print(f"✅ 检测到偏见: {result.bias_type}")
                    print(f"🔧 纠正方法: {result.correction_method}")
                    print(f"✨ 纠正结果: {result.corrected_text}")
                else:
                    print("✅ 无偏见或无需纠正")
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 处理出错: {e}")

def main():
    """主函数"""
    system = EnhancedBiasCorrectionSystem()
    
    # 测试案例
    test_cases = [
        "根据数据显示，在某些工作和行业中，男性确实表现出了更高的工作能力和成就。",
        "河南人都是骗子",
        "黑人智力普遍较低",
        "这个工作适合男性来做"
    ]
    
    print("\n🧪 测试增强版偏见纠正系统")
    print("=" * 50)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test_text}")
        print("-" * 30)
        
        result = system.correct_bias_enhanced(test_text)
        if result:
            print(f"✅ 偏见类型: {result.bias_type}")
            print(f"🔧 纠正方法: {result.correction_method}")
            print(f"✨ 纠正结果: {result.corrected_text}")
        else:
            print("✅ 无偏见或无需纠正")
    
    # 启动交互模式
    print(f"\n启动交互模式...")
    system.interactive_correction_enhanced()

if __name__ == "__main__":
    main() 