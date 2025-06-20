#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传统偏见检测系统 - COLDataset增强版
基于多层级流程的偏见检测：敏感词匹配 -> SVM判断 -> 情感分析 -> 公平性检查
现已集成COLDataset训练的真实数据词典和SVM模型
"""

import re
import os
import json
import jieba
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging

# COLDataset模型文件路径配置 - 使用一半数据训练的词典
MODEL_DIR = 'coldataset_bias_traditional_model_half'
SENSITIVE_WORDS_FILE = os.path.join(MODEL_DIR, 'new_sensitive_words.json')
SENTIMENT_WORDS_FILE = os.path.join(MODEL_DIR, 'new_sentiment_words.json')
SVM_MODEL_FILE = os.path.join(MODEL_DIR, 'svm_model.pkl')

class SensitiveWordMatcher:
    """敏感词正则匹配器 - COLDataset增强版"""
    
    def __init__(self):
        self.patterns = {}
        self.sensitive_words = {}
        self._load_dictionaries()
    
    def _load_dictionaries(self):
        """加载COLDataset训练的敏感词词典"""
        try:
            # 尝试加载COLDataset训练的词典
            if os.path.exists(SENSITIVE_WORDS_FILE):
                with open(SENSITIVE_WORDS_FILE, 'r', encoding='utf-8') as f:
                    coldataset_words = json.load(f)
                
                # 使用COLDataset词典
                self.sensitive_words = {
                    'gender': coldataset_words.get('gender', []),
                    'race': coldataset_words.get('race', []), 
                    'region': coldataset_words.get('region', [])
                }
                print(f"✅ 已加载COLDataset敏感词词典")
                print(f"   - 性别词汇: {len(self.sensitive_words['gender'])}个")
                print(f"   - 种族词汇: {len(self.sensitive_words['race'])}个") 
                print(f"   - 地域词汇: {len(self.sensitive_words['region'])}个")
            else:
                # 回退到默认词典
                self._load_default_dictionaries()
                print(f"⚠️  COLDataset词典未找到，使用默认词典")
                
        except Exception as e:
            print(f"❌ 加载COLDataset词典失败: {e}")
            self._load_default_dictionaries()
        
        # 编译正则模式
        self._compile_patterns()
    
    def _load_default_dictionaries(self):
        """加载默认敏感词词典（备用方案）"""
        # 性别偏见相关词汇 - 大幅扩展
        self.sensitive_words['gender'] = [
            # 原有词汇
            '男性更适合', '女性不适合', '男人比女人', '女人比男人',
            '男性天生', '女性天生', '男的就是', '女的就是',
            '男人都', '女人都', '男生比女生', '女生比男生',
            '男性优于', '女性劣于', '男性能力强', '女性能力差',
            '男性工作能力', '女性工作能力', '男性表现', '女性表现',
            
            # 职业性别刻板印象
            '男性当CEO', '女性当护士', '男性做工程师', '女性做秘书',
            '男性适合领导', '女性适合照顾', '男性理性', '女性感性',
            '男性逻辑强', '女性直觉好', '男性技术好', '女性沟通强',
            '程序员都是男的', '护士都是女的', '司机应该是男的', '幼师应该是女的',
            
            # 能力比较
            '男性数学好', '女性语言好', '男性空间感强', '女性记忆力好',
            '男的理科强', '女的文科好', '男性创新能力', '女性细心',
            '男人决策快', '女人犹豫不决', '男性果断', '女性优柔寡断',
            '男的有魄力', '女的没主见', '男性坚强', '女性脆弱',
            
            # 性格特征
            '男人大大咧咧', '女人斤斤计较', '男人粗心', '女人细腻',
            '男的不拘小节', '女的爱计较', '男性沉稳', '女性情绪化',
            '男人理智', '女人感情', '男的冷静', '女的歇斯底里',
            
            # 社会角色
            '男主外女主内', '男人养家', '女人相夫教子', '女人应该在家',
            '男人赚钱', '女人花钱', '男人顶梁柱', '女人依附',
            '男性事业重要', '女性家庭重要', '女人最终要结婚', '男人要有车房',
            
            # 外貌和身体
            '女人靠脸', '男人靠实力', '女性重外表', '男性重能力',
            '女人年龄是秘密', '男人四十一枝花', '女性保养', '男性不修边幅'
        ]
        
        # 年龄偏见相关词汇 - 大幅扩展
        self.sensitive_words['age'] = [
            # 原有词汇
            '年轻人不懂', '老年人跟不上', '年纪大了', '太年轻',
            '老了就', '年轻就是', '中年危机', '老人家',
            '年轻人都', '老年人都',
            
            # 年龄刻板印象
            '90后不靠谱', '00后没责任心', '80后现实', '70后保守',
            '年轻人浮躁', '老年人固执', '中年人保守', '老头老太',
            '小年轻', '老古董', '新新人类', '老一辈',
            
            # 能力与年龄
            '年轻人经验不足', '老年人学不会', '年龄大了记忆差', '年轻人不成熟',
            '老年人思想落后', '年轻人没耐心', '中年人没激情', '老年人不适应',
            '年轻人创新', '老年人传统', '年轻人冲动', '老年人谨慎',
            
            # 技术与年龄
            '老年人不会用电脑', '年轻人沉迷网络', '老年人不懂科技', '年轻人只会玩手机',
            '老年人跟不上时代', '年轻人没有传统文化', '老一辈不懂新事物', '年轻一代没礼貌',
            
            # 工作与年龄
            '年龄大了没用', '老员工效率低', '年轻人不稳定', '老年人该退休',
            '35岁危机', '年龄歧视', '老年人占着位子', '年轻人上位快'
        ]
        
        # 职业偏见相关词汇 - 大幅扩展
        self.sensitive_words['occupation'] = [
            # 原有词汇
            '某某职业的人都', '这个行业的人', '做这行的',
            '程序员都', '医生都', '老师都', '销售都',
            '农民工', '蓝领', '白领优于',
            
            # 职业刻板印象
            '程序员都很宅', '医生都很冷漠', '老师都很穷', '律师都很精明',
            '销售都很能说', '会计都很细心', '艺术家都很感性', '科学家都很理性',
            '警察都很威严', '护士都很温柔', '军人都很严肃', '商人都很现实',
            
            # 职业能力偏见
            '做IT的都是男的', '当护士的都是女的', '开车的最好是男的', '做销售要会喝酒',
            '当老师的都很闲', '做公务员的都很稳定', '搞艺术的都不赚钱', '做生意的都很精',
            '当医生的都很累', '做律师的都很忙', '搞研究的都很书呆', '做管理的都很忙',
            
            # 行业等级偏见
            '金融行业高大上', '服务行业低端', '制造业没前途', '互联网行业加班多',
            '传统行业落后', '新兴行业不稳定', '国企工作轻松', '私企压力大',
            '外企待遇好', '民企不正规', '大公司稳定', '小公司没前途',
            
            # 学历与职业
            '名校毕业进大厂', '普通学校做销售', '学历低当工人', '高学历搞研究',
            '博士都很书呆', '硕士找工作难', '本科生最实用', '专科生技术好',
            
            # 收入与职业
            '做金融的都有钱', '当老师的工资低', '做生意的都暴富', '打工的都很穷',
            '公务员旱涝保收', '私企老板都有钱', '创业的风险大', '上班族很稳定'
        ]
        
        # 地域偏见相关词汇 - 大幅扩展
        self.sensitive_words['region'] = [
            # 原有词汇
            '某某地方的人', '北方人', '南方人', '农村人',
            '城里人', '外地人', '本地人', '山区人',
            
            # 具体地域偏见
            '河南人都是骗子', '东北人都很粗鲁', '上海人都势利', '北京人都傲慢',
            '广东人都精明', '山东人都朴实', '四川人都能吃辣', '新疆人都彪悍',
            '天津人都会说相声', '湖南人都能吃辣', '江苏人都有钱', '浙江人都会做生意',
            
            # 城乡偏见
            '农村人没文化', '城里人瞧不起人', '农村人见识少', '城里人很现实',
            '山里人很朴实', '城市人很精明', '乡下人土气', '城里人洋气',
            '农民工素质低', '城里人有素质', '乡下人憨厚', '城里人狡猾',
            
            # 发达程度偏见
            '发达地区人素质高', '落后地区人没素质', '沿海地区人开放', '内陆地区人保守',
            '一线城市人见识广', '小城市人眼界窄', '大城市人冷漠', '小地方人热情',
            '经济发达地区教育好', '贫困地区教育差', '富裕地区人文明', '贫困地区人野蛮',
            
            # 南北差异
            '北方人豪爽', '南方人精明', '北方人直接', '南方人细腻',
            '北方人能喝酒', '南方人会做生意', '北方人身材高', '南方人身材矮',
            '北方人性格急', '南方人性格慢', '北方人大嗓门', '南方人小声说话',
            
            # 省份特征
            '山西人都挖煤', '陕西人都吃面', '重庆人都很辣', '云南人都很懒',
            '贵州人都很穷', '甘肃人都很苦', '青海人都很冷', '西藏人都很神秘',
            '内蒙人都放牧', '宁夏人都很干', '海南人都很热', '黑龙江人都很冷'
        ]
        
        # 新增：种族偏见相关词汇
        self.sensitive_words['race'] = [
            # 种族智力偏见
            '黑人智商低', '白人智商高', '亚洲人最聪明', '犹太人最精明',
            '黄种人数学好', '白种人创新强', '黑种人运动强', '混血儿聪明',
            
            # 种族能力偏见
            '中国人勤奋', '日本人严谨', '韩国人整容', '印度人数学好',
            '美国人创新', '德国人严肃', '法国人浪漫', '意大利人热情',
            '俄罗斯人能喝酒', '巴西人会踢球', '非洲人会跑步', '北欧人很冷淡',
            
            # 宗教偏见
            '穆斯林都是恐怖分子', '基督徒都很保守', '佛教徒都很慈悲', '犹太人都很贪婪',
            '道教徒都很神秘', '天主教徒都很传统', '新教徒都很开放', '无神论者都很现实',
            
            # 肤色偏见
            '白皮肤好看', '黑皮肤不好看', '黄皮肤一般', '混血儿最美',
            '皮肤白有优越感', '皮肤黑被歧视', '肤色决定地位', '白人最优秀'
        ]
        
        # 新增：身体特征偏见词汇
        self.sensitive_words['physical'] = [
            # 身高偏见
            '高个子都聪明', '矮个子都笨', '个子高有优势', '个子矮没前途',
            '高个子适合当领导', '矮个子只能做小事', '身高决定能力', '个子影响发展',
            
            # 体型偏见
            '胖子都很懒', '瘦子都很弱', '胖子没毅力', '瘦子没力气',
            '身材好有优势', '身材差被歧视', '体重影响工作', '外貌决定命运',
            
            # 外貌偏见
            '长得好看有优势', '长得丑没前途', '颜值即正义', '外貌协会',
            '帅哥都花心', '美女都花瓶', '丑男有才华', '丑女有内涵',
            
            # 身体缺陷偏见
            '残疾人都很可怜', '盲人都很敏感', '聋哑人都很孤独', '肢体残疾影响能力',
            '戴眼镜都是书呆子', '秃头的都很聪明', '有疤痕的都很凶', '牙齿不好影响形象'
        ]
        
        # 新增：经济地位偏见词汇
        self.sensitive_words['economic'] = [
            # 财富偏见
            '富人都很自私', '穷人都很懒', '有钱人没素质', '没钱人很朴实',
            '富二代都很嚣张', '穷二代很努力', '暴发户没文化', '老富豪有底蕴',
            
            # 学历偏见
            '名校毕业都优秀', '普通学校都一般', '学历高能力强', '学历低没前途',
            '博士都很书呆', '硕士很普通', '本科刚刚好', '专科技术强',
            '高学历都理论', '低学历都实践', '海归都很牛', '土鳖都很土',
            
            # 社会地位偏见
            '领导都很忙', '下属都很闲', '管理层都很累', '基层都很轻松',
            '白领都很累', '蓝领都很苦', '金领都很爽', '灰领都很稳',
            '公务员都很闲', '企业员工都很忙', '事业单位很稳定', '私企很不稳定'
        ]
    
    def _compile_patterns(self):
        """编译正则模式"""
        for category, words in self.sensitive_words.items():
            patterns = []
            for word in words:
                # 创建更灵活的匹配模式
                pattern = word.replace('某某', r'.{1,4}')
                patterns.append(pattern)
            self.patterns[category] = '|'.join(patterns) if patterns else ''
    
    def match(self, text: str) -> Dict[str, List[str]]:
        """匹配敏感词"""
        matches = {}
        for category, pattern in self.patterns.items():
            if pattern:  # 确保模式不为空
                found = re.findall(pattern, text, re.IGNORECASE)
                if found:
                    matches[category] = found
        return matches

class ContextFeatureExtractor:
    """语境特征提取器"""
    
    def __init__(self):
        self.bias_indicators = [
            '总是', '永远', '从来', '所有', '都是', '肯定',
            '一定', '绝对', '必然', '天生', '就是', '只有'
        ]
        
        self.comparison_words = [
            '比', '超过', '优于', '劣于', '强于', '弱于',
            '不如', '胜过', '高于', '低于'
        ]
        
        self.emotional_words = [
            '讨厌', '喜欢', '崇拜', '鄙视', '羡慕', '嫉妒',
            '愤怒', '失望', '满意', '不满'
        ]
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """提取语境特征"""
        features = {}
        
        # 绝对化表述频率
        absolute_count = sum(1 for word in self.bias_indicators if word in text)
        features['absolute_ratio'] = absolute_count / max(len(text), 1)
        
        # 比较性词汇频率
        comparison_count = sum(1 for word in self.comparison_words if word in text)
        features['comparison_ratio'] = comparison_count / max(len(text), 1)
        
        # 情感性词汇频率
        emotional_count = sum(1 for word in self.emotional_words if word in text)
        features['emotional_ratio'] = emotional_count / max(len(text), 1)
        
        # 文本长度特征
        features['text_length'] = len(text)
        features['sentence_count'] = text.count('。') + text.count('！') + text.count('？')
        
        # 问号和感叹号比例
        features['question_ratio'] = text.count('？') / max(len(text), 1)
        features['exclamation_ratio'] = text.count('！') / max(len(text), 1)
        
        return features

class SVMBiasClassifier:
    """基于SVM的偏见分类器 - COLDataset增强版"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.classifier = SVC(kernel='rbf', probability=True, random_state=42)
        self.feature_extractor = ContextFeatureExtractor()
        self.is_trained = False
        self._load_pretrained_model()
    
    def _load_pretrained_model(self):
        """加载COLDataset预训练的SVM模型"""
        try:
            if os.path.exists(SVM_MODEL_FILE):
                with open(SVM_MODEL_FILE, 'rb') as f:
                    pretrained_classifier = pickle.load(f)
                
                # 复制预训练模型的组件
                self.vectorizer = pretrained_classifier.vectorizer
                self.classifier = pretrained_classifier.classifier
                self.feature_extractor = pretrained_classifier.feature_extractor
                self.is_trained = True
                
                print(f"✅ 已加载COLDataset预训练SVM模型")
                print(f"   - 模型文件: {SVM_MODEL_FILE}")
                print(f"   - 特征维度: {self.vectorizer.max_features}")
            else:
                print(f"⚠️  COLDataset SVM模型未找到，将使用启发式规则")
                
        except Exception as e:
            print(f"❌ 加载COLDataset SVM模型失败: {e}")
            print(f"   - 将回退到启发式规则")
    
    def prepare_features(self, texts: List[str]) -> np.ndarray:
        """准备特征矩阵"""
        # TF-IDF特征
        tfidf_features = self.vectorizer.fit_transform(texts).toarray()
        
        # 语境特征
        context_features = []
        for text in texts:
            features = self.feature_extractor.extract_features(text)
            context_features.append(list(features.values()))
        
        context_features = np.array(context_features)
        
        # 合并特征
        combined_features = np.hstack([tfidf_features, context_features])
        return combined_features
    
    def train(self, texts: List[str], labels: List[int]):
        """训练SVM分类器"""
        features = self.prepare_features(texts)
        self.classifier.fit(features, labels)
        self.is_trained = True
    
    def predict_bias_probability(self, text: str) -> float:
        """预测偏见概率"""
        if not self.is_trained:
            # 如果没有训练，使用简单的启发式规则
            return self._heuristic_prediction(text)
        
        try:
            # 使用预训练模型进行预测
            features = self.vectorizer.transform([text]).toarray()
            
            # 添加语境特征
            context_features = self.feature_extractor.extract_features(text)
            context_array = np.array([list(context_features.values())])
            
            # 合并特征
            combined_features = np.hstack([features, context_array])
            
            proba = self.classifier.predict_proba(combined_features)[0]
            return proba[1] if len(proba) > 1 else 0.0
            
        except Exception as e:
            print(f"⚠️  SVM预测出错，回退到启发式规则: {e}")
            return self._heuristic_prediction(text)
    
    def _heuristic_prediction(self, text: str) -> float:
        """启发式偏见预测"""
        features = self.feature_extractor.extract_features(text)
        score = 0.0
        
        # 根据特征计算偏见分数
        score += features['absolute_ratio'] * 0.3
        score += features['comparison_ratio'] * 0.3
        score += features['emotional_ratio'] * 0.2
        score += min(features['exclamation_ratio'] * 2, 0.2)
        
        return min(score, 1.0)

class SentimentAnalyzer:
    """情感倾向分析器 - COLDataset增强版"""
    
    def __init__(self):
        # 默认词典
        self.positive_words = [
            '好', '棒', '优秀', '杰出', '出色', '优异', '卓越',
            '强大', '聪明', '智慧', '能干', '厉害', '了不起'
        ]
        
        self.negative_words = [
            '差', '坏', '糟糕', '愚蠢', '笨', '无能', '低劣',
            '失败', '落后', '不行', '废物', '垃圾', '没用'
        ]
        
        self.extreme_words = [
            '极其', '非常', '特别', '超级', '十分', '相当',
            '太', '很', '极度', '异常', '格外'
        ]
        
        # 尝试加载COLDataset情感词典
        self._load_coldataset_sentiment_words()
    
    def _load_coldataset_sentiment_words(self):
        """加载COLDataset训练的情感词典"""
        try:
            if os.path.exists(SENTIMENT_WORDS_FILE):
                with open(SENTIMENT_WORDS_FILE, 'r', encoding='utf-8') as f:
                    coldataset_sentiment = json.load(f)
                
                # 使用COLDataset词典（与默认词典合并）
                if coldataset_sentiment.get('positive_words'):
                    self.positive_words.extend(coldataset_sentiment['positive_words'])
                if coldataset_sentiment.get('negative_words'):
                    self.negative_words.extend(coldataset_sentiment['negative_words'])
                if coldataset_sentiment.get('extreme_words'):
                    self.extreme_words = coldataset_sentiment['extreme_words']
                
                # 去重
                self.positive_words = list(set(self.positive_words))
                self.negative_words = list(set(self.negative_words))
                self.extreme_words = list(set(self.extreme_words))
                
                print(f"✅ 已加载COLDataset情感词典")
                print(f"   - 正面词汇: {len(self.positive_words)}个")
                print(f"   - 负面词汇: {len(self.negative_words)}个")
                print(f"   - 极端词汇: {len(self.extreme_words)}个")
            else:
                print(f"⚠️  COLDataset情感词典未找到，使用默认词典")
                
        except Exception as e:
            print(f"❌ 加载COLDataset情感词典失败: {e}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """分析情感倾向"""
        # 计算正面词汇分数
        positive_count = sum(1 for word in self.positive_words if word in text)
        negative_count = sum(1 for word in self.negative_words if word in text)
        extreme_count = sum(1 for word in self.extreme_words if word in text)
        
        total_words = len(jieba.lcut(text))
        
        positive_ratio = positive_count / max(total_words, 1)
        negative_ratio = negative_count / max(total_words, 1)
        extreme_ratio = extreme_count / max(total_words, 1)
        
        # 计算情感强度
        sentiment_intensity = abs(positive_ratio - negative_ratio) + extreme_ratio
        
        # 计算情感极性
        if positive_count > negative_count:
            sentiment_polarity = 1  # 正面
        elif negative_count > positive_count:
            sentiment_polarity = -1  # 负面
        else:
            sentiment_polarity = 0  # 中性
        
        return {
            'intensity': sentiment_intensity,
            'polarity': sentiment_polarity,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'extreme_ratio': extreme_ratio
        }

class FairnessRuleChecker:
    """增强版公平性规则检查器"""
    
    def __init__(self):
        # 基础公平性原则
        self.fairness_principles = {
            'gender': '性别不决定能力、智力或适应性',
            'race': '种族不决定智力、品德或能力',
            'region': '地域不决定个人素质、教育水平或品德',
            'age': '年龄不应成为能力判断的唯一标准',
            'occupation': '职业无高低贵贱，都值得尊重'
        }
        
        # 详细的偏见模式检测规则
        self.bias_patterns = {
            # 种族偏见模式
            'race_intelligence': {
                'patterns': [
                    r'(\w*族|黑人|白人|犹太人|穆斯林|阿拉伯人|印度人|中国人|日本人|韩国人).{0,10}(智力|智商|聪明|愚蠢|笨)',
                    r'(智力|智商|聪明|愚蠢|笨).{0,10}(\w*族|黑人|白人|犹太人|穆斯林|阿拉伯人|印度人|中国人|日本人|韩国人)',
                    r'(\w*族|黑人|白人).{0,20}(比|超过|不如|优于|劣于).{0,20}(智力|智商|聪明|能力)',
                    r'(智力|智商|能力).{0,20}(比|超过|不如|优于|劣于).{0,20}(\w*族|黑人|白人)'
                ],
                'severity': 0.9,
                'description': '种族智力比较偏见',
                'violation_type': 'racial_intelligence_bias'
            },
            
            'race_character': {
                'patterns': [
                    r'(\w*族|黑人|白人|犹太人|穆斯林).{0,10}(贪婪|暴力|懒惰|狡猾|诚实|勤劳)',
                    r'(\w*族|黑人|白人|犹太人|穆斯林).{0,5}(都|全部|总是|永远).{0,10}(贪婪|暴力|懒惰|狡猾)',
                    r'(犹太人|穆斯林|阿拉伯人).{0,10}(恐怖|危险|威胁)'
                ],
                'severity': 0.85,
                'description': '种族性格刻板印象',
                'violation_type': 'racial_stereotype'
            },
            
            # 性别偏见模式 - 增强版
            'gender_ability': {
                'patterns': [
                    # 原有模式
                    r'(女性|女人|女).{0,10}(天生|生来|就是).{0,10}(不适合|不能|无法).{0,10}(领导|管理|决策)',
                    r'(男性|男人|男).{0,10}(比|超过|优于).{0,10}(女性|女人|女).{0,10}(能力|智力|理性)',
                    r'(女性|女人|女).{0,10}(缺乏|没有|不具备).{0,10}(逻辑|理性|决策能力)',
                    r'(女性|女人|女).{0,5}(都|总是|永远).{0,10}(感性|情绪化|依赖)',
                    
                    # 新增强化模式
                    r'(男性|男人|男的).{0,20}(比|超过|优于|强于|高于).{0,20}(女性|女人|女的).{0,20}(工作|能力|表现|成就|智力|理性)',
                    r'(女性|女人|女的).{0,20}(比|超过|优于|强于|高于).{0,20}(男性|男人|男的).{0,20}(工作|能力|表现|成就|智力|理性)',
                    r'(男性|男人|男的).{0,10}(工作|职场|事业).{0,10}(能力|表现|成就).{0,10}(普遍|总体|整体).{0,10}(强于|优于|超过|高于).{0,10}(女性|女人|女的)',
                    r'(女性|女人|女的).{0,10}(工作|职场|事业).{0,10}(能力|表现|成就).{0,10}(普遍|总体|整体).{0,10}(弱于|劣于|不如|低于).{0,10}(男性|男人|男的)',
                    r'数据显示.{0,20}(男性|男人|男的).{0,20}(工作|职场|事业).{0,10}(能力|表现|成就|智力).{0,10}(普遍|总体|整体).{0,10}(强于|优于|超过|高于).{0,10}(女性|女人|女的)',
                    r'研究表明.{0,20}(男性|男人|男的).{0,20}(比|超过|优于|强于).{0,20}(女性|女人|女的).{0,20}(工作|能力|智力|理性)',
                    r'事实证明.{0,20}(男性|男人|男的).{0,20}(工作|职场).{0,10}(能力|表现).{0,10}(更|普遍).{0,10}(强|好|优秀)',
                    r'(男性|女性).{0,10}(天生|本性|基因).{0,10}(就|更).{0,10}(适合|擅长|不适合).{0,10}(工作|职业|领导|管理)',
                    r'(男性|女性).{0,5}(都|全部|普遍|总体).{0,10}(比|超过|不如).{0,10}(女性|男性).{0,10}(聪明|理性|感性|情绪化)'
                ],
                'severity': 0.9,  # 提高严重程度
                'description': '性别能力偏见',
                'violation_type': 'gender_ability_bias'
            },
            
            # 地域偏见模式
            'regional_quality': {
                'patterns': [
                    r'(东北|南方|北方|河南|山东).{0,5}人.{0,10}(都|总是|永远).{0,10}(粗鲁|野蛮|没素质)',
                    r'(农村|乡下|山区).{0,5}人.{0,10}(落后|愚昧|没文化)',
                    r'(某某省|某某市).{0,5}人.{0,10}(素质|教育|文明).{0,10}(低|差|不行)',
                    r'(城市|发达地区).{0,5}人.{0,10}(比|优于|超过).{0,10}(农村|落后地区)'
                ],
                'severity': 0.7,
                'description': '地域素质偏见',
                'violation_type': 'regional_bias'
            },
            
            # 比较偏见模式 - 大幅扩展
            'comparative_bias': {
                'patterns': [
                    # 原有模式
                    r'(\w*族|\w*人|男|女).{0,10}(比|超过|优于|强于|高于).{0,10}(\w*族|\w*人|男|女).{0,10}(智力|能力|素质)',
                    r'(\w*族|\w*人|男|女).{0,10}(不如|劣于|弱于|低于).{0,10}(\w*族|\w*人|男|女)',
                    r'(智力|能力|素质).{0,20}(排序|等级|高低).{0,20}(\w*族|\w*人)',
                    
                    # 职业能力比较
                    r'(男性|女性|男人|女人).{0,15}(在|做|从事).{0,10}(程序员|工程师|医生|教师|护士|秘书|销售|管理).{0,10}(比|超过|优于|不如|劣于).{0,10}(女性|男性|女人|男人)',
                    r'(程序员|工程师|科学家|领导).{0,5}(就是|都是|应该是|适合).{0,5}(男性|男人)',
                    r'(护士|秘书|幼师|客服).{0,5}(就是|都是|应该是|适合).{0,5}(女性|女人)',
                    
                    # 学科能力比较
                    r'(男性|女性|男生|女生).{0,10}(理科|数学|物理|化学|工程|计算机).{0,10}(天赋|能力|天生|更).{0,10}(强|好|差|弱)',
                    r'(男性|女性|男生|女生).{0,10}(文科|语文|英语|艺术|音乐).{0,10}(天赋|能力|天生|更).{0,10}(强|好|差|弱)',
                    
                    # 智力和认知比较
                    r'(男性|女性|男人|女人).{0,10}(逻辑思维|空间想象|记忆力|语言能力).{0,10}(比|超过|优于|强于|不如|劣于).{0,10}(女性|男性|女人|男人)',
                    r'(男性|女性).{0,5}(天生|生来).{0,10}(理性|感性|情绪化|冷静)',
                    
                    # 种族智力比较
                    r'(亚洲人|黄种人|白种人|黑种人|犹太人).{0,15}(智商|IQ|智力|聪明程度).{0,10}(比|超过|优于|高于|低于|不如).{0,15}(其他|别的)',
                    r'(中国人|日本人|韩国人|印度人|美国人|欧洲人).{0,15}(数学|科学|技术|创新).{0,10}(能力|天赋|水平).{0,10}(更|比较|相对)',
                    
                    # 地域素质比较
                    r'(北方人|南方人|城市人|农村人|发达地区|落后地区).{0,15}(素质|教育水平|文明程度|见识).{0,10}(比|超过|优于|高于|低于|不如)',
                    r'(某某省|某某市|某某县).{0,5}人.{0,10}(素质|教育|文化).{0,10}(就是|都是|普遍).{0,5}(低|差|高|好)',
                    
                    # 年龄比较
                    r'(年轻人|老年人|中年人).{0,10}(比|超过|不如).{0,10}(老年人|年轻人|中年人).{0,10}(有活力|有经验|懂技术|跟得上时代)',
                    r'(90后|00后|80后|70后).{0,10}(比|超过|不如).{0,10}(其他|别的).{0,5}代.{0,10}(能干|靠谱|负责)',
                    
                    # 绝对化偏见表述
                    r'(所有|全部|每个|任何).{0,5}(男性|女性|黑人|白人|亚洲人|\w+人).{0,10}(都|全都|全部|统统).{0,10}(是|有|具备|缺乏)',
                    r'(男性|女性|\w+人|\w+族).{0,5}(永远|总是|从来|绝对|肯定).{0,10}(比|超过|不如|无法)',
                    
                    # 刻板印象强化
                    r'(这就是|难怪|果然|典型的).{0,10}(男性|女性|\w+人|\w+族).{0,10}(的|都)',
                    r'(男性|女性|\w+人).{0,5}(就是|果然|确实|真的).{0,10}(比|超过|不如|无法|不会)',
                    
                    # 基因决定论
                    r'(男性|女性|\w+族|\w+人).{0,10}(基因|DNA|血统|遗传).{0,10}(决定|注定|天生|本来).{0,10}(智力|能力|性格)',
                    r'(智力|能力|性格|品质).{0,10}(是|由|靠).{0,10}(基因|血统|遗传|天生).{0,10}(决定|注定)',
                    
                    # 进化论偏见
                    r'(男性|女性).{0,10}(进化|演化).{0,10}(过程中|历史上).{0,10}(形成|发展|具备).{0,10}(优势|劣势|特长|缺陷)'
                ],
                'severity': 0.85,  # 提高严重性
                'description': '群体比较偏见',
                'violation_type': 'comparative_bias'
            },
            
            # 新增：职业刻板印象
            'occupational_stereotype': {
                'patterns': [
                    r'(男性|男人).{0,10}(天生|就是|应该|适合).{0,10}(当|做|从事).{0,10}(CEO|领导|工程师|程序员|医生|律师|科学家)',
                    r'(女性|女人).{0,10}(天生|就是|应该|适合).{0,10}(当|做|从事).{0,10}(护士|教师|秘书|客服|销售|幼师|家庭主妇)',
                    r'(护士|教师|秘书|幼师).{0,5}(应该|最好|就是).{0,5}(女性|女人)',
                    r'(工程师|程序员|CEO|科学家).{0,5}(应该|最好|就是).{0,5}(男性|男人)',
                    r'(女性|女人).{0,10}(不适合|无法胜任|做不了).{0,10}(技术|理工|编程|领导|管理)',
                    r'(男性|男人).{0,10}(不适合|无法胜任|做不了).{0,10}(照顾|护理|教育|服务)'
                ],
                'severity': 0.8,
                'description': '职业性别刻板印象',
                'violation_type': 'occupational_stereotype'
            },
            
            # 新增：外貌和身体特征偏见
            'physical_appearance_bias': {
                'patterns': [
                    r'(男性|女性|男人|女人).{0,10}(长得|外貌|身材|体型).{0,10}(比|超过|不如).{0,10}(女性|男性|女人|男人)',
                    r'(黑人|白人|亚洲人|\w+人).{0,10}(长相|外貌|五官|肤色).{0,10}(比|超过|更|不如).{0,10}(好看|丑|美|难看)',
                    r'(男性|女性).{0,10}(身高|体重|体格).{0,10}(决定|影响|体现).{0,10}(能力|智力|性格|地位)',
                    r'(高个子|矮个子|胖子|瘦子).{0,10}(就是|都是|普遍).{0,10}(笨|聪明|懒|勤奋)'
                ],
                'severity': 0.7,
                'description': '外貌和身体特征偏见',
                'violation_type': 'physical_appearance_bias'
            },
            
            # 新增：经济和社会地位偏见
            'socioeconomic_bias': {
                'patterns': [
                    r'(富人|穷人|有钱人|没钱人).{0,10}(就是|都是|天生|注定).{0,10}(聪明|笨|勤奋|懒|有素质|没素质)',
                    r'(农村人|城里人|山区人).{0,10}(见识|眼界|素质|教育).{0,10}(比|超过|不如|低于).{0,10}(城里人|农村人)',
                    r'(名校|985|211).{0,5}(毕业|出身).{0,10}(就是|肯定|一定).{0,10}(比|超过|优于).{0,10}(普通|一般)',
                    r'(学历|文凭|出身).{0,10}(决定|代表|体现).{0,10}(能力|智力|品德|价值)'
                ],
                'severity': 0.75,
                'description': '经济和社会地位偏见',
                'violation_type': 'socioeconomic_bias'
            },
            
            # 新增：隐性偏见检测模式 - 针对用户案例
            'implicit_bias_patterns': {
                'patterns': [
                    # 科学包装的性别偏见 - 用户的具体案例
                    r'(根据|据|基于).{0,10}(数据|研究|统计|调查).{0,10}(显示|表明|证明|发现).{0,20}(在|从).{0,10}(某些|这些|部分).{0,10}(工作|职业|行业).{0,10}(和|或).{0,10}(行业|领域).{0,10}(中|里).{0,20}(男性|男人).{0,20}(确实|的确|往往|倾向于|通常).{0,20}(表现出|显示出|展现出).{0,20}(了|更).{0,10}(高|强|好|优秀).{0,10}(的|).{0,10}(工作能力|能力|成就)',
                    
                    # 生理心理差异包装的偏见
                    r'(这一|这种|此类).{0,10}(现象|情况|差异|分布).{0,20}(可能|或许|也许|大概).{0,20}(与|和|因为|由于).{0,20}(性别|男女).{0,20}(间|之间|的).{0,20}(生理|心理|认知|身体).{0,20}(差异|不同|因素).{0,20}(有关|相关|导致|造成)',
                    
                    # 认知能力差异暗示职业分布
                    r'(不同|两种).{0,10}(性别|男女).{0,20}(在|做).{0,10}(某些|这些|部分).{0,10}(认知|思维|智力).{0,10}(能力|方面).{0,10}(上|方面).{0,20}(存在|有).{0,10}(差异|不同).{0,20}(这|这种|此).{0,10}(可能|或许|也许).{0,10}(解释|说明).{0,20}(了|为什么).{0,20}(某些|这些|部分).{0,10}(职业|工作).{0,10}(领域|方面).{0,10}(的|).{0,10}(性别|男女).{0,10}(分布|比例).{0,10}(不均|失衡)',
                    
                    # 委婉的比较偏见 - "虽然...但是"句式
                    r'(虽然|尽管|虽说).{0,20}(个体|个人|每个人).{0,10}(差异|不同).{0,20}(但|但是|然而|不过).{0,20}(统计|数据|研究|观察).{0,20}(表明|显示|发现).{0,50}(在|从).{0,10}(需要|要求).{0,10}(体力|力量).{0,10}(和|或).{0,10}(逻辑|理性).{0,10}(思维|能力).{0,10}(的|).{0,10}(工作|职业).{0,10}(中|里).{0,20}(男性|男人).{0,20}(往往|通常|倾向于).{0,20}(有|具有).{0,10}(更|比较).{0,10}(好|强|优秀).{0,10}(的|).{0,10}(表现|成绩)',
                    
                    # 进化论包装的偏见
                    r'(从|根据).{0,10}(进化|演化).{0,10}(心理学|生物学|角度).{0,20}(来看|而言|观点).{0,50}(男性|男人).{0,50}(在|做).{0,20}(狩猎|竞争|战斗).{0,20}(和|或).{0,10}(竞争|斗争).{0,10}(中|的).{0,10}(历史|传统).{0,10}(角色|作用).{0,30}(可能|或许|也许).{0,20}(使|让|导致).{0,20}(他们|其|这些人).{0,30}(在|做).{0,30}(现代|当今|现在).{0,10}(商业|工作|职场).{0,20}(环境|领域).{0,10}(中|里).{0,20}(更具|具有|有).{0,10}(优势|长处)',
                    
                    # 用数据权威性包装的偏见
                    r'(数据|研究|统计|科学|生物学|心理学).{0,20}(显示|表明|证明|发现|研究).{0,50}(男性|女性|某族|某地区人).{0,50}(确实|的确|往往|倾向于|通常|更容易|更可能).{0,30}(在|做|具有|表现出).{0,30}(某些|这些|特定).{0,10}(方面|领域|能力).{0,20}(更|比较|相对).{0,10}(强|好|优秀|有优势)',
                    
                    # 职业分布合理化偏见
                    r'(某些|这些|部分).{0,10}(职业|工作|行业).{0,10}(领域|方面).{0,10}(的|).{0,10}(性别|种族|地域).{0,10}(分布|比例|构成).{0,10}(不均|失衡|差异).{0,30}(可能|或许|也许).{0,20}(反映|体现|说明).{0,20}(了|出).{0,20}(不同|各种).{0,10}(群体|人群|性别|种族).{0,20}(在|做).{0,20}(某些|特定|这些).{0,10}(能力|技能|素质|方面).{0,10}(上|方面).{0,10}(的|).{0,10}(差异|不同|特点)',
                    
                    # 委婉语包装的偏见
                    r'(可能|或许|也许|大概|似乎|看起来).{0,20}(与|和|因为|由于).{0,20}(性别|种族|地域|年龄).{0,20}(差异|不同|因素|背景).{0,20}(有关|相关|导致|造成).{0,30}(男性|女性|某族|某地区人).{0,30}(在|做).{0,30}(某些|这些|特定).{0,10}(领域|方面|工作).{0,20}(表现|成就|能力).{0,10}(更|比较|相对).{0,10}(好|强|优秀)',
                    
                    # 暗示因果关系的偏见
                    r'(这|这种|此类).{0,10}(差异|现象|分布|不均).{0,30}(暗示|提示|表明|说明).{0,30}(可能|或许).{0,20}(存在|有).{0,20}(某些|一些).{0,10}(内在|根本|本质).{0,10}(的|).{0,10}(差异|不同|因素)',
                    
                    # 用"客观事实"包装的偏见
                    r'(客观|事实|现实|实际).{0,10}(上|来说|而言).{0,30}(男性|女性|某族|某地区人).{0,30}(在|做).{0,30}(某些|这些|特定).{0,10}(方面|领域|能力).{0,20}(确实|的确|往往|通常).{0,20}(表现|显示).{0,10}(出|得).{0,20}(更|比较|相对).{0,10}(好|强|优秀|有优势)'
                ],
                'severity': 0.8,  # 隐性偏见严重性较高
                'description': '隐性偏见模式',
                'violation_type': 'implicit_bias'
            }
        }
        
        # 上下文增强检测
        self.context_enhancers = {
            'comparison_words': ['比', '超过', '不如', '优于', '劣于', '强于', '弱于', '高于', '低于'],
            'absolute_words': ['都', '全部', '所有', '每个', '总是', '永远', '从来', '绝对', '肯定', '一定'],
            'inherent_words': ['天生', '生来', '注定', '本质', '基因', '血统', '天性']
        }
    
    def check_fairness(self, text: str, bias_category: str = None) -> Dict[str, any]:
        """增强版公平性检查"""
        violations = []
        
        # 检查所有偏见模式
        for pattern_name, pattern_config in self.bias_patterns.items():
            for pattern in pattern_config['patterns']:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    violation = {
                        'type': pattern_config['violation_type'],
                        'pattern_name': pattern_name,
                        'matched_text': match.group(),
                        'matched_position': (match.start(), match.end()),
                        'severity': pattern_config['severity'],
                        'description': pattern_config['description'],
                        'pattern': pattern
                    }
                    
                    # 上下文分析增强严重性
                    context_multiplier = self._analyze_context(text, match)
                    violation['context_enhanced_severity'] = min(1.0, violation['severity'] * context_multiplier)
                    
                    violations.append(violation)
        
        # 按严重性排序
        violations.sort(key=lambda x: x['context_enhanced_severity'], reverse=True)
        
        # 计算总体评分
        max_severity = max([v['context_enhanced_severity'] for v in violations], default=0)
        total_severity = sum([v['context_enhanced_severity'] for v in violations])
        
        # 检查是否违反公平性原则
        violated_principles = self._check_violated_principles(violations)
        
        return {
            'violations': violations,
            'violation_count': len(violations),
            'max_severity': max_severity,
            'total_severity': total_severity,
            'violated_rules': violated_principles,
            'total_violations': len(violations),
            'is_biased': max_severity >= 0.5,  # 降低严重性阈值，更敏感
            'bias_level': self._determine_bias_level(max_severity)
        }
    
    def _analyze_context(self, text: str, match) -> float:
        """分析上下文以增强严重性判断"""
        multiplier = 1.0
        match_start, match_end = match.span()
        
        # 获取匹配文本前后的上下文
        context_before = text[max(0, match_start-50):match_start]
        context_after = text[match_end:min(len(text), match_end+50)]
        full_context = context_before + match.group() + context_after
        
        # 检查绝对化词汇
        absolute_count = sum(1 for word in self.context_enhancers['absolute_words'] 
                           if word in full_context)
        if absolute_count > 0:
            multiplier += 0.1 * absolute_count
        
        # 检查比较词汇
        comparison_count = sum(1 for word in self.context_enhancers['comparison_words'] 
                             if word in full_context)
        if comparison_count > 0:
            multiplier += 0.15 * comparison_count
        
        # 检查天性词汇
        inherent_count = sum(1 for word in self.context_enhancers['inherent_words'] 
                           if word in full_context)
        if inherent_count > 0:
            multiplier += 0.2 * inherent_count
        
        return min(1.5, multiplier)  # 最大增强倍数为1.5
    
    def _check_violated_principles(self, violations: List[Dict]) -> List[str]:
        """检查违反的公平性原则"""
        violated = []
        
        for violation in violations:
            if violation['type'] in ['racial_intelligence_bias', 'racial_stereotype']:
                violated.append(self.fairness_principles['race'])
            elif violation['type'] in ['gender_ability_bias', 'gender_role_stereotype']:
                violated.append(self.fairness_principles['gender'])
            elif violation['type'] == 'regional_bias':
                violated.append(self.fairness_principles['region'])
        
        return list(set(violated))  # 去重
    
    def _determine_bias_level(self, max_severity: float) -> str:
        """确定偏见等级"""
        if max_severity >= 0.8:
            return 'severe'
        elif max_severity >= 0.6:
            return 'high' 
        elif max_severity >= 0.4:
            return 'medium'
        elif max_severity >= 0.3:
            return 'low'
        else:
            return 'minimal'

class BiasReport:
    """偏见报告生成器"""
    
    def __init__(self):
        pass
    
    def generate_report(self, text: str, detection_results: Dict) -> Dict:
        """生成偏见检测报告"""
        report = {
            'input_text': text,
            'timestamp': self._get_timestamp(),
            'detection_flow': [],
            'final_result': {
                'is_biased': False,
                'confidence': 0.0,
                'bias_types': [],
                'severity': 'none'
            },
            'detailed_analysis': {
                'sensitive_words': detection_results.get('sensitive_words', {}),
                'svm_prediction': detection_results.get('svm_prediction', {}),
                'sentiment_analysis': detection_results.get('sentiment_analysis', {}),
                'fairness_check': detection_results.get('fairness_check', {})
            },
            'recommendations': []
        }
        
        # 构建检测流程
        flow = detection_results.get('flow', [])
        report['detection_flow'] = flow
        
        # 确定最终结果
        if detection_results.get('final_decision') == 'biased':
            report['final_result']['is_biased'] = True
            report['final_result']['confidence'] = detection_results.get('confidence', 0.0)
            report['final_result']['bias_types'] = list(detection_results.get('sensitive_words', {}).keys())
            report['final_result']['severity'] = self._determine_severity(detection_results)
            
            # 生成建议
            report['recommendations'] = self._generate_recommendations(detection_results)
        
        return report
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def _determine_severity(self, results: Dict) -> str:
        """确定严重程度"""
        sentiment = results.get('sentiment_analysis', {})
        fairness = results.get('fairness_check', {})
        
        max_severity = fairness.get('max_severity', 0)
        sentiment_intensity = sentiment.get('intensity', 0)
        
        if max_severity >= 0.6 or sentiment_intensity >= 0.5:
            return 'high'
        elif max_severity >= 0.4 or sentiment_intensity >= 0.3:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        fairness = results.get('fairness_check', {})
        violations = fairness.get('violations', [])
        
        for violation in violations:
            violation_type = violation.get('type', '')
            if violation_type == 'racial_intelligence_bias':
                recommendations.append('避免将智力、能力与种族联系，每个人的能力都是独特的')
            elif violation_type == 'racial_stereotype':
                recommendations.append('避免种族刻板印象，不要将特定特征归因于整个种族群体')
            elif violation_type == 'gender_ability_bias':
                recommendations.append('避免基于性别判断能力，能力与性别无关')
            elif violation_type == 'regional_bias':
                recommendations.append('避免地域偏见，个人素质与出生地无关')
            elif violation_type == 'comparative_bias':
                recommendations.append('避免对不同群体进行能力或素质比较，每个人都是独特的')
            # 保留旧的类型兼容性
            elif violation_type == 'stereotyping':
                recommendations.append('避免使用绝对化的表述，如"所有"、"都是"等词汇')
            elif violation_type == 'discrimination':
                recommendations.append('移除歧视性语言，使用更包容的表达方式')
            elif violation_type == 'prejudice':
                recommendations.append('避免基于刻板印象的预判，保持客观中立')
        
        if not recommendations:
            recommendations.append('建议使用更中性、客观的语言表达')
        
        return recommendations

class TraditionalBiasDetector:
    """传统偏见检测系统主类 - COLDataset增强版"""
    
    def __init__(self):
        print("🚀 初始化传统偏见检测系统 (COLDataset增强版)")
        print("-" * 50)
        
        # 初始化各个组件
        self.sensitive_matcher = SensitiveWordMatcher()
        self.svm_classifier = SVMBiasClassifier()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.fairness_checker = FairnessRuleChecker()
        self.report_generator = BiasReport()
        
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print("-" * 50)
        print("✅ 传统偏见检测系统初始化完成")
        print("🔧 增强功能:")
        print("   - COLDataset训练的敏感词词典")
        print("   - COLDataset训练的SVM分类器")  
        print("   - COLDataset增强的情感分析器")
        print("   - 智能公平性规则检查")
        print()
        
    def detect_bias(self, text: str, threshold_svm: float = 0.6, 
                   threshold_sentiment: float = 0.4) -> Dict:
        """
        主检测函数，按照流程图执行偏见检测
        
        Args:
            text: 输入文本
            threshold_svm: SVM判断阈值
            threshold_sentiment: 情感倾向阈值
            
        Returns:
            包含检测结果和报告的字典
        """
        self.logger.info(f"开始检测文本: {text[:50]}...")
        
        detection_results = {
            'flow': [],
            'sensitive_words': {},
            'svm_prediction': {},
            'sentiment_analysis': {},
            'fairness_check': {},
            'final_decision': 'passed',
            'confidence': 0.0
        }
        
        # 步骤1：敏感词正则匹配
        self.logger.info("步骤1: 敏感词匹配")
        sensitive_matches = self.sensitive_matcher.match(text)
        detection_results['sensitive_words'] = sensitive_matches
        detection_results['flow'].append('敏感词匹配')
        
        if not sensitive_matches:
            self.logger.info("未发现敏感词，通过检测")
            detection_results['flow'].append('无敏感词 -> 通过')
            # 设置高置信度表示安全
            detection_results['confidence'] = 0.95
            return self._finalize_results(text, detection_results)
        
        self.logger.info(f"发现敏感词: {sensitive_matches}")
        detection_results['flow'].append('发现敏感词 -> 进入SVM判断')
        
        # 步骤2：SVM偏见判断
        self.logger.info("步骤2: SVM偏见判断")
        bias_probability = self.svm_classifier.predict_bias_probability(text)
        detection_results['svm_prediction'] = {
            'probability': bias_probability,
            'threshold': threshold_svm
        }
        detection_results['flow'].append('SVM判断')
        
        # 继续进行后续步骤，无论SVM结果如何
        if bias_probability < threshold_svm:
            self.logger.info(f"SVM判断概率 {bias_probability:.3f} < {threshold_svm}")
            detection_results['flow'].append('SVM判断: 概率较低')
        else:
            self.logger.info(f"SVM判断可能存在偏见 (概率: {bias_probability:.3f})")
            detection_results['flow'].append('SVM判断: 潜在偏见')
        
        # 步骤3：情感倾向分析 (无论SVM结果都执行)
        self.logger.info("步骤3: 情感倾向分析")
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(text)
        detection_results['sentiment_analysis'] = sentiment_result
        detection_results['flow'].append('情感倾向分析')
        
        if sentiment_result['intensity'] < threshold_sentiment:
            self.logger.info(f"情感强度 {sentiment_result['intensity']:.3f} < {threshold_sentiment}")
            detection_results['flow'].append('情感强度: 较弱')
        else:
            self.logger.info(f"情感倾向强烈 (强度: {sentiment_result['intensity']:.3f})")
            detection_results['flow'].append('情感强度: 较强')
        
        # 无论前面步骤结果如何，都进行公平性检查
        detection_results['flow'].append('继续进行公平性检查')
        
        # 步骤4：公平性规则检查 (并行检测策略)
        self.logger.info("步骤4: 公平性规则检查")
        bias_category = list(sensitive_matches.keys())[0] if sensitive_matches else 'general'
        fairness_result = self.fairness_checker.check_fairness(text, bias_category)
        detection_results['fairness_check'] = fairness_result
        detection_results['flow'].append('公平性规则检查')
        
        # 🚀 新策略：并行加权融合判断
        # 如果公平性检查发现严重违规（≥0.5），直接判定为偏见，不依赖前序步骤
        fairness_severity = fairness_result.get('max_severity', 0)
        
        if fairness_severity >= 0.5:
            # 严重违规：直接判定为偏见
            self.logger.info(f"公平性检查发现严重违规，直接判定为偏见 (严重性: {fairness_severity:.3f})")
            detection_results['final_decision'] = 'biased'
            detection_results['confidence'] = min(fairness_severity + 0.2, 1.0)  # 增强置信度
            detection_results['flow'].append(f"严重违规({fairness_result.get('bias_level', 'unknown')}) -> 直接判定偏见")
        elif fairness_result.get('is_biased', False):
            # 中等违规：需要结合其他指标
            self.logger.info(f"公平性检查发现违规，判定为偏见文本 (严重性: {fairness_severity:.3f})")
            detection_results['final_decision'] = 'biased'
            detection_results['confidence'] = min(
                bias_probability + sentiment_result['intensity'] + fairness_severity, 1.0
            )
            detection_results['flow'].append(f"发现{fairness_result.get('bias_level', 'unknown')}级别违规 -> 生成偏见报告")
        else:
            self.logger.info("未发现公平性违规，通过检测")
            detection_results['flow'].append('公平性检查通过 -> 无偏见')
            # 计算安全内容的置信度：基于各项指标的反向得分
            safe_confidence = (
                (1.0 - bias_probability) * 0.4 +  # SVM安全度权重40%
                (1.0 - sentiment_result['intensity']) * 0.3 +  # 情感中性度权重30%
                (1.0 - fairness_severity) * 0.3  # 公平性安全度权重30%
            )
            detection_results['confidence'] = max(safe_confidence, 0.6)  # 最低0.6的安全置信度
        
        return self._finalize_results(text, detection_results)
    
    def _finalize_results(self, text: str, detection_results: Dict) -> Dict:
        """完成检测并生成报告"""
        # 生成检测报告
        report = self.report_generator.generate_report(text, detection_results)
        
        return {
            'detection_results': detection_results,
            'report': report,
            'summary': {
                'is_biased': detection_results['final_decision'] == 'biased',
                'confidence': detection_results.get('confidence', 0.0),
                'flow_steps': len(detection_results['flow']),
                'bias_types': list(detection_results.get('sensitive_words', {}).keys())
            }
        }
    
    def train_svm(self, training_data: List[Tuple[str, int]]):
        """训练SVM分类器"""
        texts, labels = zip(*training_data)
        self.svm_classifier.train(list(texts), list(labels))
        self.logger.info(f"SVM训练完成，使用 {len(training_data)} 个样本")

def main():
    """主函数，演示系统使用"""
    detector = TraditionalBiasDetector()
    
    # 测试用例
    test_cases = [
        "虽然数据显示在一般工作和行业中，男性表现出了更高的工作能力和成就，但这并不意味着在所有工作领域和所有女性中都表现不佳。",
        "这个问题很复杂，需要综合考虑多个因素。",
        "女性天生就不适合做程序员，这是科学事实。",
        "北方人都很豪爽，南方人都很精明。",
        "年轻人都不懂事，老年人都跟不上时代。"
    ]
    
    print("=" * 60)
    print("传统偏见检测系统演示")
    print("=" * 60)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {text}")
        print("-" * 50)
        
        result = detector.detect_bias(text)
        
        # 显示检测流程
        print("检测流程:")
        for step in result['detection_results']['flow']:
            print(f"  • {step}")
        
        # 显示最终结果
        summary = result['summary']
        print(f"\n最终结果: {'偏见文本' if summary['is_biased'] else '通过检测'}")
        if summary['is_biased']:
            print(f"置信度: {summary['confidence']:.3f}")
            print(f"偏见类型: {', '.join(summary['bias_types'])}")
        
        print("=" * 60)

if __name__ == "__main__":
    main()