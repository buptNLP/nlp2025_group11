# 传统机器学习偏见检测系统

## 1. 系统概述

本系统是一个基于传统机器学习的偏见检测系统，采用多层次检测策略，结合敏感词匹配、SVM分类、情感分析和公平性规则检查，实现对文本中偏见的全面检测。

### 1.1 核心特点

- 多层次检测：从词汇到语义再到规则
- 上下文感知：考虑文本的上下文信息
- 可解释性强：每个步骤都有明确的判断依据
- 灵活可配置：阈值和规则可调整
- 持续优化：支持模型和词典的更新

### 1.2 系统架构

```
输入文本
    ↓
敏感词匹配 → 无敏感词 → 通过检测
    ↓
SVM分类器 → 概率判断
    ↓
情感分析 → 情感强度评估
    ↓
公平性规则检查 → 最终决策
```

## 2. 核心组件

### 2.1 敏感词匹配器 (SensitiveWordMatcher)

#### 功能
- 使用预定义的敏感词词典进行匹配
- 支持多类别敏感词检测
- 实现模糊匹配和上下文分析

#### 词典结构
```json
{
    "gender": ["性别", "能力", "工作", ...],
    "race": ["种族", "能力", ...],
    "region": ["地域", "素质", ...]
}
```

### 2.2 SVM分类器 (SVMBiasClassifier)

#### 特征提取
- TF-IDF特征
- 语境特征：
  - 绝对化表述频率
  - 比较性词汇频率
  - 情感性词汇频率
  - 文本长度特征
  - 标点符号特征

#### 模型配置
- 核函数：RBF
- 支持概率预测
- 可调整阈值（默认0.6）

### 2.3 情感分析器 (SentimentAnalyzer)

#### 分析维度
- 正面词汇分数
- 负面词汇分数
- 程度词分数

#### 计算指标
- 情感强度（intensity）
- 情感极性（polarity）
- 正面/负面词汇比例
- 程度词比例

### 2.4 公平性规则检查器 (FairnessRuleChecker)

#### 基础公平性原则
- 性别不决定能力、智力或适应性
- 种族不决定智力、品德或能力
- 地域不决定个人素质、教育水平或品德
- 年龄不应成为能力判断的唯一标准
- 职业无高低贵贱，都值得尊重

#### 偏见模式检测
1. 种族偏见模式
   - 智力比较偏见
   - 性格刻板印象

2. 性别偏见模式
   - 能力偏见
   - 角色刻板印象

3. 地域偏见模式
   - 素质偏见
   - 文化偏见

4. 年龄偏见模式
   - 能力偏见
   - 代际偏见

5. 绝对化表述模式
   - 全称判断
   - 本质化表述

6. 比较偏见模式
   - 群体比较
   - 能力排序

## 3. 检测流程

### 3.1 敏感词匹配
```python
# 步骤1：敏感词正则匹配
sensitive_matches = sensitive_matcher.match(text)
if not sensitive_matches:
    return "通过检测"
```

### 3.2 SVM判断
```python
# 步骤2：SVM偏见判断
bias_probability = svm_classifier.predict_bias_probability(text)
if bias_probability < threshold_svm:
    return "概率较低"
```

### 3.3 情感分析
```python
# 步骤3：情感倾向分析
sentiment_result = sentiment_analyzer.analyze_sentiment(text)
if sentiment_result['intensity'] < threshold_sentiment:
    return "情感强度较弱"
```

### 3.4 公平性检查
```python
# 步骤4：公平性规则检查
fairness_result = fairness_checker.check_fairness(text)
if fairness_severity >= 0.8:
    return "严重违规"
```

## 4. 决策机制

### 4.1 严重性等级
- 严重（≥0.9）：直接判定为偏见
- 高（≥0.8）：需要重点关注
- 中（≥0.6）：需要结合其他指标
- 低（≥0.4）：轻微偏见
- 最小（<0.4）：基本无偏见

### 4.2 置信度计算
```python
safe_confidence = (
    (1.0 - bias_probability) * 0.4 +    # SVM安全度权重40%
    (1.0 - sentiment_intensity) * 0.3 + # 情感中性度权重30%
    (1.0 - fairness_severity) * 0.3     # 公平性安全度权重30%
)
```

## 5. 输出报告

### 5.1 报告结构
```json
{
    "input_text": "原始文本",
    "timestamp": "检测时间",
    "detection_flow": ["检测步骤"],
    "final_result": {
        "is_biased": false,
        "confidence": 0.0,
        "bias_types": [],
        "severity": "none"
    },
    "detailed_analysis": {
        "sensitive_words": {},
        "svm_prediction": {},
        "sentiment_analysis": {},
        "fairness_check": {}
    },
    "recommendations": []
}
```

### 5.2 改进建议
- 基于检测结果生成具体改进建议
- 针对不同类型的偏见提供相应的修改方案
- 包含示例和最佳实践

## 6. 使用示例

### 6.1 基本使用
```python
detector = TraditionalBiasDetector()
result = detector.detect_bias("输入文本")
```

### 6.2 自定义配置
```python
result = detector.detect_bias(
    text,
    threshold_svm=0.6,
    threshold_sentiment=0.4
)
```

## 7. 注意事项

1. 数据安全
   - 敏感词词典定期更新
   - 模型定期重训练
   - 检测结果安全存储

2. 性能优化
   - 使用缓存机制
   - 并行处理大量文本
   - 定期清理临时数据

3. 维护建议
   - 定期更新敏感词词典
   - 监控模型性能
   - 收集用户反馈
   - 优化检测规则

## 8. 未来改进

1. 模型优化
   - 引入深度学习模型
   - 优化特征提取
   - 提高检测准确率

2. 功能扩展
   - 支持多语言检测
   - 增加更多偏见类型
   - 提供API接口

3. 用户体验
   - 优化报告展示
   - 提供可视化界面
   - 增加批量处理功能 