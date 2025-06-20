import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from traditional_bias_detector import TraditionalBiasDetector
from coldataset_bias_trainer import COLDatasetBertTrainer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import jieba
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class BertBiasPredictor:
    """BERT偏见检测预测器"""
    
    def __init__(self, model_path='./coldataset_bias_bert_model'):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """加载训练好的BERT模型"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print("✅ BERT模型加载成功")
        except Exception as e:
            print(f"❌ BERT模型加载失败: {e}")
    
    def detect_bias(self, text):
        """检测文本偏见"""
        if not self.model or not self.tokenizer:
            return {'bias_types': [], 'confidence': 0.0}
        
        # 编码文本
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][prediction].item()
        
        # 返回结果（简化版，实际应该根据具体需求调整）
        if prediction == 1:  # 有偏见
            # 这里简化处理，实际应该有更复杂的逻辑来确定偏见类型
            return {
                'bias_types': ['general'],  # 简化为通用偏见
                'confidence': confidence
            }
        else:
            return {
                'bias_types': [],
                'confidence': confidence
            }

def load_test_data():
    """加载COLDataset测试集（CSV格式）"""
    df = pd.read_csv('COLDataset-main/COLDataset/test.csv', encoding='utf-8')
    # 使用正确的字段名：TEXT为文本内容，topic为偏见类型，label为是否有偏见
    test_data = []
    for _, row in df.iterrows():
        test_data.append({
            'text': row['TEXT'],
            'bias_type': row['topic'],  # race, gender, region等
            'has_bias': row['label']    # 1表示有偏见，0表示无偏见
        })
    return test_data

def evaluate_model(model, test_data, model_type):
    """评估模型性能 - 简化版：只评估偏见识别能力"""
    results = {
        'pred': [],
        'true': []
    }
    
    for item in tqdm(test_data, desc=f'评估{model_type}模型'):
        text = item['text']
        has_bias = item['has_bias']
        
        # 真实标签：1表示有偏见，0表示无偏见
        true_label = has_bias
        
        # 获取模型预测结果
        pred = model.detect_bias(text)
        
        if model_type == 'BERT':
            # BERT模型返回通用偏见检测结果
            detected_bias_types = pred['bias_types'] if pred['bias_types'] else []
            # 预测标签：检测到任何偏见类型就为1
            pred_label = 1 if len(detected_bias_types) > 0 else 0
                
        else:
            # 传统模型返回格式：{'summary': {'is_biased': bool, 'bias_types': list}}
            pred = model.detect_bias(text, threshold_svm=0.5)
            is_biased = pred['summary']['is_biased']
            # 预测标签：模型判定为偏见就为1
            pred_label = 1 if is_biased else 0
        
        results['pred'].append(pred_label)
        results['true'].append(true_label)
    
    return results

def plot_confusion_matrix(results, model_type):
    """绘制混淆矩阵 - 简化版"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    cm = confusion_matrix(results['true'], results['pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['非偏见', '偏见'],
               yticklabels=['非偏见', '偏见'],
               ax=ax)
    ax.set_title(f'{model_type}模型偏见检测混淆矩阵', fontsize=16)
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    
    plt.tight_layout()
    plt.savefig(f'{model_type}_bias_detection_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def calculate_metrics(results):
    """计算评估指标 - 简化版"""
    true = np.array(results['true'])
    pred = np.array(results['pred'])
    
    tp = np.sum((true == 1) & (pred == 1))
    fp = np.sum((true == 0) & (pred == 1))
    fn = np.sum((true == 1) & (pred == 0))
    tn = np.sum((true == 0) & (pred == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        '准确率': accuracy,
        '精确率': precision,
        '召回率': recall,
        'F1分数': f1,
        '误判率': false_positive_rate,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn
    }

def plot_metrics_comparison(bert_metrics, traditional_metrics):
    """绘制指标对比图 - 简化版"""
    metrics = ['准确率', '精确率', '召回率', 'F1分数', '误判率']
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bert_values = [bert_metrics[m] for m in metrics]
    traditional_values = [traditional_metrics[m] for m in metrics]
    
    ax.bar(x - width/2, bert_values, width, label='BERT模型')
    ax.bar(x + width/2, traditional_values, width, label='传统模型')
    
    ax.set_title('偏见检测模型性能对比', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45)
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('bias_detection_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 加载测试数据
    print("加载测试数据...")
    test_data = load_test_data()
    
    # 初始化模型
    print("初始化模型...")
    bert_model = BertBiasPredictor()
    traditional_model = TraditionalBiasDetector()
    
    # 评估BERT模型
    print("评估BERT模型...")
    bert_results = evaluate_model(bert_model, test_data, 'BERT')
    
    # 评估传统模型
    print("评估传统模型...")
    traditional_results = evaluate_model(traditional_model, test_data, 'Traditional')
    
    # 绘制混淆矩阵
    print("绘制混淆矩阵...")
    plot_confusion_matrix(bert_results, 'BERT')
    plot_confusion_matrix(traditional_results, 'Traditional')
    
    # 计算评估指标
    print("计算评估指标...")
    bert_metrics = calculate_metrics(bert_results)
    traditional_metrics = calculate_metrics(traditional_results)
    
    # 绘制指标对比图
    print("绘制指标对比图...")
    plot_metrics_comparison(bert_metrics, traditional_metrics)
    
    # 保存评估结果
    print("保存评估结果...")
    results_df = pd.DataFrame({
        '模型': ['BERT', 'Traditional'],
        '准确率': [bert_metrics['准确率'], traditional_metrics['准确率']],
        '精确率': [bert_metrics['精确率'], traditional_metrics['精确率']],
        '召回率': [bert_metrics['召回率'], traditional_metrics['召回率']],
        'F1分数': [bert_metrics['F1分数'], traditional_metrics['F1分数']],
        '误判率': [bert_metrics['误判率'], traditional_metrics['误判率']],
        'TP': [bert_metrics['TP'], traditional_metrics['TP']],
        'FP': [bert_metrics['FP'], traditional_metrics['FP']],
        'FN': [bert_metrics['FN'], traditional_metrics['FN']],
        'TN': [bert_metrics['TN'], traditional_metrics['TN']]
    })
    
    results_df.to_csv('bias_detection_comparison_results.csv', index=False, encoding='utf-8-sig')
    print("评估完成！结果已保存到 bias_detection_comparison_results.csv")
    
    # 打印结果摘要
    print("\n📊 偏见检测评估结果摘要:")
    print("="*60)
    print(f"BERT模型:")
    print(f"  准确率: {bert_metrics['准确率']:.3f}")
    print(f"  精确率: {bert_metrics['精确率']:.3f}")
    print(f"  召回率: {bert_metrics['召回率']:.3f}")
    print(f"  F1分数: {bert_metrics['F1分数']:.3f}")
    print(f"  混淆矩阵: TP={bert_metrics['TP']}, FP={bert_metrics['FP']}, FN={bert_metrics['FN']}, TN={bert_metrics['TN']}")
    
    print(f"\n传统模型:")
    print(f"  准确率: {traditional_metrics['准确率']:.3f}")
    print(f"  精确率: {traditional_metrics['精确率']:.3f}")
    print(f"  召回率: {traditional_metrics['召回率']:.3f}")
    print(f"  F1分数: {traditional_metrics['F1分数']:.3f}")
    print(f"  混淆矩阵: TP={traditional_metrics['TP']}, FP={traditional_metrics['FP']}, FN={traditional_metrics['FN']}, TN={traditional_metrics['TN']}")
    
    print(f"\n🎯 关键发现:")
    print(f"  BERT模型检测到偏见样本数: {bert_metrics['TP'] + bert_metrics['FP']}")
    print(f"  传统模型检测到偏见样本数: {traditional_metrics['TP'] + traditional_metrics['FP']}")
    print(f"  实际偏见样本总数: {bert_metrics['TP'] + bert_metrics['FN']}")
    print(f"  实际非偏见样本总数: {bert_metrics['TN'] + bert_metrics['FP']}")

if __name__ == '__main__':
    main() 