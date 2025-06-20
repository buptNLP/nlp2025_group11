#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于COLDataset的BERT偏见检测训练脚本
适配COLDataset数据格式，训练中文偏见检测模型
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EvalPrediction
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class COLDatasetBiasDataset(Dataset):
    """COLDataset偏见检测数据集"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class COLDatasetBertTrainer:
    """基于COLDataset的BERT训练器"""
    
    def __init__(self, output_dir='./coldataset_bias_bert_model'):
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_weights = None
        
        os.makedirs(output_dir, exist_ok=True)
        self.setup_environment()
        
    def setup_environment(self):
        """设置环境配置"""
        print("🔧 设置训练环境...")
        
        # 设置缓存目录
        os.environ['TRANSFORMERS_CACHE'] = './cache'
        os.environ['HF_HOME'] = './cache'
        os.makedirs('./cache', exist_ok=True)
        
    def load_model(self):
        """加载预训练模型"""
        print("📦 加载预训练模型...")
        
        try:
            # 使用本地缓存的bert-base-chinese
            self.tokenizer = AutoTokenizer.from_pretrained(
                'bert-base-chinese',
                cache_dir='./cache',
                local_files_only=True,
                trust_remote_code=True
            )
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                'bert-base-chinese',
                num_labels=2,
                cache_dir='./cache',
                local_files_only=True,
                trust_remote_code=True
            )
            
            self.model.to(self.device)
            
            print(f"✅ 模型加载成功")
            print(f"   模型: bert-base-chinese")
            print(f"   设备: {self.device}")
            print(f"   参数量: {self.model.num_parameters():,}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def analyze_data_distribution(self, df, dataset_name):
        """分析COLDataset数据分布"""
        print(f"\n📊 {dataset_name}数据分布分析:")
        
        # 标签分布
        label_dist = df['label'].value_counts().sort_index()
        print("标签分布:")
        for label, count in label_dist.items():
            percentage = count / len(df) * 100
            label_name = "safe" if label == 0 else "offensive"
            print(f"  {label} ({label_name}): {count:,} ({percentage:.1f}%)")
        
        # 主题分布
        if 'topic' in df.columns:
            topic_dist = df['topic'].value_counts()
            print("主题分布:")
            for topic, count in topic_dist.items():
                percentage = count / len(df) * 100
                print(f"  {topic}: {count:,} ({percentage:.1f}%)")
        
        # 细粒度标签分布（仅测试集）
        if 'fine-grained-label' in df.columns:
            fine_dist = df['fine-grained-label'].value_counts()
            print("细粒度标签分布:")
            for fine_label, count in fine_dist.items():
                percentage = count / len(df) * 100
                print(f"  {fine_label}: {count:,} ({percentage:.1f}%)")
        
        return label_dist
    
    def load_coldataset(self):
        """加载COLDataset数据"""
        print("\n📊 加载COLDataset数据...")
        
        # 加载数据
        train_df = pd.read_csv('COLDataset-main/COLDataset/train.csv')
        dev_df = pd.read_csv('COLDataset-main/COLDataset/dev.csv')
        test_df = pd.read_csv('COLDataset-main/COLDataset/test.csv')
        
        # 数据清洗
        for df_name, df in [('训练集', train_df), ('开发集', dev_df), ('测试集', test_df)]:
            before_len = len(df)
            df.dropna(subset=['TEXT', 'label'], inplace=True)
            after_len = len(df)
            if before_len != after_len:
                print(f"{df_name}: 清理 {before_len - after_len} 个缺失样本")
        
        print(f"✅ 数据加载完成: 训练集{len(train_df)}，开发集{len(dev_df)}，测试集{len(test_df)}")
        
        # 分析原始数据分布
        for df_name, df in [('训练集', train_df), ('开发集', dev_df), ('测试集', test_df)]:
            self.analyze_data_distribution(df, df_name)
        
        # 计算类别权重
        self.class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_df['label']),
            y=train_df['label']
        )
        self.class_weights = torch.tensor(self.class_weights, dtype=torch.float).to(self.device)
        
        print(f"\n⚖️ 类别权重计算完成:")
        print(f"  safe (0): {self.class_weights[0]:.3f}")
        print(f"  offensive (1): {self.class_weights[1]:.3f}")
        
        return train_df, dev_df, test_df
    
    def create_datasets(self, train_df, dev_df, test_df):
        """创建数据集"""
        print("📦 创建数据集...")
        
        train_dataset = COLDatasetBiasDataset(
            texts=train_df['TEXT'].astype(str).tolist(),
            labels=train_df['label'].tolist(),
            tokenizer=self.tokenizer
        )
        
        dev_dataset = COLDatasetBiasDataset(
            texts=dev_df['TEXT'].astype(str).tolist(),
            labels=dev_df['label'].tolist(),
            tokenizer=self.tokenizer
        )
        
        test_dataset = COLDatasetBiasDataset(
            texts=test_df['TEXT'].astype(str).tolist(),
            labels=test_df['label'].tolist(),
            tokenizer=self.tokenizer
        )
        
        print(f"✅ 数据集创建完成")
        print(f"   训练集: {len(train_dataset)} 样本")
        print(f"   开发集: {len(dev_dataset)} 样本")
        print(f"   测试集: {len(test_dataset)} 样本")
        
        return train_dataset, dev_dataset, test_dataset
    
    def compute_metrics(self, eval_pred: EvalPrediction):
        """计算评估指标"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # 计算基础指标
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        # 计算偏见检测专项指标
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        # safe类指标
        safe_recall = recall_per_class[0] if len(recall_per_class) > 0 else 0
        safe_precision = precision_per_class[0] if len(precision_per_class) > 0 else 0
        safe_f1 = f1_per_class[0] if len(f1_per_class) > 0 else 0
        
        # offensive类指标
        offensive_recall = recall_per_class[1] if len(recall_per_class) > 1 else 0
        offensive_precision = precision_per_class[1] if len(precision_per_class) > 1 else 0
        offensive_f1 = f1_per_class[1] if len(f1_per_class) > 1 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'safe_recall': safe_recall,
            'safe_precision': safe_precision,
            'safe_f1': safe_f1,
            'offensive_recall': offensive_recall,
            'offensive_precision': offensive_precision,
            'offensive_f1': offensive_f1,
        }
    
    def create_trainer(self, train_dataset, dev_dataset):
        """创建训练器"""
        print("🎯 创建COLDataset训练器...")
        
        class WeightedTrainer(Trainer):
            def __init__(self, class_weights, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.class_weights = class_weights
                
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get('logits')
                
                # 使用加权交叉熵损失处理类别不平衡
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
                loss = loss_fct(logits, labels)
                
                # 添加L2正则化防止过拟合
                l2_lambda = 0.001
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + l2_lambda * l2_norm
                
                return (loss, outputs) if return_outputs else loss
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=2e-5,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="offensive_f1",  # 优化offensive类F1
            greater_is_better=True,
            logging_steps=50,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=2,
            seed=42,
            report_to=None,
            gradient_accumulation_steps=2,
        )
        
        trainer = WeightedTrainer(
            class_weights=self.class_weights,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        return trainer
    
    def train_and_evaluate(self, trainer, test_dataset):
        """训练和评估模型"""
        print("\n🚀 开始训练...")
        
        # 训练
        train_result = trainer.train()
        
        # 保存模型和tokenizer
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print("✅ 训练完成，开始评估...")
        
        # 评估测试集
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # 计算详细指标
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # 每类指标
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 构建详细结果
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'safe_precision': float(precision_per_class[0]) if len(precision_per_class) > 0 else 0,
            'safe_recall': float(recall_per_class[0]) if len(recall_per_class) > 0 else 0,
            'safe_f1': float(f1_per_class[0]) if len(f1_per_class) > 0 else 0,
            'offensive_precision': float(precision_per_class[1]) if len(precision_per_class) > 1 else 0,
            'offensive_recall': float(recall_per_class[1]) if len(recall_per_class) > 1 else 0,
            'offensive_f1': float(f1_per_class[1]) if len(f1_per_class) > 1 else 0,
            'confusion_matrix': cm.tolist(),
            'training_time': str(datetime.now()),
        }
        
        # 保存结果
        with open(f'{self.output_dir}/training_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 详细结果输出
        print("\n" + "="*60)
        print("🎯 COLDataset模型训练完成! 详细结果:")
        print("="*60)
        
        print(f"总体性能:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  加权精确率: {precision:.4f}")
        print(f"  加权召回率: {recall:.4f}")
        print(f"  加权F1: {f1:.4f}")
        
        print(f"\n各类别详细性能:")
        print(f"Safe类检测:")
        print(f"  精确率: {results['safe_precision']:.4f}")
        print(f"  召回率: {results['safe_recall']:.4f}")
        print(f"  F1分数: {results['safe_f1']:.4f}")
        
        print(f"Offensive类检测:")
        print(f"  精确率: {results['offensive_precision']:.4f}")
        print(f"  召回率: {results['offensive_recall']:.4f}")
        print(f"  F1分数: {results['offensive_f1']:.4f}")
        
        print(f"\n混淆矩阵:")
        print(f"  预测\\真实    Safe    Offensive")
        print(f"  Safe       {cm[0,0]:4d}    {cm[0,1]:4d}")
        print(f"  Offensive  {cm[1,0]:4d}    {cm[1,1]:4d}")
        
        # 性能评价
        offensive_f1 = results['offensive_f1']
        print(f"\n🎯 模型性能评估:")
        if offensive_f1 > 0.8:
            print("✅ Offensive检测F1优秀! (>0.8)")
        elif offensive_f1 > 0.7:
            print("✅ Offensive检测F1良好! (>0.7)")
        elif offensive_f1 > 0.6:
            print("⚠️ Offensive检测F1一般 (>0.6)")
        else:
            print("❌ Offensive检测F1较低 (<0.6)")
        
        print(f"📁 模型和结果已保存到: {self.output_dir}")
        
        return results
    
    def run_training(self):
        """执行完整训练流程"""
        print("🚀 开始COLDataset偏见检测模型训练...")
        print(f"输出目录: {self.output_dir}")
        
        # 1. 加载模型
        if not self.load_model():
            return False
        
        # 2. 加载数据
        train_df, dev_df, test_df = self.load_coldataset()
        
        # 3. 创建数据集
        train_dataset, dev_dataset, test_dataset = self.create_datasets(train_df, dev_df, test_df)
        
        # 4. 创建训练器
        trainer = self.create_trainer(train_dataset, dev_dataset)
        
        # 5. 训练和评估
        results = self.train_and_evaluate(trainer, test_dataset)
        
        print("🎉 COLDataset模型训练完成!")
        return True

def main():
    """主函数"""
    trainer = COLDatasetBertTrainer()
    success = trainer.run_training()
    
    if success:
        print("✅ 训练成功完成!")
    else:
        print("❌ 训练失败!")

if __name__ == "__main__":
    main() 