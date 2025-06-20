import pandas as pd
import jieba
import pickle
from collections import Counter
import re
import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from traditional_bias_detector import SVMBiasClassifier, SensitiveWordMatcher, SentimentAnalyzer

# --- 配置 ---
DATA_DIR = 'COLDataset-main/COLDataset'
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

OUTPUT_DIR = 'coldataset_bias_traditional_model_half'
OUTPUT_SVM_MODEL = os.path.join(OUTPUT_DIR, 'svm_model.pkl')
OUTPUT_SENSITIVE_WORDS = os.path.join(OUTPUT_DIR, 'new_sensitive_words.json')
OUTPUT_SENTIMENT_WORDS = os.path.join(OUTPUT_DIR, 'new_sentiment_words.json')

# 数据采样配置
SAMPLE_RATIO = 0.5  # 使用一半的数据
MIN_TEXT_LENGTH = 5   # 最小文本长度
MAX_TEXT_LENGTH = 300 # 文本长度限制

def load_and_sample_data():
    """
    加载并采样一半的COLDataset数据
    """
    print("🚀 加载并采样COLDataset数据集...")
    
    try:
        # 加载数据
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
        
        print(f"  - 训练集: {len(train_df)} 样本")
        print(f"  - 测试集: {len(test_df)} 样本")
        
        # 数据清洗和选择
        train_df = train_df[['TEXT', 'label']].dropna()
        if 'topic' in test_df.columns:
            test_df = test_df[['TEXT', 'label', 'topic']].dropna()
        else:
            test_df = test_df[['TEXT', 'label']].dropna()
        
        # 合并数据集
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        print(f"  - 合并后总数据: {len(full_df)} 样本")
        
        # 基本质量过滤：文本长度
        full_df['text_length'] = full_df['TEXT'].astype(str).str.len()
        filtered_df = full_df[
            (full_df['text_length'] >= MIN_TEXT_LENGTH) & 
            (full_df['text_length'] <= MAX_TEXT_LENGTH)
        ]
        
        print(f"  - 质量过滤后: {len(filtered_df)} 样本")
        
        # 平衡采样：确保正负样本比例保持原有平衡
        safe_samples = filtered_df[filtered_df['label'] == 0]
        offensive_samples = filtered_df[filtered_df['label'] == 1]
        
        # 计算采样数量
        target_safe = int(len(safe_samples) * SAMPLE_RATIO)
        target_offensive = int(len(offensive_samples) * SAMPLE_RATIO)
        
        print(f"  - 采样比例: {SAMPLE_RATIO*100:.0f}%")
        print(f"  - Safe样本: {len(safe_samples)} -> {target_safe}")
        print(f"  - Offensive样本: {len(offensive_samples)} -> {target_offensive}")
        
        # 随机采样
        sampled_safe = safe_samples.sample(n=target_safe, random_state=42)
        sampled_offensive = offensive_samples.sample(n=target_offensive, random_state=42)
        
        # 合并采样结果
        sampled_df = pd.concat([sampled_safe, sampled_offensive], ignore_index=True)
        
        # 如果test数据有topic信息，尽量保留
        if 'topic' in test_df.columns:
            # 为sampled_df添加topic信息（如果原始数据有的话）
            topic_map = {}
            if 'topic' in full_df.columns:
                for idx, row in full_df.iterrows():
                    if pd.notna(row.get('topic')):
                        topic_map[idx] = row['topic']
                
                # 为采样后的数据添加topic信息
                sampled_df['topic'] = sampled_df.index.map(lambda x: topic_map.get(x, None))
        
        print(f"  - 最终采样结果: {len(sampled_df)} 样本")
        print(f"  - Safe: {sum(sampled_df['label'] == 0)}")
        print(f"  - Offensive: {sum(sampled_df['label'] == 1)}")
        
        return sampled_df
        
    except FileNotFoundError:
        print(f"❌ 错误: 数据文件未找到。请确保 '{DATA_DIR}' 目录存在且包含所需文件。")
        return None
    except Exception as e:
        print(f"❌ 加载数据时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- 1. 使用一半数据训练并保存SVM模型 ---
def train_and_save_svm_half(df):
    """使用一半COLDataset训练并保存SVM分类器"""
    print("\n🚀 使用一半数据训练SVM分类器...")
    
    try:
        # 准备训练数据
        texts = df['TEXT'].astype(str).tolist()
        labels = df['label'].astype(int).tolist()
        
        print(f"  - 训练样本数: {len(texts)}")
        print(f"  - 正样本(偏见): {sum(labels)}")
        print(f"  - 负样本(正常): {len(labels) - sum(labels)}")

        # 训练SVM
        svm_classifier = SVMBiasClassifier()
        print("  - 正在训练SVM模型...")
        svm_classifier.train(texts, labels)
        print("  - SVM模型训练完成。")

        # 创建输出目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 保存模型
        with open(OUTPUT_SVM_MODEL, 'wb') as f:
            pickle.dump(svm_classifier, f)
        
        model_size = os.path.getsize(OUTPUT_SVM_MODEL) / (1024 * 1024)  # MB
        print(f"✅ SVM模型已成功保存到: {OUTPUT_SVM_MODEL}")
        print(f"  - 模型大小: {model_size:.1f} MB")

    except Exception as e:
        print(f"❌ 训练SVM时发生错误: {e}")
        import traceback
        traceback.print_exc()

# --- 2. 从一半数据提取并保存敏感词词典 ---
def extract_and_save_sensitive_words_half(df):
    """从一半COLDataset中提取并保存敏感词"""
    print("\n🚀 从一半数据提取敏感词词典...")
    
    try:
        # 如果有topic信息，使用topic分类；否则使用默认分类
        if 'topic' in df.columns:
            # 筛选出有偏见且有主题分类的文本
            offensive_df = df[(df['label'] == 1) & (df['topic'].notna())]
            print(f"  - 找到 {len(offensive_df)} 条带主题的偏见样本。")
        else:
            # 没有topic信息，使用所有偏见样本并手动分类
            offensive_df = df[df['label'] == 1]
            print(f"  - 找到 {len(offensive_df)} 条偏见样本，将进行手动分类。")

        # 扩展的停用词列表
        stopwords = set([
            '的', '了', '是', '我', '你', '他', '她', '它', '都', '也', '还', '就', 
            '这个', '一个', '什么', '我们', '你们', '他们', '但是', '所以', '不是', 
            '有', '没有', '可以', '会', '要', '很', '更', '最', '在', '和', '或者',
            '因为', '如果', '那么', '这样', '那样', '应该', '可能', '已经', '然后',
            '现在', '以后', '之前', '一直', '总是', '从来', '永远', '绝对'
        ])
        
        new_sensitive_words = {}
        
        if 'topic' in df.columns and not df['topic'].isna().all():
            # 按主题分组提取敏感词
            topic_mapping = {
                'gender': 'gender',
                'race': 'race', 
                'region': 'region'
            }
            
            for topic, group in offensive_df.groupby('topic'):
                print(f"  - 正在处理主题: {topic} ({len(group)} 样本)")
                text_corpus = ' '.join(group['TEXT'].astype(str))
                
                # 使用jieba分词
                words = [word for word in jieba.lcut(text_corpus) 
                        if len(word) > 1 and word not in stopwords and not word.isdigit()]
                
                # 统计词频
                word_counts = Counter(words)
                
                # 提取高频词，根据样本数量动态调整
                min_freq = max(2, len(group) // 200)  # 动态阈值，比完整版更宽松
                top_words = [word for word, count in word_counts.most_common(80) 
                           if count >= min_freq]
                
                # 映射到标准分类
                standard_topic = topic_mapping.get(topic, topic)
                new_sensitive_words[standard_topic] = top_words
                
                print(f"    - 提取关键词 ({len(top_words)}个，最小频次≥{min_freq})")
                print(f"    - 示例词汇: {', '.join(top_words[:10])}...")
        else:
            # 没有topic信息，根据关键词模式手动分类
            print("  - 没有主题信息，使用关键词模式进行自动分类...")
            
            # 分类关键词模式
            gender_patterns = ['女', '男', '性别', '妇女', '男人', '女人', '姑娘', '小伙']
            race_patterns = ['黑人', '白人', '种族', '民族', '中国', '美国', '日本', '韩国', '印度']
            region_patterns = ['地区', '地方', '城市', '农村', '东北', '南方', '北方', '上海', '北京']
            
            all_text = ' '.join(offensive_df['TEXT'].astype(str))
            words = [word for word in jieba.lcut(all_text) 
                    if len(word) > 1 and word not in stopwords and not word.isdigit()]
            word_counts = Counter(words)
            
            # 按模式分类
            for category, patterns in [('gender', gender_patterns), 
                                     ('race', race_patterns), 
                                     ('region', region_patterns)]:
                
                category_words = []
                for word, count in word_counts.most_common(500):
                    if any(pattern in word for pattern in patterns) or count >= 8:
                        category_words.append(word)
                        if len(category_words) >= 60:  # 每类最多60个词
                            break
                
                new_sensitive_words[category] = category_words
                print(f"    - {category}: {len(category_words)}个词汇")

        # 保存为JSON
        with open(OUTPUT_SENSITIVE_WORDS, 'w', encoding='utf-8') as f:
            json.dump(new_sensitive_words, f, ensure_ascii=False, indent=4)
        
        # 统计信息
        total_words = sum(len(words) for words in new_sensitive_words.values())
        print(f"✅ 敏感词词典已保存到: {OUTPUT_SENSITIVE_WORDS}")
        print(f"  - 总词汇数: {total_words}")
        for category, words in new_sensitive_words.items():
            print(f"  - {category}: {len(words)}个")

    except Exception as e:
        print(f"❌ 提取敏感词时发生错误: {e}")
        import traceback
        traceback.print_exc()

# --- 3. 从一半数据提取并保存情感词词典 ---
def extract_and_save_sentiment_words_half(df):
    """从一半COLDataset中提取并保存情感词典"""
    print("\n🚀 从一半数据提取情感词典...")
    
    try:
        # 分别处理正面和负面样本
        safe_texts = df[df['label'] == 0]['TEXT'].astype(str)
        offensive_texts = df[df['label'] == 1]['TEXT'].astype(str)
        
        print(f"  - 正常样本: {len(safe_texts)}条")
        print(f"  - 偏见样本: {len(offensive_texts)}条")
        
        # 停用词
        stopwords = set([
            '的', '了', '是', '我', '你', '他', '她', '它', '都', '也', '还', '就',
            '这个', '一个', '什么', '我们', '你们', '他们', '但是', '所以', '不是',
            '有', '没有', '可以', '会', '要', '很', '更', '最'
        ])

        # 从正常文本中提取正面词汇
        print("  - 正在分析正常文本...")
        safe_corpus = ' '.join(safe_texts)
        safe_words = [word for word in jieba.lcut(safe_corpus) 
                     if len(word) > 1 and word not in stopwords and not word.isdigit()]
        safe_word_counts = Counter(safe_words)

        # 从偏见文本中提取负面词汇
        print("  - 正在分析偏见文本...")
        offensive_corpus = ' '.join(offensive_texts)
        offensive_words = [word for word in jieba.lcut(offensive_corpus) 
                          if len(word) > 1 and word not in stopwords and not word.isdigit()]
        offensive_word_counts = Counter(offensive_words)

        # 计算词汇的情感倾向性
        print("  - 正在计算词汇情感倾向...")
        
        # 正面词：在正常文本中频率高，在偏见文本中频率低
        positive_words = []
        for word, safe_freq in safe_word_counts.most_common(800):
            offensive_freq = offensive_word_counts.get(word, 0)
            safe_ratio = safe_freq / len(safe_texts) if len(safe_texts) > 0 else 0
            offensive_ratio = offensive_freq / len(offensive_texts) if len(offensive_texts) > 0 else 0
            
            # 正面词条件：在正常文本中出现频率明显高于偏见文本
            if safe_ratio > offensive_ratio * 1.8 and safe_freq >= 3:  # 降低阈值
                positive_words.append(word)
                if len(positive_words) >= 60:  # 限制数量
                    break

        # 负面词：在偏见文本中频率高，在正常文本中频率低
        negative_words = []
        for word, offensive_freq in offensive_word_counts.most_common(800):
            safe_freq = safe_word_counts.get(word, 0)
            safe_ratio = safe_freq / len(safe_texts) if len(safe_texts) > 0 else 0
            offensive_ratio = offensive_freq / len(offensive_texts) if len(offensive_texts) > 0 else 0
            
            # 负面词条件：在偏见文本中出现频率明显高于正常文本
            if offensive_ratio > safe_ratio * 1.8 and offensive_freq >= 3:  # 降低阈值
                negative_words.append(word)
                if len(negative_words) >= 60:  # 限制数量
                    break

        # 极端词：表示绝对化、极端化的词汇
        extreme_patterns = ['都', '全部', '所有', '一律', '统统', '总是', '永远', '从来', '绝对', '肯定', '必然', '天生', '生来', '就是']
        extreme_words = []
        
        all_words = list(safe_word_counts.keys()) + list(offensive_word_counts.keys())
        for word in set(all_words):
            if any(pattern in word for pattern in extreme_patterns) or word in extreme_patterns:
                total_freq = safe_word_counts.get(word, 0) + offensive_word_counts.get(word, 0)
                if total_freq >= 2:  # 降低阈值：至少出现2次
                    extreme_words.append(word)

        # 去重并限制数量
        extreme_words = list(set(extreme_words))[:30]

        # 构建情感词典
        sentiment_dict = {
            'positive_words': positive_words,
            'negative_words': negative_words,
            'extreme_words': extreme_words
        }

        # 保存为JSON
        with open(OUTPUT_SENTIMENT_WORDS, 'w', encoding='utf-8') as f:
            json.dump(sentiment_dict, f, ensure_ascii=False, indent=4)

        print(f"✅ 情感词典已保存到: {OUTPUT_SENTIMENT_WORDS}")
        print(f"  - 正面词汇: {len(positive_words)}个")
        print(f"  - 负面词汇: {len(negative_words)}个") 
        print(f"  - 极端词汇: {len(extreme_words)}个")
        
        # 显示示例
        print(f"  - 正面词示例: {', '.join(positive_words[:8])}")
        print(f"  - 负面词示例: {', '.join(negative_words[:8])}")
        print(f"  - 极端词示例: {', '.join(extreme_words[:8])}")

    except Exception as e:
        print(f"❌ 提取情感词时发生错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数：使用一半COLDataset训练增强版传统偏见检测器"""
    print("🚀 基于一半COLDataset的传统偏见检测器增强工具")
    print("=" * 70)
    
    # 加载一半数据
    df = load_and_sample_data()
    if df is None:
        print("❌ 数据加载失败，程序退出。")
        return
    
    print(f"\n📊 数据统计:")
    print(f"  - 总样本数: {len(df)}")
    print(f"  - 正常样本: {sum(df['label'] == 0)} ({sum(df['label'] == 0)/len(df)*100:.1f}%)")
    print(f"  - 偏见样本: {sum(df['label'] == 1)} ({sum(df['label'] == 1)/len(df)*100:.1f}%)")
    
    if 'topic' in df.columns:
        print(f"  - 主题分布:")
        topic_counts = df[df['label'] == 1]['topic'].value_counts()
        for topic, count in topic_counts.items():
            print(f"    - {topic}: {count}样本")
    
    # 1. 训练SVM模型
    train_and_save_svm_half(df)
    
    # 2. 提取敏感词词典
    extract_and_save_sensitive_words_half(df)
    
    # 3. 提取情感词词典
    extract_and_save_sentiment_words_half(df)
    
    print("\n" + "=" * 70)
    print("🎉 一半数据增强版传统偏见检测器训练完成！")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("📋 生成文件:")
    print(f"  - SVM模型: {OUTPUT_SVM_MODEL}")
    print(f"  - 敏感词词典: {OUTPUT_SENSITIVE_WORDS}")
    print(f"  - 情感词词典: {OUTPUT_SENTIMENT_WORDS}")
    
    # 显示文件大小
    if os.path.exists(OUTPUT_SVM_MODEL):
        model_size = os.path.getsize(OUTPUT_SVM_MODEL) / (1024 * 1024)
        print(f"  - 模型大小: {model_size:.1f} MB")

if __name__ == '__main__':
    main() 