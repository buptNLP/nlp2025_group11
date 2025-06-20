import pandas as pd
import json
from visualization_utils import plot_pie_chart, plot_bar_chart

def load_coldataset():
    """加载COLDataset数据"""
    print("正在加载COLDataset数据...")
    
    # 加载数据
    train_df = pd.read_csv('COLDataset-main/COLDataset/train.csv')
    dev_df = pd.read_csv('COLDataset-main/COLDataset/dev.csv')
    test_df = pd.read_csv('COLDataset-main/COLDataset/test.csv')
    
    return train_df, dev_df, test_df

def analyze_and_visualize():
    """分析并可视化COLDataset数据分布"""
    # 加载数据
    train_df, dev_df, test_df = load_coldataset()
    
    # 1. 标签分布可视化
    print("\n生成标签分布饼图...")
    label_dist = train_df['label'].value_counts()
    labels = ['安全文本', '偏见文本']
    plot_pie_chart(
        data=label_dist.values,
        labels=labels,
        title='COLDataset训练集标签分布',
        output_file='coldataset_label_distribution.png'
    )
    
    # 2. 主题分布可视化
    print("生成主题分布柱状图...")
    topic_dist = train_df['topic'].value_counts()
    topic_names = {
        'race': '种族偏见',
        'gender': '性别偏见',
        'region': '地域偏见'
    }
    topic_labels = [topic_names[topic] for topic in topic_dist.index]
    plot_bar_chart(
        data=topic_dist.values,
        labels=topic_labels,
        title='COLDataset训练集主题分布',
        xlabel='偏见类型',
        ylabel='样本数量',
        output_file='coldataset_topic_distribution.png'
    )
    
    # 3. 细粒度标签分布（仅测试集）
    if 'fine-grained-label' in test_df.columns:
        print("生成细粒度标签分布饼图...")
        fine_dist = test_df['fine-grained-label'].value_counts()
        fine_labels = {
            '0': '无偏见',
            '1': '轻微偏见',
            '2': '中度偏见',
            '3': '严重偏见'
        }
        fine_label_names = [fine_labels[str(label)] for label in fine_dist.index]
        plot_pie_chart(
            data=fine_dist.values,
            labels=fine_label_names,
            title='COLDataset测试集细粒度标签分布',
            output_file='coldataset_fine_grained_distribution.png'
        )
    
    print("\n可视化完成！生成的图表文件：")
    print("- coldataset_label_distribution.png")
    print("- coldataset_topic_distribution.png")
    if 'fine-grained-label' in test_df.columns:
        print("- coldataset_fine_grained_distribution.png")

if __name__ == '__main__':
    analyze_and_visualize() 