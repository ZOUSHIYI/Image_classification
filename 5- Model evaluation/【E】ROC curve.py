import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib

# 设置中文字体（路径可选）
def setup_chinese_font(font_path=None):
    if font_path and os.path.exists(font_path):
        print(f"加载字体：{font_path}")
        matplotlib.font_manager.fontManager.addfont(font_path)
        font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
        matplotlib.rcParams['font.sans-serif'] = [font_name]
    else:
        print("未指定或找不到字体路径，使用系统默认SimHei字体")
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

# 选择字体路径
font_path = 'E:/Image classification/SimHei.ttf'
setup_chinese_font(font_path)

# 类别映射和数据加载
idx_to_labels = np.load('E:/Image classification/3- Model training/output/idx_to_labels.npy', allow_pickle=True).item()
classes = list(idx_to_labels.values())

df = pd.read_csv('E:/Image classification/5- Model evaluation/output/测试集预测结果.csv')
print(f'加载预测结果，共 {len(df)} 条样本')

# 绘制单类别 ROC 曲线
def plot_single_class_roc(df, class_name, save_dir='E:/Image classification/5- Model evaluation/output'):
    y_true = (df['标注类别名称'] == class_name).astype(int)
    score_col = f'{class_name}-预测置信度'
    if score_col not in df.columns:
        raise ValueError(f"列缺失: {score_col}")

    y_score = df[score_col]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, label=f'{class_name} (AUC={auc_score:.3f})', linewidth=3)
    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=2, label='随机模型')

    plt.title(f'{class_name} ROC 曲线', fontsize=20)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True)
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{class_name}-ROC曲线.pdf', dpi=150, bbox_inches='tight')
    plt.show()
    return auc_score

# 示例：绘制“大象”类别的ROC曲线
plot_single_class_roc(df, '大象')


# 绘制所有类别 ROC 曲线
def get_line_style():
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green',
              'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive',
              'tab:cyan', 'black', 'indianred', 'firebrick', 'darkred', 'darkblue']
    styles = ['-', '--', '-.', ':']
    return {
        'color': random.choice(colors),
        'linestyle': random.choice(styles),
        'linewidth': 2
    }

def plot_all_classes_roc(df, classes, save_path='E:/Image classification/5- Model evaluation/output/各类别ROC曲线.pdf'):
    plt.figure(figsize=(14, 10))
    plt.plot([0, 1], [0, 1], ls='--', c='gray', label='随机模型')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('所有类别 ROC 曲线', fontsize=20)
    plt.grid(True)

    auc_list = []
    for class_name in classes:
        score_col = f'{class_name}-预测置信度'
        if score_col not in df.columns:
            print(f"跳过: {score_col} 缺失")
            auc_list.append(0.0)
            continue
        y_true = (df['标注类别名称'] == class_name).astype(int)
        y_score = df[score_col]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = auc(fpr, tpr)
        auc_list.append(auc_score)
        plt.plot(fpr, tpr, label=f'{class_name} ({auc_score:.3f})', **get_line_style())

    plt.legend(loc='lower right', fontsize=10)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return auc_list

# 绘制所有类别 ROC 曲线
auc_list = plot_all_classes_roc(df, classes)


# 更新 AUC 到评估表格
report_path = 'E:/Image classification/5- Model evaluation/output/各类别准确率评估指标.csv'
df_report = pd.read_csv(report_path)

# 计算加权平均、宏平均 AUC
support = df_report.iloc[:-2]['support'].values
macro_avg_auc = np.mean(auc_list)
weighted_avg_auc = np.sum(np.array(auc_list[:len(support)]) * support / np.sum(support))

auc_list.append(macro_avg_auc)
auc_list.append(weighted_avg_auc)

df_report['AUC'] = auc_list
df_report.to_csv(report_path, index=False)
print(f'已保存更新后的 AUC 至 {report_path}')
