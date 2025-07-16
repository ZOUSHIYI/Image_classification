import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

def setup_chinese_font(font_path=None):
    """
    设置 matplotlib 中文字体。
    如果 font_path 指定，则加载该路径字体文件；否则使用系统默认 SimHei。
    """
    if font_path and os.path.exists(font_path):
        print(f"加载自定义字体: {font_path}")
        matplotlib.font_manager.fontManager.addfont(font_path)
        font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
        matplotlib.rcParams['font.sans-serif'] = [font_name]
    else:
        print("未指定字体路径或路径无效，使用系统默认 SimHei")
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']

    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 调用示例，传入你的字体路径或者None
font_path = 'E:/Image classification/SimHei.ttf'  # 你的字体文件路径，也可以改为 None
setup_chinese_font(font_path)

# 载入类别映射和预测结果
idx_to_labels = np.load('E:/Image classification/3- Model training/output/idx_to_labels.npy', allow_pickle=True).item()
classes = list(idx_to_labels.values())

df = pd.read_csv('E:/Image classification/5- Model evaluation/output/测试集预测结果.csv')

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(df['标注类别名称'], df['top-1-预测名称'], labels=classes)

def plot_confusion_matrix(cm, classes, normalize=False, title='混淆矩阵', cmap=plt.cm.Blues, save_path=None):
    if normalize:
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        cm_sum[cm_sum == 0] = 1
        cm = cm.astype('float') / cm_sum
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=10)

    plt.ylabel('真实类别', fontsize=20, color='r')
    plt.xlabel('预测类别', fontsize=20, color='r')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f'混淆矩阵已保存至 {save_path}')
    plt.show()

# 使用示例
plot_confusion_matrix(cm, classes, normalize=False, save_path='E:/Image classification/5- Model evaluation/output/混淆矩阵.pdf')
