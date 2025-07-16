import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib
from matplotlib import font_manager

# === 中文字体设置 ===
font_path = 'E:/Image classification/SimHei.ttf'
my_font = font_manager.FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = my_font.get_name()
matplotlib.rcParams['axes.unicode_minus'] = False

# === 可配置路径 ===
output_dir = 'E:/Image classification/3- Model training/图表'  # 图表输出路径
train_log_path = 'E:/Image classification/3- Model training/output/训练日志-训练集.csv'  # 训练日志 CSV 路径
test_log_path = 'E:/Image classification/3- Model training/output/训练日志-测试集.csv'   # 测试日志 CSV 路径

# 自动创建图表保存路径
os.makedirs(output_dir, exist_ok=True)

# === 绘图样式配置 ===
colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
linestyles = ['-', '--', '-.']

def get_line_arg():
    return {
        'color': random.choice(colors),
        'linestyle': random.choice(linestyles),
        'linewidth': random.randint(2, 4)
    }

def plot_line(x, ys, labels, title, xlabel, ylabel, save_name, ylim=None):
    """
    通用绘图函数
    save_name: 保存文件名，如 '训练集损失函数.pdf'
    """
    plt.figure(figsize=(16, 8))
    if isinstance(ys[0], pd.Series) or isinstance(ys[0], list):
        for y, label in zip(ys, labels):
            plt.plot(x, y, label=label, **get_line_arg())
    else:
        plt.plot(x, ys, label=labels, **get_line_arg())

    plt.tick_params(labelsize=16)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title, fontsize=22)
    if ylim:
        plt.ylim(ylim)
    plt.legend(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, save_name), dpi=150)
    plt.show()

# === 加载日志文件 ===
df_train = pd.read_csv(train_log_path)
df_test = pd.read_csv(test_log_path)

# === 绘图 ===
plot_line(
    x=df_train['batch'],
    ys=df_train['train_loss'],
    labels='训练集',
    title='训练集损失函数',
    xlabel='batch',
    ylabel='损失函数',
    save_name='训练集损失函数.pdf'
)

plot_line(
    x=df_train['batch'],
    ys=df_train['train_accuracy'],
    labels='训练集',
    title='训练集准确率',
    xlabel='batch',
    ylabel='准确率',
    save_name='训练集准确率.pdf'
)

plot_line(
    x=df_test['epoch'],
    ys=df_test['test_loss'],
    labels='测试集',
    title='测试集损失函数',
    xlabel='epoch',
    ylabel='损失函数',
    save_name='测试集损失函数.pdf'
)

metric_names = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1-score']
plot_line(
    x=df_test['epoch'],
    ys=[df_test[m] for m in metric_names],
    labels=metric_names,
    title='测试集分类评估指标',
    xlabel='epoch',
    ylabel='评估指标',
    save_name='测试集分类评估指标.pdf',
    ylim=[0, 1]
)
