import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

import matplotlib

def setup_chinese_font(font_path=None):
    """
    设置matplotlib中文字体，如果font_path有效则加载该字体，否则使用系统默认SimHei。
    """
    if font_path and os.path.exists(font_path):
        print(f"加载自定义字体：{font_path}")
        matplotlib.font_manager.fontManager.addfont(font_path)
        font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
        matplotlib.rcParams['font.sans-serif'] = [font_name]
    else:
        print("未指定字体路径或路径无效，使用系统默认SimHei")
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# 这里填写你的SimHei.ttf路径，或者设为None使用系统默认字体
font_path = 'E:/Image classification/SimHei.ttf'
setup_chinese_font(font_path)

# 载入类别映射
idx_to_labels = np.load('E:/Image classification/3- Model training/output/idx_to_labels.npy', allow_pickle=True).item()
classes = list(idx_to_labels.values())
print("类别列表:", classes)

# 载入预测结果
pred_result_path = 'E:/Image classification/5- Model evaluation/output/测试集预测结果.csv'
if not os.path.exists(pred_result_path):
    raise FileNotFoundError(f"找不到预测结果文件：{pred_result_path}")

df = pd.read_csv(pred_result_path)
print(df.head())

# 绘制某一类别的PR曲线
def plot_single_class_pr_curve(df, class_name):
    y_true = (df['标注类别名称'] == class_name).astype(int)
    score_col = f'{class_name}-预测置信度'
    if score_col not in df.columns:
        raise KeyError(f"数据中不存在列: {score_col}")
    y_scores = df[score_col]
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    AP = average_precision_score(y_true, y_scores, average='weighted')
    print(f"{class_name} 的 AP: {AP:.3f}")

    plt.figure(figsize=(12, 8))
    plt.plot(recall, precision, linewidth=3, label=class_name)
    plt.plot([0, 0], [0, 1], ls="--", c='.3', linewidth=2, label='随机模型')
    plt.plot([0, 1], [sum(y_true) / len(y_true)]*2, ls="--", c='.3', linewidth=2)

    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.title(f'{class_name} PR曲线  AP: {AP:.3f}', fontsize=22)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(fontsize=14)
    plt.grid(True)

    os.makedirs('E:/Image classification/5- Model evaluation/output', exist_ok=True)
    plt.savefig(f'E:/Image classification/5- Model evaluation/output/{class_name}-PR曲线.pdf', dpi=120, bbox_inches='tight')
    plt.show()

# 示例：绘制“大象”类别PR曲线
plot_single_class_pr_curve(df, '大象')


# 绘制所有类别的PR曲线
def get_line_arg():
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green',
              'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive',
              'tab:cyan', 'black', 'indianred', 'brown', 'firebrick', 'maroon', 'darkred',
              'red', 'sienna', 'chocolate', 'yellow', 'olivedrab', 'yellowgreen',
              'darkolivegreen', 'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime',
              'seagreen', 'mediumseagreen', 'darkslategray', 'darkslategrey', 'teal',
              'darkcyan', 'dodgerblue', 'navy', 'darkblue', 'mediumblue', 'blue',
              'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple',
              'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid', 'darkviolet',
              'mediumorchid', 'purple', 'darkmagenta', 'fuchsia', 'magenta', 'orchid',
              'mediumvioletred', 'deeppink', 'hotpink']
    linestyles = ['--', '-.', '-']
    return {
        'color': random.choice(colors),
        'linestyle': random.choice(linestyles),
        'linewidth': random.randint(1, 4)
    }

plt.figure(figsize=(14, 10))
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('各类别 PR 曲线', fontsize=22)
plt.grid(True)

ap_list = []
for class_name in classes:
    y_true = (df['标注类别名称'] == class_name).astype(int)
    score_col = f'{class_name}-预测置信度'
    if score_col not in df.columns:
        print(f"跳过缺失的类别置信度列: {score_col}")
        continue
    y_scores = df[score_col]
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    AP = average_precision_score(y_true, y_scores, average='weighted')
    plt.plot(recall, precision, **get_line_arg(), label=class_name)
    ap_list.append(AP)

plt.legend(loc='best', fontsize=12)
os.makedirs('E:/Image classification/5- Model evaluation/output', exist_ok=True)
plt.savefig('E:/Image classification/5- Model evaluation/output/各类别PR曲线.pdf', dpi=120, bbox_inches='tight')
plt.show()

# 读取准确率评估指标表格，增加AP列
eval_report_path = 'E:/Image classification/5- Model evaluation/output/各类别准确率评估指标.csv'
if os.path.exists(eval_report_path):
    df_report = pd.read_csv(eval_report_path)
    # 计算宏平均和加权平均AP
    support = df_report['support'][:-2].values if 'support' in df_report.columns else None
    if support is not None and len(support) == len(ap_list):
        macro_avg_ap = np.mean(ap_list)
        weighted_avg_ap = np.sum(np.array(ap_list) * support / support.sum())
        ap_list.append(macro_avg_ap)
        ap_list.append(weighted_avg_ap)
    else:
        macro_avg_ap = weighted_avg_ap = np.nan
        print("警告：无法计算加权平均AP，support列缺失或长度不匹配")

    df_report['AP'] = ap_list
    df_report.to_csv(eval_report_path, index=False)
    print(f"已更新准确率评估指标表格，保存路径：{eval_report_path}")
else:
    print(f"找不到准确率评估指标文件：{eval_report_path}，请先生成该文件")
