import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 中文字体路径
font_path = 'E:/Image classification/SimHei.ttf'

# 动态加载字体
if os.path.exists(font_path):
    from matplotlib.font_manager import FontProperties
    simhei_font = FontProperties(fname=font_path)
    matplotlib.rcParams['font.family'] = simhei_font.get_name()
else:
    print("警告：未找到SimHei字体文件，可能无法正常显示中文。")

# 负号显示
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
csv_path = 'E:/Image classification/1-Build a dataset/output/数据量统计.csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"数据文件不存在: {csv_path}")

df = pd.read_csv(csv_path)

df_sorted = df.sort_values(by='total', ascending=False)

# --- 各类别图片总数柱状图 ---
plt.figure(figsize=(22, 7))
plt.bar(df_sorted['class'], df_sorted['total'], color='#1f77b4', edgecolor='k')
plt.xticks(rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('类别', fontsize=20)
plt.ylabel('图像数量', fontsize=20)
plt.title('各类别图片总数分布', fontsize=22)
plt.tight_layout()
plt.savefig('E:/Image classification/1-Build a dataset/output/各类别图片数量.pdf', dpi=300, bbox_inches='tight')  # dpi建议用300更高清
plt.show()

# --- 训练集与测试集堆叠柱状图 ---
plt.figure(figsize=(22, 7))
bar_width = 0.55
plt.bar(df_sorted['class'], df_sorted['testset'], bar_width, label='测试集')
plt.bar(df_sorted['class'], df_sorted['trainset'], bar_width, bottom=df_sorted['testset'], label='训练集')
plt.xticks(rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('类别', fontsize=20)
plt.ylabel('图像数量', fontsize=20)
plt.title('训练集与测试集图像数量分布', fontsize=22)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig('E:/Image classification/1-Build a dataset/output/训练集与测试集图像数量分布.pdf', dpi=300, bbox_inches='tight')
plt.show()
