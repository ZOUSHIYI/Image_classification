import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# --------- 可选中文字体路径设置 ---------
def setup_chinese_font(font_path=None):
    try:
        if font_path and os.path.exists(font_path):
            matplotlib.font_manager.fontManager.addfont(font_path)
            font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
            matplotlib.rcParams['font.sans-serif'] = [font_name]
        else:
            matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print("⚠️ 字体加载失败，继续使用默认设置：", e)

# 字体路径可选（本地中文字体）
setup_chinese_font('E:/Image classification/SimHei.ttf')

# --------- 参数配置 ---------
report_path = 'E:/Image classification/5- Model evaluation/output/各类别准确率评估指标.csv'   # 表格路径
save_dir = 'E:/Image classification/5- Model evaluation/output'                             # 图像保存路径
feature = 'recall'                              # 可选：'precision' | 'recall' | 'f1-score' | 'accuracy' | 'AP' | 'AUC'

# --------- 加载数据 ---------
df = pd.read_csv(report_path)
df = df.copy()

# 移除宏平均与加权平均两行（通常在最后）
if df.iloc[-1]['类别'] == 'weighted avg':
    df = df.iloc[:-2]

# 排序后绘图
df_plot = df.sort_values(by=feature, ascending=False)
x = df_plot['类别']
y = df_plot[feature]

# --------- 绘制柱状图 ---------
plt.figure(figsize=(max(10, len(x) * 0.6), 7))
bars = plt.bar(x, y, width=0.6, facecolor='#1f77b4', edgecolor='k')
plt.bar_label(bars, fmt='%.2f', fontsize=13)

plt.xticks(rotation=45, ha='right')
plt.tick_params(labelsize=13)
plt.ylabel(feature, fontsize=18)
plt.title(f'各类别 {feature} 指标柱状图', fontsize=22)
plt.tight_layout()

# 保存图像
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f'各类别准确率评估指标柱状图-{feature}.pdf')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()

print(f'图像已保存至：{save_path}')
