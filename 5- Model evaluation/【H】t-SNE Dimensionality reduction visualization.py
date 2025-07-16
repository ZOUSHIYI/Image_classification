import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
from PIL import Image
from sklearn.manifold import TSNE

# 忽略警告
warnings.filterwarnings("ignore")

# =================== 可配置路径 =====================
output_dir = 'E:/Image classification/5- Model evaluation/output'
semantic_feature_path = os.path.join(output_dir, '测试集语义特征.npy')
csv_path = os.path.join(output_dir, '测试集预测结果.csv')
font_path = 'E:/Image classification/SimHei.ttf'  # 本地字体路径可选
# ====================================================

# =================== 中文字体设置 ====================
import matplotlib
if os.path.exists(font_path):
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
# =====================================================

# =================== 加载数据 ========================
encoding_array = np.load(semantic_feature_path, allow_pickle=True)
df = pd.read_csv(csv_path)
classes = df['标注类别名称'].unique()
print(f'✅ 类别数: {len(classes)}, 语义特征维度: {encoding_array.shape}')
# =====================================================

# =================== 可视化参数 ======================
marker_list = ['o', 's', '^', 'D', '*', 'v', '>', '<', 'h', 'p', 'X']
palette = sns.color_palette("hls", len(classes))
random.seed(1234)
random.shuffle(marker_list)
random.shuffle(palette)
# =====================================================

# =================== t-SNE 降维至二维 =================

tsne_2d = TSNE(n_components=2, max_iter=10000)

X_tsne_2d = tsne_2d.fit_transform(encoding_array)

# 可视化
plt.figure(figsize=(14, 14))
for idx, cls in enumerate(classes):
    indices = np.where(df['标注类别名称'] == cls)
    plt.scatter(X_tsne_2d[indices, 0], X_tsne_2d[indices, 1],
                c=[palette[idx]], label=cls, marker=marker_list[idx % len(marker_list)], s=100)
plt.legend(fontsize=14, bbox_to_anchor=(1.05, 1))
plt.title('t-SNE二维语义特征可视化', fontsize=20)
plt.xticks([]); plt.yticks([])
plt.savefig(os.path.join(output_dir, '语义特征t-SNE二维降维可视化.pdf'), dpi=300, bbox_inches='tight')
plt.show()

# 保存交互式 Plotly 可视化
df_2d = pd.DataFrame({
    'X': X_tsne_2d[:, 0],
    'Y': X_tsne_2d[:, 1],
    '标注类别名称': df['标注类别名称'],
    '预测类别': df['top-1-预测名称'],
    '图像路径': df['图像路径']
})
df_2d.to_csv(os.path.join(output_dir, 't-SNE-2D.csv'), index=False)

pio.renderers.default = 'browser'
fig2d = px.scatter(df_2d, x='X', y='Y',
                   color='标注类别名称',
                   symbol='标注类别名称',
                   hover_name='图像路径',
                   opacity=0.8, width=1000, height=600)
fig2d.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig2d.write_html(os.path.join(output_dir, '语义特征t-SNE二维降维plotly可视化.html'))
fig2d.show()

# =================== t-SNE 降维至三维 =================

tsne_3d = TSNE(n_components=3, max_iter=10000)

X_tsne_3d = tsne_3d.fit_transform(encoding_array)

df_3d = pd.DataFrame({
    'X': X_tsne_3d[:, 0],
    'Y': X_tsne_3d[:, 1],
    'Z': X_tsne_3d[:, 2],
    '标注类别名称': df['标注类别名称'],
    '预测类别': df['top-1-预测名称'],
    '图像路径': df['图像路径']
})
df_3d.to_csv(os.path.join(output_dir, 't-SNE-3D.csv'), index=False)

fig3d = px.scatter_3d(df_3d, x='X', y='Y', z='Z',
                      color='标注类别名称',
                      symbol='标注类别名称',
                      hover_name='图像路径',
                      opacity=0.6, width=1000, height=800)
fig3d.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig3d.write_html(os.path.join(output_dir, '语义特征t-SNE三维降维plotly可视化.html'))
fig3d.show()
