import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
from matplotlib import font_manager

# ========== 1. 路径设置 ==========
dataset_path = 'E:/Image classification/1-Build a dataset/dataset'
output_dir = 'E:/Image classification/1-Build a dataset/output'
os.makedirs(output_dir, exist_ok=True)

# 设置中文字体 SimHei
font_path = 'E:/Image classification/SimHei.ttf'
my_font = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = my_font.get_name()
plt.rcParams['axes.unicode_minus'] = False

# ========== 2. 图像信息读取 ==========
records = []

for category in tqdm(os.listdir(dataset_path), desc="读取图像尺寸"):
    category_path = os.path.join(dataset_path, category)
    if not os.path.isdir(category_path):
        continue

    for file in os.listdir(category_path):
        file_path = os.path.join(category_path, file)
        try:
            img = Image.open(file_path).convert('RGB')
            width, height = img.size
            records.append([category, file, width, height])
        except Exception as e:
            print(f"[读取失败] {file_path} 错误信息：{e}")

# 构建 DataFrame
df = pd.DataFrame(records, columns=['类别', '文件名', '图像宽', '图像高'])

# ========== 3. 密度分布可视化 ==========
x = df['图像宽'].values
y = df['图像高'].values

# 计算密度
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

# 排序后绘图（密度低的先绘制）
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

plt.figure(figsize=(10, 10))
plt.scatter(x, y, c=z, s=5, cmap='Spectral_r', norm=LogNorm())
plt.tick_params(labelsize=14)

# 坐标设置
plt.xlim(0, max(df['图像宽']) + 50)
plt.ylim(0, max(df['图像高']) + 50)
plt.xlabel('图像宽度 (pixels)', fontsize=18)
plt.ylabel('图像高度 (pixels)', fontsize=18)
plt.title('图像尺寸密度分布图', fontsize=20)

# ========== 4. 保存图像 ==========
for fmt in ['pdf']:
    filename = os.path.join(output_dir, f'图像尺寸分布.{fmt}')
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[✓] 成功保存：{filename}")
    except Exception as e:
        print(f"[✗] 保存失败：{filename}，错误信息：{e}")

plt.show()
