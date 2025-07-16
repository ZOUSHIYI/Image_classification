import os
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from tqdm import tqdm

# 指定文件夹路径
folder_path = 'E:/Image classification/1-Build a dataset/dataset/train/大象'

# 要显示的图片数量
N = 36

# 计算网格大小 (向上取整)
n = math.ceil(math.sqrt(N))

# 检查路径是否存在
if not os.path.exists(folder_path):
    raise FileNotFoundError(f"Folder not found: {folder_path}")

# 读取图像
images = []
for i, filename in enumerate(os.listdir(folder_path)):
    if i >= N:
        break
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(f"跳过非图像文件: {filename}")
        continue

    img_path = os.path.join(folder_path, filename)
    try:
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        images.append(img_np)
    except Exception as e:
        print(f"读取图像 {filename} 错误: {e}")

print(f"成功加载 {len(images)} 张图像")

# 绘图
fig = plt.figure(figsize=(12, 12))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(n, n),
                 axes_pad=0.05,
                 share_all=True)

for ax, img in zip(grid, images):
    ax.imshow(img)
    ax.axis('off')

# 如果图片数量小于网格数量，隐藏多余的子图
for ax in grid[len(images):]:
    ax.axis('off')

plt.show()
