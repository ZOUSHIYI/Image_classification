import os
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms
from matplotlib import font_manager

# ========== 1. 设置设备 ==========
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# ========== 2. 中文字体 ==========
font_path = 'E:/Image classification/SimHei.ttf'
font = ImageFont.truetype(font_path, 32)
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# ========== 3. 加载标签、模型 ==========
idx_to_labels = np.load('E:/Image classification/3- Model training/output/idx_to_labels.npy', allow_pickle=True).item()
model = torch.load('E:/Image classification/3- Model training/model/best-0.868.pt', map_location=device)
model.eval().to(device)

# ========== 4. 图像路径 ==========
img_path = 'E:/Image classification/test_img/熊猫1.png'
img_pil = Image.open(img_path).convert('RGB')

# ========== 5. 图像预处理 ==========
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
input_tensor = test_transform(img_pil).unsqueeze(0).to(device)

# ========== 6. 模型推理 ==========
with torch.no_grad():
    logits = model(input_tensor)
    softmax_scores = F.softmax(logits, dim=1).cpu().numpy()[0]

# ========== 7. 可视化完整预测柱状图 ==========
plt.figure(figsize=(18, 6))
x = list(idx_to_labels.values())
y = softmax_scores * 100
bars = plt.bar(x, y, color='skyblue', edgecolor='black')
plt.bar_label(bars, fmt='%.2f', fontsize=12)
plt.xticks(rotation=45)
plt.xlabel('类别', fontsize=15)
plt.ylabel('置信度(%)', fontsize=15)
plt.title('图像分类预测结果', fontsize=20)
plt.tight_layout()
plt.show()

# ========== 8. 获取前 N 个分类 ==========
n = 10
top_n = torch.topk(F.softmax(logits, dim=1), n)
top_scores = top_n[0].cpu().numpy().squeeze() * 100
top_ids = top_n[1].cpu().numpy().squeeze()

# ========== 9. 图像绘制分类结果 ==========
draw = ImageDraw.Draw(img_pil)
for i in range(n):
    label = idx_to_labels[top_ids[i]]
    conf = top_scores[i]
    text = f"{label:<15} {conf:>6.2f}%"
    draw.text((50, 100 + 40 * i), text, font=font, fill=(255, 0, 0))

# ========== 10. 左图原图 + 右图柱状图 ==========
fig = plt.figure(figsize=(20, 6))

# 原图 + 分类结果
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(img_pil)
ax1.axis('off')
ax1.set_title('预测结果图', fontsize=20)

# 分类柱状图
ax2 = plt.subplot(1, 2, 2)
bars = ax2.bar(x, y, alpha=0.5, color='orange', edgecolor='red', lw=2)
plt.bar_label(bars, fmt='%.2f', fontsize=10)
plt.xlabel('类别', fontsize=15)
plt.ylabel('置信度(%)', fontsize=15)
plt.title('分类柱状图', fontsize=20)
plt.xticks(rotation=90)
plt.ylim([0, 110])
ax2.tick_params(labelsize=12)

# 保存图像
save_path = 'E:/Image classification/4- Use the trained model to predict images/output/预测图+柱状图.jpg'
os.makedirs('output', exist_ok=True)
fig.tight_layout()
fig.savefig(save_path)
plt.show()
print(f'保存图像至: {save_path}')

# ========== 11. 输出预测前 N 表格 ==========
pred_df = pd.DataFrame([
    {
        'Class': idx_to_labels[top_ids[i]],
        'Class_ID': int(top_ids[i]),
        'Confidence(%)': top_scores[i]
    }
    for i in range(n)
])
print(pred_df)
