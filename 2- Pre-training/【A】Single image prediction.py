import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageFont, ImageDraw
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 设置matplotlib中文字体（替换为你自己的SimHei.ttf路径）
font_path = 'E:/Image classification/SimHei.ttf'
if os.path.exists(font_path):
    from matplotlib.font_manager import FontProperties
    simhei_font = FontProperties(fname=font_path)
    matplotlib.rcParams['font.family'] = simhei_font.get_name()
else:
    print("警告：未找到SimHei字体文件，中文可能无法显示正常。")

plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# 设备设置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# 加载预训练模型（ResNet152）
model = models.resnet152(pretrained=True)
model = model.eval().to(device)

# 载入 ImageNet 标签（CSV 格式：ID, wordnet, Chinese）
df_labels = pd.read_csv('E:/Image classification/imagenet_class_index.csv')
idx_to_labels = {row['ID']: [row['wordnet'], row['Chinese']] for _, row in df_labels.iterrows()}

# 图像预处理：缩放、裁剪、转Tensor、归一化
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 载入测试图像
img_path = 'E:/Image classification/1-Build a dataset/dataset/train/大象/7.jpeg'
img_pil = Image.open(img_path).convert('RGB')

# 预处理图像并转到设备
input_tensor = test_transform(img_pil).unsqueeze(0).to(device)

# 模型前向推理
with torch.no_grad():
    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1)

probs_np = probs.cpu().numpy()[0]  # 转为numpy数组

# 取置信度最高的前n个类别
n = 10
top_n = torch.topk(probs, n)
top_indices = top_n.indices.cpu().numpy().squeeze()
top_probs = top_n.values.cpu().numpy().squeeze()

# 加载中文字体用于PIL绘图（字号32）
font = ImageFont.truetype(font_path, 32) if os.path.exists(font_path) else None
draw = ImageDraw.Draw(img_pil)

# 在图像上绘制分类结果文字
for i in range(n):
    class_name = idx_to_labels[top_indices[i]][1]
    confidence = top_probs[i] * 100
    text = f'{class_name:<15} {confidence:.4f}%'
    print(text)
    if font:
        draw.text((50, 100 + 50 * i), text, font=font, fill=(255, 0, 0, 255))  # 红色文字

# 保存带文字的图像
os.makedirs('E:/Image classification/2- Pre-training/output', exist_ok=True)
img_pil.save('E:/Image classification/2- Pre-training/output/img_pred.jpg')

# 绘制图像和置信度柱状图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

ax1.imshow(img_pil)
ax1.axis('off')

ax2.bar(df_labels['ID'], probs_np, alpha=0.5, width=0.3, color='yellow', edgecolor='red', lw=3)
ax2.set_ylim([0, 1.0])
ax2.set_xlabel('类别', fontsize=20)
ax2.set_ylabel('置信度', fontsize=20)
ax2.tick_params(labelsize=16)
ax2.set_title(f'{os.path.basename(img_path)} 图像分类预测结果', fontsize=22)

plt.tight_layout()
fig.savefig('E:/Image classification/2- Pre-training/output/预测图+柱状图.jpg')
plt.show()

# 生成预测结果表格
pred_data = []
for i in range(n):
    class_id = int(top_indices[i])
    wordnet = idx_to_labels[class_id][0]
    pred_data.append({
        'Class': idx_to_labels[class_id][1],
        'Class_ID': class_id,
        'Confidence(%)': top_probs[i] * 100,
        'WordNet': wordnet
    })

pred_df = pd.DataFrame(pred_data)
print(pred_df)

