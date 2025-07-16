import os
import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.models import resnet50
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask

# 中文字体路径，确保该路径正确存在你的字体文件
font_path = 'E:/Image classification/SimHei.ttf'
assert os.path.exists(font_path), f"字体文件不存在: {font_path}"
font = ImageFont.truetype(font_path, 50)

# 设置matplotlib中文字体和负号显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设备配置：有GPU就用GPU，否则用CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f' 当前设备: {device}')

# 图像路径，确保该路径存在图片文件
img_path = 'E:/Image classification/test_img/熊猫1.png'
assert os.path.exists(img_path), f"图像路径不存在: {img_path}"

# 指定展示的类别ID，如果为None则自动使用模型预测类别
show_class_id = 231  
# 是否显示中文类别名称
show_chinese = True  

# 加载预训练ResNet50模型，切换到eval模式，送入设备
model = resnet50(pretrained=True).eval().to(device)

# 创建SmoothGradCAM++解释器
cam_extractor = SmoothGradCAMpp(model)

# 定义图像预处理流程
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 载入ImageNet类别索引文件，生成ID到标签和中文标签的映射字典
label_df = pd.read_csv('E:/Image classification/imagenet_class_index.csv')
idx_to_labels = dict(zip(label_df['ID'], label_df['class']))
idx_to_labels_cn = dict(zip(label_df['ID'], label_df['Chinese']))

# 打开图像并预处理
img_pil = Image.open(img_path).convert('RGB')
input_tensor = test_transform(img_pil).unsqueeze(0).to(device)

# 开启梯度计算，计算模型预测结果（CAM需要梯度）
model.eval()
input_tensor.requires_grad_()  # 开启输入张量的梯度追踪
pred_logits = model(input_tensor)
pred_id = torch.topk(pred_logits, 1)[1].item()  # 获取预测类别ID

# 如果没有指定显示类别ID，则用预测类别ID
if show_class_id is None:
    show_class_id = pred_id

# 生成CAM热力图（SmoothGradCAM++方法）
activation_map = cam_extractor(show_class_id, pred_logits)[0][0].detach().cpu().numpy()

# 将热力图叠加到原图上
result = overlay_mask(img_pil, Image.fromarray(activation_map), alpha=0.6)

# 在图像上写文字（预测类别和展示类别）
draw = ImageDraw.Draw(result)
text_pred = f"Pred Class: {idx_to_labels_cn[pred_id] if show_chinese else idx_to_labels[pred_id]}"
text_show = f"Show Class: {idx_to_labels_cn[show_class_id] if show_chinese else idx_to_labels[show_class_id]}"
draw.text((50, 100), text_pred, font=font, fill=(255, 0, 0, 255))
draw.text((50, 200), text_show, font=font, fill=(255, 0, 0, 255))

# 显示结果图像
plt.figure(figsize=(8, 8))
plt.imshow(result)
plt.axis('off')
plt.title('CAM 可解释性分析')
plt.show()

# 保存结果图像
output_path = 'E:/Image classification/6-Interpretability analysis, significance analysis/output/CAM解释结果.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
result.save(output_path)
print(f"解释性分析图像已保存：{output_path}")
