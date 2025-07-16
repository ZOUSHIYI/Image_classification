import os
import time
import gc
import shutil
from tqdm import tqdm
import tempfile

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F
from torchvision import models, transforms

import mmcv

# 设备：GPU 或 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# 加载ResNet152预训练模型
model = models.resnet152(pretrained=True)
model.eval()
model.to(device)

# 载入标签映射，假设文件路径正确
import pandas as pd
df_labels = pd.read_csv('E:/Image classification/imagenet_class_index.csv')
idx_to_labels = {row['ID']: [row['wordnet'], row['Chinese']] for _, row in df_labels.iterrows()}

# 中文字体（请确保SimHei.ttf在当前目录）
font_path = 'E:/Image classification/SimHei.ttf'
if not os.path.exists(font_path):
    print("警告：SimHei.ttf字体文件未找到，中文可能无法显示")
font = ImageFont.truetype(font_path, 32) if os.path.exists(font_path) else None

# 图像预处理
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def pred_single_frame(img_bgr, top_n=5):
    """预测单张图像，返回带预测文字的图像（BGR）和所有类别的softmax概率"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    input_tensor = test_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
    
    topk = torch.topk(probs, top_n)
    pred_ids = topk.indices.cpu().numpy().squeeze()
    confs = topk.values.cpu().numpy().squeeze()
    
    draw = ImageDraw.Draw(img_pil)
    for i in range(top_n):
        label = idx_to_labels[pred_ids[i]][1] if pred_ids[i] in idx_to_labels else str(pred_ids[i])
        text = f"{label:<15} {confs[i]:.3f}"
        if font:
            draw.text((50, 100 + 50 * i), text, font=font, fill=(255,0,0,255))
        else:
            draw.text((50, 100 + 50 * i), text, fill=(255,0,0,255))
    
    img_bgr_out = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_bgr_out, probs

def pred_single_frame_bar(img_bgr, probs, save_path):
    """绘制预测结果条形图，并保存带条形图的复合图片"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(18,6))
    ax1 = plt.subplot(1,2,1)
    ax1.imshow(img_rgb)
    ax1.axis('off')
    
    ax2 = plt.subplot(1,2,2)
    x = range(1000)
    y = probs.cpu().numpy().squeeze()
    ax2.bar(x, y, alpha=0.5, width=0.3, color='yellow', edgecolor='red', lw=3)
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('类别', fontsize=20)
    ax2.set_ylabel('置信度', fontsize=20)
    ax2.tick_params(labelsize=16)
    plt.title('图像分类预测结果', fontsize=30)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    gc.collect()

# 输入视频路径
input_video = 'E:/Image classification/test_img/7月12日.mp4'


# 临时目录存放每帧图，改为系统临时目录
timestamp = time.strftime('%Y%m%d%H%M%S')
base_tmp_dir = tempfile.gettempdir()
temp_out_dir = os.path.join(base_tmp_dir, f'temp_{timestamp}')
os.makedirs(temp_out_dir, exist_ok=True)
print(f'创建临时文件夹：{temp_out_dir}')

# 读取视频
video = mmcv.VideoReader(input_video)
fps = video.fps

# 处理每帧
pbar = tqdm(total=len(video), desc="视频帧处理", ncols=100)
for frame_id, frame in enumerate(video):
    img_pred, prob = pred_single_frame(frame, top_n=5)
    save_path = os.path.join(temp_out_dir, f'{frame_id:06d}.jpg')
    pred_single_frame_bar(img_pred, prob, save_path)
    pbar.update(1)
pbar.close()

# 合成视频
output_video = 'E:/Image classification/2- Pre-training/output/output_bar.mp4'
mmcv.frames2video(temp_out_dir, output_video, fps=fps, fourcc='mp4v')
print(f'已生成输出视频：{output_video}')

# 删除临时文件夹
shutil.rmtree(temp_out_dir)
print(f'已删除临时文件夹：{temp_out_dir}')
