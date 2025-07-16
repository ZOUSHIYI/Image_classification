import os
import time
import shutil
import gc

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image, ImageDraw, ImageFont
import cv2
import mmcv

# ============ 设置字体 =============
font_path = 'E:/Image classification/SimHei.ttf'
# PIL 字体加载
font = ImageFont.truetype(font_path, 32)
# Matplotlib 字体配置
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rc('font', family='SimHei')

# ============ 设备设置 ============
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# ============ 加载模型与标签 ============
idx_to_labels = np.load('E:/Image classification/3- Model training/output/idx_to_labels.npy', allow_pickle=True).item()
model = torch.load('E:/Image classification/3- Model training/model/best-0.868.pt')
model.eval().to(device)

# ============ 图像预处理流程 ============
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============ 单帧预测 ============
def pred_single_frame(img, n=5):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    input_img = test_transform(img_pil).unsqueeze(0).to(device)
    pred_logits = model(input_img)
    pred_softmax = F.softmax(pred_logits, dim=1)
    top_n = torch.topk(pred_softmax, n)
    pred_ids = top_n[1].cpu().detach().numpy().squeeze()
    confs = top_n[0].cpu().detach().numpy().squeeze()

    draw = ImageDraw.Draw(img_pil)
    for i in range(n):
        text = f'{idx_to_labels[pred_ids[i]]:<15} {confs[i]:.3f}'
        draw.text((50, 100 + 50 * i), text, font=font, fill=(255, 0, 0, 255))

    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_bgr, pred_softmax


# ============ 绘制柱状图 ============
def draw_bar_frame(img, pred_softmax, frame_id, out_dir):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    axs[0].imshow(img_rgb)
    axs[0].axis('off')

    x = list(idx_to_labels.values())
    y = pred_softmax.cpu().detach().numpy()[0] * 100

    axs[1].bar(x, y, color='yellow', edgecolor='red', lw=3, width=0.4)
    axs[1].set_title('图像分类预测结果', fontsize=22)
    axs[1].set_xlabel('类别', fontsize=18)
    axs[1].set_ylabel('置信度 (%)', fontsize=18)
    axs[1].set_ylim([0, 100])
    axs[1].tick_params(labelsize=12)
    axs[1].set_xticklabels(x, rotation=90)

    plt.tight_layout()
    fig.savefig(f'{out_dir}/{frame_id:06d}.jpg')
    plt.close(fig)
    gc.collect()

# ============ 视频处理主函数 ============
from tqdm import tqdm  # 加在文件开头导入

def process_video(input_video, output_path, mode='text'):
    temp_dir = time.strftime('%Y%m%d%H%M%S')
    os.makedirs(temp_dir, exist_ok=True)
    print(f'创建临时文件夹 {temp_dir} 用于存放每帧预测结果')

    imgs = mmcv.VideoReader(input_video)
    total_frames = len(imgs)

    for frame_id, img in tqdm(enumerate(imgs), total=total_frames, desc='处理视频帧'):
        pred_img, pred_softmax = pred_single_frame(img, n=5)

        if mode == 'bar':
            draw_bar_frame(pred_img, pred_softmax, frame_id, temp_dir)
        else:
            cv2.imwrite(f'{temp_dir}/{frame_id:06d}.jpg', pred_img)

    mmcv.frames2video(temp_dir, output_path, fps=imgs.fps, fourcc='mp4v')
    shutil.rmtree(temp_dir)
    print(f'删除临时文件夹 {temp_dir}')
    print(f'视频已生成 {output_path}')


# ============ 执行 ============
# 方案一：原始图像+预测文字
#process_video('E:\\Image classification\\test_img\\7月12日.mp4', 'E:\\Image classification\\4- Use the trained model to predict images\\output\\output_pred.mp4', mode='text')

# 方案二：原始图像+预测文字+置信度柱状图
process_video('E:/Image classification/test_img/7月12日.mp4', 'E:/Image classification/4- Use the trained model to predict images/output/output_bar.mp4', mode='bar')
