import os
import time
import shutil

import cv2
from PIL import Image, ImageDraw, ImageFont
import mmcv

import torch
from torchcam.utils import overlay_mask
from torchvision.models import resnet50
from torchvision import transforms
import pandas as pd
from torchcam.methods import SmoothGradCAMpp

from tqdm import tqdm  # tqdm 进度条替代 mmcv.ProgressBar

# 设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

# 字体路径，注意改成你实际路径
font_path = 'E:/Image classification/SimHei.ttf'
font = ImageFont.truetype(font_path, 50)

# 载入ImageNet分类标签
df = pd.read_csv('E:/Image classification/imagenet_class_index.csv')
idx_to_labels = dict(zip(df['ID'], df['class']))
idx_to_labels_cn = dict(zip(df['ID'], df['Chinese']))

# 模型加载
model = resnet50(pretrained=True).eval().to(device)
cam_extractor = SmoothGradCAMpp(model)

# 预处理
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
])

def pred_single_frame(img, show_class_id=None, Chinese=True):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    input_tensor = test_transform(img_pil).unsqueeze(0).to(device)
    pred_logits = model(input_tensor)
    pred_id = torch.topk(pred_logits, 1)[1].item()
    show_id = show_class_id if show_class_id is not None else pred_id

    activation_map = cam_extractor(show_id, pred_logits)[0][0].detach().cpu().numpy()
    result = overlay_mask(img_pil, Image.fromarray(activation_map), alpha=0.7)

    draw = ImageDraw.Draw(result)
    if Chinese:
        text_pred = f'Pred Class: {idx_to_labels_cn[pred_id]}'
        text_show = f'Show Class: {idx_to_labels_cn[show_id]}'
    else:
        text_pred = f'Pred Class: {idx_to_labels[pred_id]}'
        text_show = f'Show Class: {idx_to_labels[show_id]}'
    draw.text((50, 100), text_pred, font=font, fill=(255, 0, 0, 255))
    draw.text((50, 200), text_show, font=font, fill=(255, 0, 0, 255))

    return result

# 视频路径
input_video = 'E:/Image classification/test_img/7月12日.mp4'
output_path = 'E:/Image classification/6-Interpretability analysis, significance analysis/output/output_pred.mp4'

# 临时文件夹
temp_out_dir = os.path.join('E:/Image classification/6-Interpretability analysis, significance analysis/output', time.strftime('%Y%m%d%H%M%S'))
os.makedirs(temp_out_dir, exist_ok=True)
print(f'创建文件夹 {temp_out_dir} 用于存放每帧预测结果')

imgs = mmcv.VideoReader(input_video)
prog_bar = tqdm(total=len(imgs), desc='Processing frames')

for frame_id, img in enumerate(imgs):
    out_img = pred_single_frame(img)
    out_img.save(f'{temp_out_dir}/{frame_id:06d}.jpg', "BMP")
    prog_bar.update(1)

prog_bar.close()

# 视频帧合成视频文件
mmcv.frames2video(temp_out_dir, output_path, fps=imgs.fps, fourcc='mp4v')

# 删除临时帧文件夹
shutil.rmtree(temp_out_dir)
print(f'删除临时文件夹 {temp_out_dir}')
print(f'视频已生成 {output_path}')
