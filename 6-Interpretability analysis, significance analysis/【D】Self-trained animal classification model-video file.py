import os
import time
import shutil
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import mmcv
import torch
from torchcam.utils import overlay_mask
from torchvision import transforms
from torchcam.methods import SmoothGradCAMpp

def run_video_cam(
    input_video='E:/Image classification/test_img/7月12日.mp4',
    output_path='E:/Image classification/6-Interpretability analysis, significance analysis/output_pred1.mp4',
    font_path='E:/Image classification/SimHei.ttf',
    model_path='E:/Image classification/3- Model training/model/best-0.868.pt',
    idx_to_labels_path='E:/Image classification/3- Model training/output/idx_to_labels.npy',
    alpha=0.7
):
    # 设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # 字体
    assert os.path.exists(font_path), f"字体文件不存在: {font_path}"
    font = ImageFont.truetype(font_path, 50)

    # 载入类别字典
    assert os.path.exists(idx_to_labels_path), f"类别文件不存在: {idx_to_labels_path}"
    idx_to_labels_cn = np.load(idx_to_labels_path, allow_pickle=True).item()

    # 载入模型
    assert os.path.exists(model_path), f"模型文件不存在: {model_path}"
    model = torch.load(model_path, map_location=device).eval().to(device)

    # 创建 CAM 解释器
    cam_extractor = SmoothGradCAMpp(model)

    # 预处理
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    def pred_single_frame(img, show_class_id=None):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        input_tensor = test_transform(img_pil).unsqueeze(0).to(device)
        pred_logits = model(input_tensor)
        pred_id = torch.topk(pred_logits, 1)[1].detach().cpu().numpy().squeeze().item()
        show_id = show_class_id if show_class_id is not None else pred_id

        activation_map = cam_extractor(show_id, pred_logits)
        activation_map = activation_map[0][0].detach().cpu().numpy()
        result = overlay_mask(img_pil, Image.fromarray(activation_map), alpha=alpha)

        draw = ImageDraw.Draw(result)
        text_pred = f'Pred Class: {idx_to_labels_cn[pred_id]}'
        text_show = f'Show Class: {idx_to_labels_cn[show_id]}'
        draw.text((50, 100), text_pred, font=font, fill=(255, 0, 0, 255))
        draw.text((50, 200), text_show, font=font, fill=(255, 0, 0, 255))
        return result

    # 创建临时文件夹
    temp_out_dir = os.path.join('E:/Image classification/6-Interpretability analysis, significance analysis/output', time.strftime('%Y%m%d%H%M%S'))
    os.makedirs(temp_out_dir, exist_ok=True)
    print(f'创建文件夹 {temp_out_dir} 用于存放每帧预测结果')

    # 读取视频
    imgs = mmcv.VideoReader(input_video)

    # 进度条
    from tqdm import tqdm
    prog_bar = tqdm(total=len(imgs), desc="Processing frames")

    for frame_id, img in enumerate(imgs):
        out_img = pred_single_frame(img)
        out_img.save(f'{temp_out_dir}/{frame_id:06d}.jpg', "BMP")
        prog_bar.update(1)

    prog_bar.close()

    # 合成视频
    mmcv.frames2video(temp_out_dir, output_path, fps=imgs.fps, fourcc='mp4v')

    # 清理临时文件夹
    shutil.rmtree(temp_out_dir)
    print(f'删除临时文件夹 {temp_out_dir}')
    print(f'视频已生成: {output_path}')


if __name__ == "__main__":
    # 直接运行时调用函数，参数可根据需要调整
    run_video_cam(
        input_video='E:/Image classification/test_img/7月12日.mp4',
        output_path='E:/Image classification/6-Interpretability analysis, significance analysis/output/output_pred1.mp4',
        font_path='E:/Image classification/SimHei.ttf',
        model_path='E:/Image classification/3- Model training/model/best-0.868.pt',
        idx_to_labels_path='E:/Image classification/3- Model training/output/idx_to_labels.npy',
        alpha=0.7
    )
