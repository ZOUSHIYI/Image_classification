import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchcam.methods import GradCAMpp
from torchcam.utils import overlay_mask
from matplotlib.font_manager import FontProperties

def setup_matplotlib_chinese_font(font_path=None):
    """
    加载中文字体，返回 FontProperties 对象。
    """
    if font_path and os.path.exists(font_path):
        myfont = FontProperties(fname=font_path)
    else:
        # 使用系统已安装的 SimHei 字体作为备用
        myfont = FontProperties(family='SimHei')
    plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示
    return myfont

def load_model(model_path, device):
    assert os.path.exists(model_path), f"模型文件不存在: {model_path}"
    model = torch.load(model_path, map_location=device)
    model.eval()
    model.to(device)
    return model

def load_class_mappings(idx_to_labels_path, labels_to_idx_path):
    assert os.path.exists(idx_to_labels_path), f"类别映射文件不存在: {idx_to_labels_path}"
    assert os.path.exists(labels_to_idx_path), f"类别映射文件不存在: {labels_to_idx_path}"
    idx_to_labels = np.load(idx_to_labels_path, allow_pickle=True).item()
    labels_to_idx = np.load(labels_to_idx_path, allow_pickle=True).item()
    return idx_to_labels, labels_to_idx

def get_preprocess_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

def predict_and_visualize(
    img_path,
    model,
    cam_extractor,
    idx_to_labels,
    labels_to_idx,
    myfont,
    show_class=None,
    device='cpu',
    alpha=0.4,
    show_chinese=True,
    save_path=None
):
    assert os.path.exists(img_path), f"图像路径不存在: {img_path}"
    
    test_transform = get_preprocess_transform()
    img_pil = Image.open(img_path).convert('RGB')
    input_tensor = test_transform(img_pil).unsqueeze(0).to(device)

    pred_logits = model(input_tensor)
    pred_id = torch.topk(pred_logits, 1)[1].detach().cpu().numpy().squeeze().item()

    if show_class:
        if show_chinese:
            assert show_class in labels_to_idx, f"类别名 {show_class} 不在映射字典中"
        class_id = labels_to_idx.get(show_class, pred_id)
        show_id = class_id
    else:
        show_id = pred_id

    activation_map = cam_extractor(show_id, pred_logits)
    activation_map = activation_map[0][0].detach().cpu().numpy()

    result = overlay_mask(img_pil, Image.fromarray(activation_map), alpha=alpha)

    plt.figure(figsize=(6, 6))
    plt.imshow(result)
    plt.axis('off')

    pred_label = idx_to_labels[pred_id] if not show_chinese else (show_class if show_class else idx_to_labels[pred_id])
    show_label = idx_to_labels[show_id] if not show_chinese else (show_class if show_class else idx_to_labels[show_id])

    plt.title(f"{os.path.basename(img_path)}\nPred: {pred_label}   Show: {show_label}", fontproperties=myfont, fontsize=14)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"保存结果图像到: {save_path}")

    plt.show()

    return result

if __name__ == "__main__":
    # --------- 用户自定义路径 ---------
    font_path = 'E:/Image classification/SimHei.ttf'
    model_path = 'E:/Image classification/3- Model training/model/best-0.868.pt'
    idx_to_labels_path = 'E:/Image classification/3- Model training/output/idx_to_labels.npy'
    labels_to_idx_path = 'E:/Image classification/3- Model training/output/labels_to_idx.npy'
    img_path = 'E:/Image classification/test_img/熊猫1.png'
    show_class = '熊猫'  # 指定展示类别，None则自动使用预测类别
    save_result_path = 'E:/Image classification/6-Interpretability analysis, significance analysis/output/熊猫1_cam.png'

    # --------- 设备 ---------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # --------- 设置中文字体 ---------
    myfont = setup_matplotlib_chinese_font(font_path)

    # --------- 加载模型和映射 ---------
    model = load_model(model_path, device)
    idx_to_labels, labels_to_idx = load_class_mappings(idx_to_labels_path, labels_to_idx_path)

    # --------- 创建CAM解释器 ---------
    cam_extractor = GradCAMpp(model)

    # --------- 执行预测与可视化 ---------
    predict_and_visualize(
        img_path,
        model,
        cam_extractor,
        idx_to_labels,
        labels_to_idx,
        myfont,
        show_class=show_class,
        device=device,
        alpha=0.4,
        show_chinese=True,
        save_path=save_result_path
    )
