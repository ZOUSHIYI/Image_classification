import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

# ----------------------- 基本配置 -----------------------
warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('当前设备:', device)

# 模型路径和预测表路径（可修改）
model_path = 'E:/Image classification/3- Model training/model/best-0.868.pt'
result_csv_path = 'E:/Image classification/5- Model evaluation/output/测试集预测结果.csv'
save_npy_path = 'E:/Image classification/5- Model evaluation/output/测试集语义特征.npy'

# 图像预处理
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------- 加载模型 -----------------------
model = torch.load(model_path, map_location=device)
model = model.eval().to(device)

# 提取 avgpool 中间层作为语义特征
model_trunc = create_feature_extractor(model, return_nodes={'avgpool': 'semantic_feature'})
print('模型加载完成，已截断至 avgpool 层')

# ----------------------- 单图特征提取函数 -----------------------
def extract_feature(img_path):
    try:
        with Image.open(img_path).convert('RGB') as img_pil:
            input_tensor = test_transform(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model_trunc(input_tensor)['semantic_feature']
            return features.squeeze().detach().cpu().numpy()
    except Exception as e:
        print(f"读取失败: {img_path} | 错误: {e}")
        return None

# ----------------------- 批量计算语义特征 -----------------------
df = pd.read_csv(result_csv_path)
print(f'共有 {len(df)} 张图像待处理')

semantic_features = []
valid_img_paths = []

for img_path in tqdm(df['图像路径'], desc='🚀 提取语义特征中'):
    feature = extract_feature(img_path)
    if feature is not None:
        semantic_features.append(feature)
        valid_img_paths.append(img_path)

semantic_array = np.array(semantic_features)
print('特征提取完成，shape:', semantic_array.shape)

# ----------------------- 保存结果 -----------------------
os.makedirs(os.path.dirname(save_npy_path), exist_ok=True)
np.save(save_npy_path, semantic_array)
print(f'特征已保存至: {save_npy_path}')
