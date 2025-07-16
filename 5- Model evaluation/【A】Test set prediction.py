import os
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets

# 设备配置：GPU优先
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# 图像预处理：Resize -> CenterCrop -> ToTensor -> Normalize
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 数据集路径
dataset_dir = 'E:\\Image classification\\1-Build a dataset\\dataset'
test_path = os.path.join(dataset_dir, 'val')

# 加载测试集，ImageFolder自动按文件夹名分类
test_dataset = datasets.ImageFolder(test_path, transform=test_transform)
print('测试集图像数量:', len(test_dataset))
print('类别个数:', len(test_dataset.classes))
print('类别名称:', test_dataset.classes)

# 载入类别ID与名称映射字典
idx_to_labels = np.load('E:\\Image classification\\3- Model training\\output\\idx_to_labels.npy', allow_pickle=True).item()

# 确认类别名称列表与映射一致
classes = [idx_to_labels[i] for i in range(len(idx_to_labels))]
print('类别名称(映射):', classes)

# 加载训练好的模型，设置为评估模式，移至设备
model = torch.load('E:\\Image classification\\3- Model training\\model\\best-0.868.pt')
model.eval()
model.to(device)

# 构造测试集数据路径和标签信息表
df = pd.DataFrame()
df['图像路径'] = [p for p, _ in test_dataset.imgs]
df['标注类别ID'] = test_dataset.targets
df['标注类别名称'] = [idx_to_labels[cls_id] for cls_id in test_dataset.targets]

print(df.head())

# 设置top-n预测个数
top_n = 3

results = []

# 逐行预测，带进度条
for row in tqdm(df.itertuples(), total=len(df), desc='预测测试集'):
    pred_dict = {}

    img_path = row.图像路径
    try:
        img_pil = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f'警告: 图像读取失败，跳过 {img_path}，错误：{e}')
        continue

    # 图像预处理
    input_tensor = test_transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)

    # 获取top-n预测ID和名称
    topk = torch.topk(probs, top_n)
    topk_ids = topk.indices[0].cpu().numpy()

    for i, pred_id in enumerate(topk_ids):
        pred_dict[f'top-{i+1}-预测ID'] = int(pred_id)
        pred_dict[f'top-{i+1}-预测名称'] = idx_to_labels[pred_id]

    # 判断标注类别是否在top-n预测内
    pred_dict['top-n预测正确'] = row.标注类别ID in topk_ids

    # 记录所有类别置信度
    for idx, cls_name in enumerate(classes):
        pred_dict[f'{cls_name}-预测置信度'] = float(probs[0][idx].cpu())

    # 额外保存原始信息
    pred_dict['图像路径'] = img_path
    pred_dict['标注类别ID'] = row.标注类别ID

    results.append(pred_dict)

# 构建预测结果DataFrame
df_pred = pd.DataFrame(results)

# 合并原始标签表与预测结果表
df_all = pd.concat([df.reset_index(drop=True), df_pred.reset_index(drop=True)], axis=1)

# 保存到CSV文件
output_csv = 'E:\\Image classification\\5- Model evaluation\\output\\测试集预测结果.csv'
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df_all.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f'预测结果保存至 {output_csv}')
