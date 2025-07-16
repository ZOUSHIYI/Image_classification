import torch
from torchvision import transforms
from PIL import Image
import sqlite3
import os
import numpy as np

# 模型加载
model_path = 'E:/Image classification/3- Model training/model/best-0.868.pt'
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# 读取类别索引映射
class_index_path = 'E:/Image classification/3- Model training/output/idx_to_labels.npy'
class_index = np.load(class_index_path, allow_pickle=True).item()
class_names = [class_index[i] for i in range(len(class_index))]

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(file):
    image = Image.open(file).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()
        label = class_names[predicted.item()]
    return f"{label} ({confidence:.2%})"

def predict_video(file):
    # TODO: 视频处理逻辑
    return "视频中检测到对象（示例返回）"

def save_record(username, filename, result):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO records (username, filename, result) VALUES (?, ?, ?)",
                   (username, filename, result))
    conn.commit()
    conn.close()
