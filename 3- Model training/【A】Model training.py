import os
import time
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Matplotlib inline in notebook or fallback
try:
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
except:
    import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ==============================
# 1. 全局参数与设备配置
# ==============================
BATCH_SIZE = 32
EPOCHS = 100
dataset_dir = 'E:/Image classification/1-Build a dataset/dataset'
train_path = os.path.join(dataset_dir, 'train')
test_path = os.path.join(dataset_dir, 'val')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# 2. 数据预处理与加载
# ==============================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_dataloaders(train_path, test_path, batch_size):
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_path, transform=test_transform)

    os.makedirs("E:/Image classification/3- Model training/output", exist_ok=True)
    np.save('E:/Image classification/3- Model training/output/idx_to_labels.npy', {v: k for k, v in train_dataset.class_to_idx.items()})
    np.save('E:/Image classification/3- Model training/output/labels_to_idx.npy', train_dataset.class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_dataset, test_dataset, train_loader, test_loader

# ==============================
# 3. 模型构建与优化器
# ==============================
def build_model(n_class, finetune_strategy='partial_layer4'):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_class)

    if finetune_strategy == 'last_layer':
        # 只微调 fc 层
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        print("使用策略：仅微调最后一层 fc")

    elif finetune_strategy == 'partial_layer4':
        # 微调 layer4 + fc
        for name, param in model.named_parameters():
            param.requires_grad = ("layer4" in name or "fc" in name)
        print("使用策略：微调 layer4 + fc")

    elif finetune_strategy == 'all':
        # 微调全部层
        for param in model.parameters():
            param.requires_grad = True
        print("使用策略：微调所有层（从头训练）")

    else:
        raise ValueError(f"未知微调策略: {finetune_strategy}")
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    return model.to(device), optimizer


# ==============================
# 4. 单 batch 训练函数
# ==============================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

def train_one_batch(model, optimizer, images, labels, epoch, batch_idx):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, preds = torch.max(outputs, 1)

    return {
        'epoch': epoch,
        'batch': batch_idx,
        'train_loss': loss.item(),
        'train_accuracy': accuracy_score(labels.cpu(), preds.cpu())
    }

# ==============================
# 5. 测试集评估函数
# ==============================
def evaluate(model, test_loader, epoch):
    model.eval()
    loss_list, preds_list, labels_list = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            loss_list.append(loss.item())
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    return {
        'epoch': epoch,
        'test_loss': np.mean(loss_list),
        'test_accuracy': accuracy_score(labels_list, preds_list),
        'test_precision': precision_score(labels_list, preds_list, average='macro'),
        'test_recall': recall_score(labels_list, preds_list, average='macro'),
        'test_f1-score': f1_score(labels_list, preds_list, average='macro'),
    }

# ==============================
# 6. 主训练函数
# ==============================
def train_model():
    import wandb
    os.environ["WANDB_MODE"] = "offline"
    wandb.init(project="fruit32", name=time.strftime('%m%d%H%M%S'), mode="offline")

    train_dataset, test_dataset, train_loader, test_loader = get_dataloaders(train_path, test_path, BATCH_SIZE)
    n_class = len(train_dataset.classes)
    model, optimizer = build_model(n_class)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0
    os.makedirs("E:/Image classification/3- Model training/model", exist_ok=True)

    df_train_log = pd.DataFrame()
    df_test_log = pd.DataFrame()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        print(f"\nEpoch {epoch}/{EPOCHS}")

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            log_train = train_one_batch(model, optimizer, images, labels, epoch, batch_idx)
            df_train_log = pd.concat([df_train_log, pd.DataFrame([log_train])], ignore_index=True)
            wandb.log(log_train)

        scheduler.step()

        log_test = evaluate(model, test_loader, epoch)
        df_test_log = pd.concat([df_test_log, pd.DataFrame([log_test])], ignore_index=True)
        wandb.log(log_test)

        if log_test['test_accuracy'] > best_acc:
            best_acc = log_test['test_accuracy']
            best_path = f'E:/Image classification/3- Model training/model/best-{best_acc:.3f}.pt'
            torch.save(model, best_path)
            print(f"新最佳模型已保存至 {best_path}")

    df_train_log.to_csv('E:/Image classification/3- Model training/output/训练日志-训练集.csv', index=False)
    df_test_log.to_csv('E:/Image classification/3- Model training/output/训练日志-测试集.csv', index=False)
    print("训练完成，日志与模型已保存。")

    return model, test_loader, best_acc

# ==============================
# 7. 最佳模型测试
# ==============================
def test_best_model(best_acc, test_loader):
    model = torch.load(f'E:/Image classification/3- Model training/model/best-{best_acc:.3f}.pt')
    model.eval()
    result = evaluate(model, test_loader, epoch=0)
    print("\n最佳模型在测试集上的表现：")
    for k, v in result.items():
        print(f"{k}: {v:.4f}")

# ==============================
# 8. 启动训练流程
# ==============================
if __name__ == '__main__':
    model, test_loader, best_acc = train_model()
    test_best_model(best_acc, test_loader)
