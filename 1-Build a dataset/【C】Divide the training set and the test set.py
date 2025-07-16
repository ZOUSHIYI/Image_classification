import os
import shutil
import random
import pandas as pd

# 设置参数
dataset_path = 'E:/Image classification/1-Build a dataset/dataset'
output_dir = 'E:/Image classification/1-Build a dataset/output'
os.makedirs(output_dir, exist_ok=True)

# 获取真实类别名，过滤掉 train、val 文件夹或隐藏文件
all_dirs = os.listdir(dataset_path)
classes = [c for c in all_dirs if os.path.isdir(os.path.join(dataset_path, c)) and c not in ['train', 'val'] and not c.startswith('.')]

# 打印数据集名
dataset_name = os.path.basename(dataset_path)
print('数据集:', dataset_name)
print('共 {} 个类别'.format(len(classes)))

# 创建 train 和 val 子目录
for phase in ['train', 'val']:
    phase_dir = os.path.join(dataset_path, phase)
    os.makedirs(phase_dir, exist_ok=True)
    for cls in classes:
        os.makedirs(os.path.join(phase_dir, cls), exist_ok=True)

# 划分比例和结果表格
test_frac = 0.2
random.seed(123)
df = pd.DataFrame(columns=['class', 'trainset', 'testset'])

print('{:^20} {:^20} {:^20}'.format('类别', '训练集数量', '测试集数量'))

# 执行划分与移动
for cls in classes:
    src_dir = os.path.join(dataset_path, cls)
    images = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    random.shuffle(images)

    num_test = int(len(images) * test_frac)
    test_images = images[:num_test]
    train_images = images[num_test:]

    # 移动测试集
    for img in test_images:
        src = os.path.join(src_dir, img)
        dst = os.path.join(dataset_path, 'val', cls, img)
        if os.path.exists(src):
            shutil.move(src, dst)

    # 移动训练集
    for img in train_images:
        src = os.path.join(src_dir, img)
        dst = os.path.join(dataset_path, 'train', cls, img)
        if os.path.exists(src):
            shutil.move(src, dst)

    # 输出统计信息
    print('{:^20} {:^20} {:^20}'.format(cls, len(train_images), len(test_images)))

    # 添加到表格
    df = pd.concat([
        df,
        pd.DataFrame({'class': [cls], 'trainset': [len(train_images)], 'testset': [len(test_images)]})
    ], ignore_index=True)

# 添加总数列，保存为 CSV
df['total'] = df['trainset'] + df['testset']
df.to_csv(os.path.join(output_dir, '数据量统计.csv'), index=False, encoding='utf-8-sig')
print('\n 数据集划分完成，统计已保存至 output/数据量统计.csv')
