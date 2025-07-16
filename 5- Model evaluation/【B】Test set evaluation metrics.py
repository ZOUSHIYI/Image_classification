import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report

# 载入类别ID到名称映射
idx_to_labels = np.load('E:/Image classification/3- Model training/output/idx_to_labels.npy', allow_pickle=True).item()
classes = list(idx_to_labels.values())
print('类别列表:', classes)

# 载入测试集预测结果表格
df = pd.read_csv('E:/Image classification/5- Model evaluation/output/测试集预测结果.csv')

# 计算 top-1 准确率
top1_acc = (df['标注类别名称'] == df['top-1-预测名称']).mean()
print(f'Top-1 准确率: {top1_acc:.4f}')

# 计算 top-n 准确率
topn_acc = df['top-n预测正确'].mean()
print(f'Top-{3} 准确率: {topn_acc:.4f}')

# 生成分类报告，输出字典格式方便后续处理
report_dict = classification_report(df['标注类别名称'], df['top-1-预测名称'], target_names=classes, output_dict=True)
# 删除accuracy键，后续手动添加更全面的准确率指标
report_dict.pop('accuracy', None)

# 转成 DataFrame 方便查看和保存
df_report = pd.DataFrame(report_dict).transpose()

# 计算每类准确率（等同于召回率 recall）
# 利用groupby和agg避免循环过滤
grouped = df.groupby('标注类别名称')
accuracy_per_class = grouped.apply(lambda x: (x['标注类别名称'] == x['top-1-预测名称']).mean())
accuracy_per_class = accuracy_per_class.reindex(classes)  # 保持顺序一致

# 计算宏平均和加权平均准确率
support = df_report.loc[classes, 'support']
acc_macro = accuracy_per_class.mean()
acc_weighted = np.average(accuracy_per_class, weights=support)

# 将准确率追加到 df_report 中
df_report['accuracy'] = accuracy_per_class.tolist() + [acc_macro, acc_weighted]

# 设置索引名称，方便保存
df_report.index.name = '类别'

# 保存目录检测与创建
output_dir = 'E:/Image classification/5- Model evaluation/output'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, '各类别准确率评估指标.csv')

df_report.to_csv(output_path, encoding='utf-8-sig')
print(f'评估指标保存至 {output_path}')
