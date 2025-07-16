## 1-Build a dataset--构建数据集
【A】Image acquisition：自动爬取指定关键词（如“兔子”“熊猫”等）对应的图片，并下载保存。
【B】Statistical image information：对图像数据集中的所有图片尺寸进行统计分析，并绘制图像尺寸的密度分布图。
【C】Divide the training set and the test set：将图像分类数据集按训练集（train）和验证集（val）划分，并输出每类图像数量统计表。
【D】Image visualization：将指定文件夹中的图像以网格形式展示，便于快速预览某一类图像内容（如“训练集中的大象图片”）。
【E】Count the number of images：对图像分类数据集中各类别的样本数量进行可视化，包括总数量柱状图和训练集/测试集的堆叠柱状图。

## 2- Pre-training
【A】Single image prediction：使用 PyTorch 预训练的 ResNet152 模型对一张图像进行分类预测，并可视化预测结果、置信度柱状图及生成表格。
【B】Video prediction：对输入视频中的每一帧进行图像分类预测，并可视化分类结果及柱状图，最终合成为新的视频。

## 3- Model training
【A】Model training：基于 PyTorch 框架，使用 ResNet50 预训练模型，完成了从数据加载、模型构建、训练、评估、日志记录、模型保存到最佳模型测试的全过程。
【B】visualization：用于将图像分类模型的训练过程中的关键指标（如损失值、准确率、精确率等）绘制成折线图并保存为 PDF 图表，从而更直观地观察模型训练与评估的过程。

## 4- Use the trained model to predict images
【A】Predict new images：对单张图像进行分类预测并可视化结果，包括柱状图、标注图和Top-N分类表格。它是一个完整的推理与结果展示流程，适用于你已经训练好的图像分类模型的测试与可视化阶段。
【B】Prediction video：对输入视频的每一帧图像进行图像分类预测，并根据不同模式生成带有预测信息的视频输出，包含文字标签或柱状图两种展示方式。

## 5- Model evaluation
【A】Test set prediction：对测试集图像逐张进行分类模型预测，并将预测结果与真实标签整合保存的完整流程。
【B】Test set evaluation metrics:对测试集预测结果进行了详细的分类性能评估，计算了Top-1和Top-N准确率，并生成了包含每个类别准确率、宏平均和加权平均准确率的分类报告，最后将结果保存为CSV文件。
【C】confusion matrix：加载模型分类预测结果，计算并绘制混淆矩阵，支持标准和归一化两种形式。
【D】PR curve：对图像分类模型预测结果的精确率-召回率（PR）曲线绘制和平均精确度（AP）计算，支持单类别和多类别绘图。
【E】ROC curve：基于模型预测结果计算和绘制ROC曲线，并计算AUC指标，同时将AUC结果整合进已有评估表的完整流程。
【F】Bar chart：读取各类别评估指标CSV文件，并绘制指定指标（如召回率、精确率、F1分数、准确率、AP、AUC等）的柱状图，最后保存为PDF文件。
【G】Semantic features：基于已训练图像分类模型，批量提取测试集图像的语义特征并保存。
【H】t-SNE Dimensionality reduction visualization：基于已提取的图像语义特征，使用 t-SNE 算法进行二维和三维降维，并通过 Matplotlib 和 Plotly 分别生成静态和交互式的可视化图表。

## 6-Interpretability analysis, significance analysis
【A】Pytorch pre-trained ImageNet image classification - single image：利用预训练的 ResNet50 模型和 SmoothGradCAM++ 方法，对指定输入图像进行 CAM（Class Activation Map，类激活映射）可视化解释。
【B】Pytorch pre-trained ImageNet image classification-video file：利用预训练的 ResNet50 模型和 SmoothGradCAM++ 方法，对指定输入视频进行 CAM（Class Activation Map，类激活映射）可视化解释。
【C】Self-trained animal classification model - single image：基于用户自定义的图像分类模型，使用 GradCAM++ 方法对输入图像进行可解释性热力图生成和可视化展示，帮助用户理解模型预测时关注的图像区域。
【D】Self-trained animal classification model-video file：基于用户自定义的图像分类模型，使用 GradCAM++ 方法对输入视频进行可解释性热力图生成和可视化展示，帮助用户理解模型预测时关注的图像区域。

## 7-Image Classification Deployment
模型本地部署


