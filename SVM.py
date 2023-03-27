from math import log
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import numpy as np
import random
from sklearn.metrics import accuracy_score


# 随机选择训练集
def random_select(dataset, data_split, classification, C):
    n_samples = 2 * len(data_split[C])
    # 记录剩余的类别
    n = (len(classification)-1)
    # 先抽取C类别中的数据
    train_indices = data_split[C]
    t = 0.5
    # 再随机抽取其他类别的数据
    for type in classification:
        if type != C:
            if len(data_split[C]) > round((t / n) * n_samples):
                random_value = random.sample(data_split[type], round((t / n) * n_samples))
                train_indices = train_indices + random_value
            else:
                train_indices = train_indices + data_split[type]
                n -= 1
                t = t - (len(data_split[type]) / n_samples)
    # 打乱抽取的索引
    random.shuffle(train_indices)
    # 根据索引获取训练集数据
    training_data = dataset.iloc[train_indices]
    # 得到训练集标签名称
    label_class = training_data.columns[-1]
    # 若当前实例不属于当前类别标签改为-1
    training_data.loc[training_data[label_class] != C, label_class] = -1
    return training_data

# 处理训练集的标签，分为C类和no类
def training_data_process(dataset, train_idx, C):
    # 根据索引获取训练集数据
    training_data = dataset.iloc[train_idx]
    # 得到训练集标签名称
    label_class = training_data.columns[-1]
    # 若当前实例属于当前类别标签就改为1，否则就改为0
    for i in train_idx:
        if training_data.loc[i, label_class] != C:
            training_data.loc[i, label_class] = -1
        else:
            training_data.loc[i, label_class] = 1
    return training_data

# 预测，最后加权集成分类结果
def weigth(SVM, train_data, classification_list):
    # 用所有类别的SVM预测一个实例，谁大就选谁
    predicted = []
    for i in range(train_data.shape[0]):
        # 依次取一个实例
        features_test = train_data.iloc[i:i+1, :-1]
        predict_dict = {}
        for C in classification_list:
            de_f = SVM[C].decision_function(features_test)
            predict_dict[C] = de_f[0]
        # 得到值最大的那个类别，即键名
        result = max(predict_dict, key=predict_dict.get)
        # 存储分类结果
        predicted.append(result)
    # 获取训练集真实值
    labels_test = train_data[train_data.columns[-1]]
    # 计算准确率
    acc = accuracy_score(predicted, labels_test)
    # 返回计算的权重
    return 0.5 * log(acc/(1-acc))

# 生成SVM分类器模型
def SVM(dataset, dataset_idx):
    # 获取类别列表和分组数目
    classification_list = np.unique(dataset[dataset.columns[-1]])
    # 对训练数据根据类别数量进行分组
    train_data = dataset.iloc[dataset_idx]
    data_split = {}
    for C in classification_list:
        data_split[C] = list(train_data[train_data[dataset.columns[-1]] == C].index)

    # 为每一个类别训练一个一对多的SVM分类器
    svm = {}
    for C in classification_list:
        print(f"正在生成类别{C}的SVM训练集")
        # 选取训练集并处理，将标签更改为1和0。1为当前类别，0为其他
        training_data = random_select(dataset, data_split, classification_list, C)
        features_train = training_data[training_data.columns[:-1]]
        labels_train = training_data[training_data.columns[-1]]
        print(f"正在生成类别{C}的SVM模型")
        # 训练当前类别的SVM分类器
        SVM = OneVsRestClassifier(SVC(kernel='rbf')).fit(features_train, labels_train.astype(int))
        # 将分类器存放到字典里，形式为[类别：SVM]
        svm[C] = SVM
        print("="*40)
    print("正在计算SVM支持向量机的权重！")
    # 计算SVM权重，这里的权重为预测训练集数据的准确率，然后计算：1/2 * log(acc/(1-acc), 2)
    weigth_SVM = weigth(svm, dataset.iloc[dataset_idx], classification_list)
    print("weigth_SVM:", weigth_SVM)
    print(f"已完成各类别一对多SVM分类器的生成！")
    print("="*40)
    return svm, weigth_SVM


import DatasetProcessing as DP
from Predict import classifySVM

# if __name__ == '__main__':
#     # 获取数据集
#     dataset = DP.create_dataset()
#     classification = np.unique(dataset[dataset.columns[-1]])
#     # 8：2 划分数据集为训练集和测试集，这里划分的是index
#     train_idx, test_idx = DP.dataset_split(dataset.shape[0], random_state=0)
#     # 训练一对多SVM分类器
#     svm, weigthSVM = SVM(dataset, train_idx)
#
#     # 根据index获取测试集
#     testing_data = dataset.iloc[test_idx]
#     # 获取预测结果
#     predicted = classifySVM(svm, weigthSVM, testing_data, classification)
#     print('每个实例的决策值:\n', predicted)
#     y_pred = []
#     # 依次得出测试实例的最终分类结果
#     for i in range(testing_data.shape[0]):
#         # 选取值最有可能的类别，即值最大的键名
#         y_pred.append(max(predicted[i], key=predicted[i].get))
#     print('分类结果如下:\n', y_pred)
#     # 获取真实值
#     y_true = testing_data[testing_data.columns[-1]]
#     print('真实值如下:\n', list(y_true))
#
#     # 计算准确率
#     acc = accuracy_score(y_pred, y_true)
#     print(f"准确率: {acc*100}%")