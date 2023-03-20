from math import log
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score

# 给训练集分组
def fold_split(dataset_idx, n_fold):
    # 获取训练样本数量
    n_samples = len(dataset_idx)
    # 确定每一份训练集的大小
    fold_sizes = np.floor(n_samples / n_fold) * np.ones(n_fold, dtype=int)
    # 分配剩下的样本
    r = n_samples % n_fold
    for i in range(r):
        fold_sizes[i] += 1

    train_indices = []
    # 这里采用的是交叉验证的方式划分训练集，比如要划分11份训练集，那就将样本划分11份，然后每个样本依次选10份
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + int(fold_size)
        test_mask = np.zeros(n_samples, dtype=np.bool_)
        test_mask[start:stop] = True
        train_mask = np.logical_not(test_mask)
        train_indices.append(dataset_idx[train_mask])
        current = stop

    return train_indices

# 处理训练集的标签，分为C类和no类
def training_data_process(dataset, train_idx, C):
    # 根据索引获取训练集数据
    training_data = dataset.iloc[train_idx]
    # 得到训练集标签名称
    label_class = training_data.columns[-1]
    # 若当前实例属于当前类别标签就改为1，否则就改为0
    for i in train_idx:
        if training_data.loc[i, label_class] != C:
            training_data.loc[i, label_class] = 0
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
    return 0.5 * log(acc/(1-acc), 2)

def SVM(dataset, dataset_idx):
    # 获取类别列表和分组数目
    classification_list = np.unique(dataset[dataset.columns[-1]])
    train_set_num = len(classification_list)
    # 对训练数据根据类别数量进行分组
    train_idx = fold_split(dataset_idx, n_fold=train_set_num)

    # 为每一个类别训练一个一对多的SVM分类器
    svm = {}
    for i in range(len(classification_list)):
        print(f"正在生成类别{classification_list[i]}的SVM训练集")
        # 选取训练集并处理，将标签更改为1和0。1为当前类别，0为其他
        training_data = training_data_process(dataset, train_idx[i], classification_list[i])
        features_train = training_data[training_data.columns[:-1]]
        labels_train = training_data[training_data.columns[-1]]
        print(f"正在生成类别{classification_list[i]}的SVM模型")
        # 训练当前类别的SVM分类器
        SVM = OneVsRestClassifier(SVC(kernel='rbf')).fit(features_train, labels_train.astype(int))
        # 将分类器存放到字典里，形式为[类别：SVM]
        svm[classification_list[i]] = SVM
        print("="*40)
    print("正在计算SVM支持向量机的权重！")
    # 计算SVM权重，这里的权重为预测训练集数据的准确率，然后计算：1/2 * log(acc/(1-acc), 2)
    weigthSVM = weigth(svm, dataset.iloc[dataset_idx], classification_list)
    print(f"已完成各类别一对多SVM分类器的生成！")
    print("="*40)
    return svm, weigthSVM
