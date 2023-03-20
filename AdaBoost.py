import numpy as np
from sklearn.ensemble import AdaBoostClassifier

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
    training_data = dataset.iloc[train_idx]
    # 得到训练集标签名称
    label_class = training_data.columns[-1]
    for i in train_idx:
        if training_data.loc[i, label_class] != C:
            training_data.loc[i, label_class] = 0
        else:
            training_data.loc[i, label_class] = 1
    return training_data

def AdaBoost(dataset, dataset_idx, q_rounds):
    # 获取类别列表和分组数目
    classification_list = np.unique(dataset[dataset.columns[-1]])
    train_set_num = len(classification_list)
    # 对训练数据根据类别数量进行分组
    train_idx = fold_split(dataset_idx, n_fold=train_set_num)

    # 为每一个类别训练一组增强树
    AdaTree = {}
    for i in range(len(classification_list)):
        # 选取训练集
        print(f"正在生成类别{classification_list[i]}的增强树训练集")
        training_data = training_data_process(dataset, train_idx[i], classification_list[i])
        features_train = training_data[training_data.columns[:-1]]
        labels_train = training_data[training_data.columns[-1]]

        # n_estimators表示迭代的次数
        print(f"正在生成类别{classification_list[i]}的增强树")
        Ada = AdaBoostClassifier(n_estimators=q_rounds)
        Ada.fit(features_train, labels_train.astype(int))

        AdaTree[classification_list[i]] = Ada
        print("=" * 40)

    print(f"已完成各类别增强树分类器的生成！")
    print("=" * 40)
    return AdaTree
