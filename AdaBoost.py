import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import random

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

# 生成自适应增强树分类器模型
def AdaBoost(dataset, dataset_idx, q_rounds):
    # 获取类别列表和分组数目
    classification_list = np.unique(dataset[dataset.columns[-1]])
    # 对训练数据根据类别数量进行分组
    train_data = dataset.iloc[dataset_idx]
    data_split = {}
    for C in classification_list:
        data_split[C] = list(train_data[train_data[dataset.columns[-1]] == C].index)

    # 为每一个类别训练一组增强树
    AdaTree = {}
    for C in classification_list:
        print(f"正在生成类别{C}的增强树训练集")
        # 选取训练集并处理，将标签更改为1和0。1为当前类别，0为其他
        training_data = random_select(dataset, data_split, classification_list, C)
        features_train = training_data[training_data.columns[:-1]]
        labels_train = training_data[training_data.columns[-1]]
        print(f"正在生成类别{C}的增强树")
        # 训练当前类别的SVM分类器，n_estimators表示迭代的次数
        Ada = AdaBoostClassifier(n_estimators=q_rounds)
        Ada.fit(features_train, labels_train.astype(int))
        # test_data = dataset.iloc[test_idx]
        # test_data.loc[test_data['price_range'] != C, 'price_range'] = -1
        # t = Ada.predict(test_data.iloc[:, :-1])
        # print("测试集准确率：", accuracy_score(t, test_data['price_range']))
        # 将分类器存放到字典里，形式为[类别：Ada]
        AdaTree[C] = Ada
        print("=" * 40)
    print(f"已完成各类别增强树分类器的生成！")
    print("=" * 40)
    return AdaTree


import DatasetProcessing as DP
from Predict import classifyAda
from sklearn.metrics import accuracy_score

# if __name__ == '__main__':
#     # 获取数据集
#     dataset = DP.create_dataset()
#     classification = np.unique(dataset[dataset.columns[-1]])
#     # 8：2 划分数据集为训练集和测试集，这里划分的是index
#     train_idx, test_idx = DP.dataset_split(dataset.shape[0], random_state=0)
#     # 根据index获取测试集
#     testing_data = dataset.iloc[test_idx]
#     # 训练自适应增强树分类器
#     ada = AdaBoost(dataset, train_idx, 10) # 25也好
#
#     # 获取预测结果
#     predicted = classifyAda(ada, testing_data, classification)
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