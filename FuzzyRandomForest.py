import pandas as pd
import numpy as np
import sys

def create_dataset():
    # Import the dataset and define the feature as well as the target datasets / columns#
    # names=["Age","BMI","Glucose","Insulin","HOMA","Leptin","Adiponectin","Resistin","MCP.1","class",]
    dataset = pd.read_csv('data/FuzzyData.data', names=["ID", "Clumpthickness", "Uniformitycellsize", "Uniformitycellshape",
                                 "marginaladhesion", "singleepithelialsize", "Barenuclei", "Chromatin", "Nucleoli",
                                 "Mitoses", "class", ])
    dataset = dataset.drop('ID', axis=1)

    return dataset

def kfold_split(n_samples, n_fold=10, random_state=np.nan):
    """
    划分训练集和测试集
    @param n_samples: 样本数量，这里划分的是索引下标
    @param n_fold: 划分训练集的数量，由决策树的数量决定
    @param random_state: 随机数
    @return: 返回n_fold个训练集和测试集
    """
    # 确定没有设置随机种子
    if np.isnan(random_state):
        np.random.seed(random_state)

    # 将样本 80% 划分给训练集
    n_train = round(n_samples * 0.8)

    # 确定每一份训练集的大小
    fold_sizes = np.floor(n_train / n_fold) * np.ones(n_fold - 1, dtype=int)

    # 分配剩下的样本
    r = n_train % n_fold
    for i in range(r):
        fold_sizes[i] += 1

    # 创建样本空间并随机打乱
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # 将20%的样本划分给测试集
    train_indices = []
    test_indices = indices[:n_samples - n_train]
    indices = indices[n_samples - n_train:]

    # 从训练数据中划分一份独立的训练集，增加鲁棒性
    fold_last = n_train // n_fold
    train_last = indices[:fold_last]
    indices = indices[fold_last:]
    n_train -= fold_last

    # 这里采用的是交叉验证的方式划分训练集，比如要划分10份训练集，那就将样本划分10份，然后每个样本依次选9份
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + int(fold_size)
        test_mask = np.zeros(n_train, dtype=np.bool_)
        test_mask[start:stop] = True
        train_mask = np.logical_not(test_mask)
        train_indices.append(indices[train_mask])
        current = stop

    train_indices.append(train_last[:])

    return train_indices, test_indices
# 计算当前集合的Gini系数
def calcGini(dataset):
    # 求总样本数
    num_of_examples = len(dataset[dataset.columns[0]])

    currentLabel = np.unique(dataset[dataset.columns[-1]])
    count = np.unique(dataset[dataset.columns[-1]], return_counts=True)[1]

    labelCnt = {}
    # 遍历整个样本集合
    for i in range(len(currentLabel)):
        labelCnt[currentLabel[i]] = count[i]

    # 得到了当前集合中每个标签的样本个数后，计算它们的p值
    for key in labelCnt:
        labelCnt[key] /= num_of_examples
        labelCnt[key] = labelCnt[key] * labelCnt[key]
    # 计算Gini系数
    Gini = 1 - sum(labelCnt.values())
    return Gini

def choose_best_feature(dataset):
    # 特征总数
    numFeatures = len(dataset.columns[:-1])

    # 当只有一个特征时
    if numFeatures == 1:
        return 0
    # 初始化最佳基尼系数
    bestGini = 1
    # 初始化最优特征
    index_of_best_feature = -1
    # 遍历所有特征，寻找最优特征和该特征下的最优切分点
    for i in range(numFeatures):
        # 记录当前特征名称
        column = dataset.columns[i]
        # 去重，每个属性值唯一
        uniqueVals = np.unique(np.array(dataset[column]))
        # Gini字典中的每个值代表以该值对应的键作为切分点对当前集合进行划分后的Gini系数
        Gini = {}
        # 对于当前特征的每个取值
        for value in uniqueVals:
            # 先求由该值进行划分得到的两个子集
            sub_dataset1 = dataset[dataset[column] == value].drop(column, axis=1)
            sub_dataset2 = dataset[dataset[column] != value].drop(column, axis=1)

            # 求两个子集占原集合的比例系数prob1 prob2
            prob1 = sub_dataset1.shape[0] / float(dataset.shape[0])
            prob2 = sub_dataset2.shape[0] / float(dataset.shape[0])

            # 计算子集1的Gini系数
            Gini_of_sub_dataset1 = calcGini(sub_dataset1)
            # 计算子集2的Gini系数
            Gini_of_sub_dataset2 = calcGini(sub_dataset2)

            # 计算由当前最优切分点划分后的最终Gini系数
            Gini[value] = prob1 * Gini_of_sub_dataset1 + prob2 * Gini_of_sub_dataset2

            # 更新最优特征和最优切分点
            if Gini[value] < bestGini:
                bestGini = Gini[value]
                index_of_best_feature = i
                best_split_point = value
    return index_of_best_feature, best_split_point

def create_decision_tree(dataset, deep=9):
    # 求出训练集所有样本的标签
    label_list = np.array(dataset[dataset.columns[-1]])
    # 有四个递归结束的情况：
    # 若数据集为空，说明分不出来，则输出"error"
    if dataset.empty:
        return "error"
    # 若当前集合的所有样本标签相等（即样本已被分“纯”）
    # 则直接返回该标签值作为一个叶子节点
    if len(np.unique(label_list)) <= 1:
        return label_list[0]
    # 若训练集的所有特征都被使用完毕，当前无可用特征，但样本仍未被分“纯”
    # 则返回所含样本最多的标签作为结果
    if len(dataset.columns) == 1:
        return np.unique(label_list)[np.argmax(np.unique(label_list, return_counts=True)[1])]
    # 若决策树的深度等于规定的深度
    # 则返回所含样本最多的标签作为结果
    if deep == 1:
        return np.unique(label_list)[np.argmax(np.unique(label_list, return_counts=True)[1])]

    # 下面是正式建树的过程
    # 选取进行分支的最佳特征的下标和最佳切分点
    index_of_best_feature, best_split_point = choose_best_feature(dataset)
    # 得到最佳特征
    features = dataset.columns[:-1]
    best_feature = features[index_of_best_feature]
    # 初始化决策树
    decision_tree = {best_feature: {}}
    # # 使用过当前最佳特征后将其删去
    # features.pop(index_of_best_feature)
    # # 子特征 = 当前特征（因为刚才已经删去了用过的特征）
    # sub_labels = features[:]
    # 递归调用create_decision_tree去生成新节点
    # 生成由最优切分点划分出来的二分子集
    sub_dataset1 = dataset[dataset[best_feature] == best_split_point].drop(best_feature, axis=1)
    sub_dataset2 = dataset[dataset[best_feature] != best_split_point].drop(best_feature, axis=1)
    # 构造左子树
    decision_tree[best_feature][best_split_point] = create_decision_tree(sub_dataset1, deep-1)
    # 构造右子树
    decision_tree[best_feature]['others'] = create_decision_tree(sub_dataset2, deep-1)

    return decision_tree

# 用上面训练好的决策树对新样本分类
def predict(decision_tree, test_example):
    # 根节点代表的属性
    first_feature = list(decision_tree.keys())[0]
    # second_dict是第一个分类属性的值（也是字典）
    second_dict = decision_tree[first_feature]

    # 对于second_dict中的每一个key
    for key in second_dict.keys():
        # 不等于'others'的key
        if key != 'others':
            if test_example[first_feature] == key:
            # 若当前second_dict的key的value是一个字典
                if type(second_dict[key]).__name__ == 'dict':
                    # 则需要递归查询
                    classLabel = predict(second_dict[key], test_example)
                # 若当前second_dict的key的value是一个单独的值
                else:
                    # 则就是要找的标签值
                    classLabel = second_dict[key]
            # 如果测试样本在当前特征的取值不等于key，就说明它在当前特征的取值属于'others'
            else:
                # 如果second_dict['others']的值是个字符串，则直接输出
                if isinstance(second_dict['others'], str):
                    classLabel = second_dict['others']
                # 如果second_dict['others']的值是个字典，则递归查询
                else:
                    classLabel = predict(second_dict['others'], test_example)
    return classLabel

def classify(data, frf):
    # Create new query instances by simply removing the target feature column from the original dataset and
    # convert it to a dictionary
    queries = data.iloc[:, :-1].to_dict(orient="records")

    # Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"])

    # Calculate the prediction accuracy
    for i in range(data.shape[0]):
        res = []
        for tree in frf:
            res.append(predict(tree, queries[i]))
        predicted.loc[i, "predicted"] = max(set(res), key=res.count)
    return predicted["predicted"]

def result(n_fold,treeDeep):
    dataset = create_dataset()
    train_idx, test_idx = kfold_split(dataset.shape[0], n_fold=n_fold, random_state=0)

    # 训练模糊随机森林生成决策树
    frf = []
    for i in range(len(train_idx)):
        training_data = dataset.iloc[train_idx[i]]
        # 使用cart算法生成决策树
        tree = create_decision_tree(training_data, deep=treeDeep)
        frf.append(tree)

    # 对新样本进行分类测试
    testing_data = dataset.iloc[test_idx]

    y_pred = classify(testing_data, frf)
    y_true = testing_data["class"]

    y_pred = np.array(y_pred).astype(str)
    y_true = np.array(y_true).astype(str)

    return y_pred, y_true


if __name__ == '__main__':
    n_fold = 10
    treeDeep = 9
    dataset = create_dataset()
    train_idx, test_idx = kfold_split(dataset.shape[0], n_fold=n_fold, random_state=0)

    # 训练模糊随机森林生成决策树
    frf = []
    for i in range(len(train_idx)):
        training_data = dataset.iloc[train_idx[i]]
        # 使用cart算法生成决策树
        tree = create_decision_tree(training_data, deep=treeDeep)
        frf.append(tree)

    # 对新样本进行分类测试
    testing_data = dataset.iloc[test_idx]

    y_pred = classify(testing_data, frf)
    y_true = testing_data["class"]

    y_pred = np.array(y_pred).astype(str)
    y_true = np.array(y_true).astype(str)

    acc = (np.sum(y_true == y_pred) / y_true.shape[0]) * 100
    print("Fold: Accuracy: {:.3f}".format(acc))