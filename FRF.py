import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import random
import math
import sys

def create_dataset():
    # Import the dataset and define the feature as well as the target datasets / columns#
    # names=["Age","BMI","Glucose","Insulin","HOMA","Leptin","Adiponectin","Resistin","MCP.1","class",]
    # dataset = pd.read_csv('data/FuzzyData.data', names=["ID", "Clumpthickness", "Uniformitycellsize", "Uniformitycellshape",
    #                              "marginaladhesion", "singleepithelialsize", "Barenuclei", "Chromatin", "Nucleoli",
    #                              "Mitoses", "class", ])
    # dataset = dataset.drop('ID', axis=1)
    dataset = pd.read_csv('data/drug200.data')

    for i in range(len(dataset['BP'])):
        if dataset.loc[i, 'BP'] == 'HIGH':
            dataset.loc[i, 'BP'] = random.randint(667, 1000)
        if dataset.loc[i, 'BP'] == 'LOW':
            dataset.loc[i, 'BP'] = random.randint(0, 333)
        if dataset.loc[i, 'BP'] == 'NORMAL':
            dataset.loc[i, 'BP'] = random.randint(333, 667)

        if dataset.loc[i, 'Cholesterol'] == 'HIGH':
            dataset.loc[i, 'Cholesterol'] = round(random.uniform(0.7, 1), 4)
        if dataset.loc[i, 'Cholesterol'] == 'NORMAL':
            dataset.loc[i, 'Cholesterol'] = round(random.uniform(0.3, 0.7), 4)

        if dataset.loc[i, 'Sex'] == 'F':
            dataset.loc[i, 'Sex'] = 1
        else:
            dataset.loc[i, 'Sex'] = 0

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

# 处理训练集的标签，分为C类和no类
def training_data_process(dataset, train_idx, C):
    training_data = dataset.iloc[train_idx]
    # 得到训练集标签名称
    label_class = training_data.columns[-1]
    for i in train_idx:
        if training_data.loc[i, label_class] != C:
            training_data.loc[i, label_class] = 'no'
    return training_data

# 判断该特征下的数据是否为连续值
def judge_continuous(data_feature):
    if len(np.unique(data_feature)) < len(data_feature) / 3:
        return 0
    else:
        return 1

# 计算特征数据的隶属度，返回隶属度列表
def membership_value_list(data_features, c, a):
    ans = []
    if judge_continuous(data_features) == 0:
        for i in range(len(data_features)):
            x = data_features[i]
            ans.append(membership_value(x, c, a, 0))
    else:
        for i in range(len(data_features)):
            x = data_features[i]
            ans.append(membership_value(x, c, a, 1))
    return np.array(ans)

# 计算x的隶属度
def membership_value(x, c, a, continuous):
    ans = 0
    if continuous == 0:
        if x <= c - a:
            ans = 0
        elif c-a < x <= c:
            ans = (a-c+x) / a
        elif c < x < c+a:
            ans = (a+c-x) / a
        elif x >= c+a:
            ans = 0
    else:
        if x <= c:
            ans = 1
        elif c < x < c+a:
            ans = (a+c-x) / a
        elif x >= c+a:
            ans = 0
    return ans

# 计算当前集合的熵
def calcEntropy(dataset, name_feature, c, a):

    data_features = np.array(dataset[name_feature])
    label_list = np.array(dataset[dataset.columns[-1]])
    u_list = membership_value_list(data_features, c, a)
    # N1 = u_list.sum()
    # N2 = len(data_features) - N1

    C1_list = [0, 0]
    C2_list = [0, 0]
    label = np.unique(label_list)
    for i in range(len(label_list)):
        if label_list[i] == label[0]:
            C1_list[0] += u_list[i]
            C2_list[0] += (1 - u_list[i])
        else:
            C1_list[1] += u_list[i]
            C2_list[1] += (1 - u_list[i])
    # C1 = label[C1_list.index(max(C1_list))]
    # C2 = label[C2_list.index(max(C2_list))]
    # E1, E2 = 0, 0
    # for i in range(len(label_list)):
    #     if label_list[i] != C1:
    #         E1 += u_list[i]
    #     if label_list[i] != C2:
    #         E2 += (1 - u_list[i])
    E1 = max(C1_list)
    E2 = max(C2_list)

    Entropy = ((E1 + E2) / len(data_features))
    return Entropy

# 选择最佳特征
def choose_best_feature(dataset):
    # 特征总数
    numFeatures = len(dataset.columns[:-1])

    # 当只有一个特征时
    if numFeatures == 0:
        return 0
    # 初始化最佳熵
    bestEntropy = 0
    # 初始化最优特征
    index_of_best_feature = -1
    best_c = -1
    best_a = -1
    # 遍历所有特征，寻找最优特征和该特征下的最优切分点
    for i in range(numFeatures):
        # 记录当前特征名称
        name_feature = dataset.columns[i]
        # 去重，每个属性值唯一
        uniqueVals = np.unique(np.array(dataset[name_feature]))
        # 对于当前特征的每个取值
        for c in uniqueVals:
            # 确定参数a
            b = 0.1
            a = c * b
            while b > 0.01 and a < 0.35*c:
                # 计算的熵
                Entropy = calcEntropy(dataset, name_feature, c, a)
                # 更新最优特征和最优切分点
                if Entropy > bestEntropy:
                    bestEntropy = Entropy
                    index_of_best_feature = i
                    best_c = c
                    best_a = a
                    a = a + c * b
                else:
                    b = b / 2
                    a = a - c * b
    return index_of_best_feature, best_c, best_a

def create_decision_tree(dataset, tree_type, deep=9):
    # 求出训练集所有样本的标签
    label_list = np.array(dataset[dataset.columns[-1]])
    # 有四个递归结束的情况：
    if dataset.empty:
        return 'no' if tree_type == 'decision' else 0
    # 若当前集合的所有样本标签相等（即样本已被分“纯”）
    # 则直接返回该标签值作为一个叶子节点
    if len(np.unique(label_list)) <= 1:
        return label_list[0] if tree_type == 'decision' else np.sum(label_list == 1)/len(label_list)
    # 若训练集的所有特征都被使用完毕，当前无可用特征，但样本仍未被分“纯”
    # 则返回所含样本最多的标签作为结果
    if len(dataset.columns) == 1:
        result = np.unique(label_list)[np.argmax(np.unique(label_list, return_counts=True)[1])]
        return result if tree_type == 'decision' else np.sum(label_list == 1)/len(label_list)
    # 若决策树的深度等于规定的深度
    # 则返回所含样本最多的标签作为结果
    if deep == 1:
        result = np.unique(label_list)[np.argmax(np.unique(label_list, return_counts=True)[1])]
        return result if tree_type == 'decision' else np.sum(label_list == 1)/len(label_list)

    # 下面是正式建树的过程
    # 选取进行分支的最佳特征的下标和最佳切分点
    index_of_best_feature, best_c, best_a = choose_best_feature(dataset)
    # 得到最佳特征
    best_feature = dataset.columns[:-1][index_of_best_feature]
    # 初始化决策树
    decision_tree = {best_feature: {}}
    # 确定选定的特征是否为连续值
    continuous = judge_continuous(dataset[best_feature])
    if continuous == 1:
        # 生成由最优切分点划分出来的二分子集
        sub_dataset1 = dataset[dataset[best_feature] < (best_c + best_a)].drop(best_feature, axis=1)
        sub_dataset2 = dataset[dataset[best_feature] > best_c].drop(best_feature, axis=1)
    else:
        # 生成由最优切分点划分出来的二分子集
        sub_dataset1 = dataset[(dataset[best_feature] > (best_c - best_a)) & (dataset[best_feature] < (best_c + best_a))].drop(best_feature, axis=1)
        sub_dataset2 = dataset[dataset[best_feature] != best_c].drop(best_feature, axis=1)
    # 构造左子树
    decision_tree[best_feature]['left'] = create_decision_tree(sub_dataset1, tree_type, deep-1)
    # 构造右子树
    decision_tree[best_feature]['right'] = create_decision_tree(sub_dataset2, tree_type, deep-1)
    decision_tree[best_feature]['parameter'] = {'continuous': continuous, 'c': best_c, 'a': best_a}

    return decision_tree

# 用上面训练好的决策树对新样本分类
def predict(decision_tree, test_example, leaf_list, weight, weight_list):
    if not isinstance(decision_tree, dict):
        # 则就是要找的标签值
        leaf_list.append(decision_tree)
        weight_list.append(weight)
        return leaf_list, weight_list
    # 根节点代表的属性
    first_feature = list(decision_tree.keys())[0]
    # second_dict是第一个分类属性的值（也是字典）
    second_dict = decision_tree[first_feature]
    parameter = second_dict['parameter']
    # print(parameter, first_feature, test_example[first_feature])
    keys = []
    weights = {}
    if parameter['continuous'] == 1:
        u = membership_value(test_example[first_feature], parameter['c'], parameter['a'], 1)
        if parameter['a'] == 0:
            keys.append('left')
            weights['left'] = weight
        if test_example[first_feature] < parameter['c'] + parameter['a']:
            keys.append('left')
            weights['left'] = weight * u
        if test_example[first_feature] > parameter['c']:
            keys.append('right')
            weights['right'] = weight * (1 - u)
    else:
        u = membership_value(test_example[first_feature], parameter['c'], parameter['a'], 0)
        if parameter['a'] == 0 and test_example[first_feature] == parameter['c']:
            keys.append('left')
            weights['left'] = weight
        if parameter['c'] - parameter['a'] < test_example[first_feature] < parameter['c'] + parameter['a']:
            keys.append('left')
            weights['left'] = weight * u
        if test_example[first_feature] != parameter['c']:
            keys.append('right')
            weights['right'] = weight * (1 - u)
    # print('keys:', keys)
    # print('weights:', weights)
    for key in keys:
        # 若当前second_dict的key的value是一个字典
        # if type(second_dict[key]).__name__ == 'dict':
        if isinstance(second_dict[key], dict):
            # 则需要递归查询
            predict(second_dict[key], test_example, leaf_list, weights[key], weight_list)
        # 若当前second_dict的key的value是一个单独的值
        else:
            # 则就是要找的标签值
            classLabel = second_dict[key]
            leaf_list.append(classLabel)
            weight_list.append(weights[key])
    return leaf_list, weight_list

# 处理训练数据，返回错误树训练数据
def process_error_train_data(train_idx, training_data, decision_tree):
    queries = training_data.iloc[:, :-1].to_dict(orient="records")
    for i in range(training_data.shape[0]):
        predict_result_list, weight_list = predict(decision_tree, queries[i], [], 1, [])
        type = np.unique(np.array(predict_result_list))
        temp = [0]*len(type)
        for j in range(len(predict_result_list)):
            for k in range(len(type)):
                if predict_result_list[j] == type[k]:
                    temp[k] += weight_list[j]
        predict_result = type[np.argmax(np.array(temp))]
        true_result = training_data.loc[train_idx[i], training_data.columns[-1]]
        if predict_result == true_result:
            training_data.loc[train_idx[i], training_data.columns[-1]] = 1
        else:
            training_data.loc[train_idx[i], training_data.columns[-1]] = 0
    return training_data

def classify(testing_data, frf, classification, n_fold):
    # Create new query instances by simply removing the target feature column from the original dataset and
    # convert it to a dictionary
    queries = testing_data.iloc[:, :-1].to_dict(orient="records")

    # Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"])

    # Calculate the prediction accuracy
    for i in range(testing_data.shape[0]):
        predict_dict = {}
        for C in classification:
            predict_dict[C] = 0
            for j in range(n_fold):
                leaf_decision_tree, weight_decision_tree = predict(frf[C]['decision_tree_'+str(j+1)], queries[i], [], 1, [])
                result_decision_tree = 0
                for k in range(len(leaf_decision_tree)):
                    if leaf_decision_tree[k] == C:
                        result_decision_tree += weight_decision_tree[k]

                leaf_error_tree, weight_error_tree = predict(frf[C]['error_tree_'+str(j+1)], queries[i], [], 1, [])
                print('leaf_error_tree:', leaf_error_tree)
                print('weight_error_tree:', weight_error_tree)
                result_error_tree = 0
                for k in range(len(leaf_error_tree)):
                    result_error_tree += weight_error_tree[k] * leaf_error_tree[k]

                predict_dict[C] += result_decision_tree * result_error_tree

        predicted.loc[i, "predicted"] = max(predict_dict,key=predict_dict.get)
        # print(f'predict_dict:', predict_dict)
    return predicted["predicted"]

# if __name__ == '__main__':
def main():
    n_fold = 10
    treeDeep = 9
    dataset = create_dataset()

    train_idx, test_idx = kfold_split(dataset.shape[0], n_fold=n_fold, random_state=0)
    # 训练模糊随机森林生成决策树
    frf = {}
    for C in np.unique(dataset[dataset.columns[-1]]):
        temp = {}
        for i in range(len(train_idx)):
            # 选取训练集
            print(f"正在生成类别{C}的{i+1}号决策树训练集")
            training_data = training_data_process(dataset, train_idx[i], C)
            # 生成决策树
            print(f"正在生成类别{C}的{i+1}号模糊决策树")
            decision_tree = create_decision_tree(training_data, 'decision', deep=treeDeep)
            # 获取错误树训练数据
            print(f"正在生成类别{C}的{i+1}号错误树训练集")
            error_train_data = process_error_train_data(train_idx[i], training_data.copy(), decision_tree)
            # 生成错误树
            print(f"正在生成类别{C}的{i+1}号错误树")
            error_tree = create_decision_tree(error_train_data, 'error', deep=treeDeep)
            temp['decision_tree_'+str(i+1)] = decision_tree
            temp['error_tree_'+str(i+1)] = error_tree

        print(f"已完成类别{C}的{i+1}号决策树和错误树的存放")
        print("="*40)
        frf[C] = temp

    print('模糊随机森林:', frf)

    # sys.exit()
    # frf = np.load('data/FuzzyRandomForest_.npy', allow_pickle='TRUE').item()
    # 对测试集进行分类预测
    testing_data = dataset.iloc[test_idx]

    y_pred = classify(testing_data, frf, np.unique(dataset[dataset.columns[-1]]), n_fold)
    y_true = testing_data[testing_data.columns[-1]]

    y_pred = np.array(y_pred).astype(str)
    y_true = np.array(y_true).astype(str)

    acc = (np.sum(y_true == y_pred) / y_true.shape[0]) * 100
    print("Fold: Accuracy: ", round(acc, 3))

    np.save(f'data/FuzzyRandomForest_{round(acc, 1)}%.npy', frf)
    return acc