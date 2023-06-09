import sys
import random
import numpy as np
from math import log2

# 给训练集分组
def random_select(dataset, data_split, classification, C, n_samples):
    # 确定其他类别所占的训练集比例
    n = (len(classification)-1)
    t = 0.5
    # 循环随机抽取各类别中的数据
    train_indices = []
    if len(data_split[C]) > 0.5*n_samples:
        random_value = random.sample(data_split[C], round(0.5 * n_samples))
        train_indices = train_indices + random_value
    else:
        train_indices = train_indices + data_split[C]
        t = (1 - len(data_split[C])/n_samples)
    for type in classification:
        if type != C:
            if len(data_split[C]) > (t / n) * n_samples:
                random_value = random.sample(data_split[type], round((t / n) * n_samples))
                train_indices = train_indices + random_value
            else:
                train_indices = train_indices + data_split[type]
                n -= 1
                t = (t - len(data_split[type]) / n_samples)
    # 打乱抽取的索引
    random.shuffle(train_indices)
    # 根据索引获取训练集数据
    training_data = dataset.iloc[train_indices]
    # 得到训练集标签名称
    label_class = training_data.columns[-1]
    # 若当前实例不属于当前类别标签改为-1
    training_data.loc[training_data[label_class] != C, label_class] = -1
    return train_indices, training_data

# 判断该特征下的数据是否为连续值
def judge_continuous(data_feature):
    # 若数据是连续值就返回1，是离散值就返回0
    if len(np.unique(data_feature)) < 15:
        return 0
    else:
        return 1

# 计算x的隶属度
def membership_value(x, c, a, continuous):
    ans = 0
    # continuous=0，则是离散值；continuous=1，则是连续值
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

# 计算特征数据的隶属度，返回隶属度列表
def membership_value_list(data_features, c, a):
    ans = []
    # 连续值和离散值的隶属度计算公式是不一样的
    if judge_continuous(data_features) == 0:
        # 离散值
        for i in range(len(data_features)):
            x = data_features[i]
            ans.append(membership_value(x, c, a, 0))
    else:
        # 连续值
        for i in range(len(data_features)):
            x = data_features[i]
            ans.append(membership_value(x, c, a, 1))
    return np.array(ans)

# 计算当前集合的熵
def calcEntropy(dataset, name_feature, c, a):
    # 获取当前特征的数据
    data_features = np.array(dataset[name_feature])
    # 获取当前特征数据的结果标签
    label_list = np.array(dataset[dataset.columns[-1]])
    # 获取当前特征每条数据的隶属度
    u_list = membership_value_list(data_features, c, a)

    # 简单版本:
    # Entropy = ((E1 + E2) / len(data_features))  准确率为 65.5%
    # N1 = u_list.sum()
    # N2 = len(data_features) - N1
    # 这里是一一对应的，label中下标为0的类别对应C1_list、C2_list中下标为0的数据
    # C1_list = [0, 0]
    # C2_list = [0, 0]
    # label = np.unique(label_list)
    # for i in range(len(label_list)):
    #     if label_list[i] == label[0]:
    #         C1_list[0] += u_list[i]
    #         C2_list[0] += (1 - u_list[i])
    #     else:
    #         C1_list[1] += u_list[i]
    #         C2_list[1] += (1 - u_list[i])
    # C1 = max(C1_list)
    # C2 = max(C2_list)
    # E1 = min(C1_list)
    # E2 = min(C2_list)
    # # 计算该组合下的熵
    # Entropy = ((E1 + E2) / len(data_features))

    # 使用基尼指数的方式计算熵，注意这里是越小越好,准确率为 65.25%
    # 使用改版的信息熵准确率为73%
    D = len(data_features)
    D1, D2 = 0, 0
    C11, C12, C21, C22 = 0, 0, 0, 0
    label = np.unique(label_list)
    for i in range(len(label_list)):
        if u_list[i] > 0:
            D1 += 1
            if label_list[i] == label[0]:
                C11 += u_list[i]
            else:
                C12 += u_list[i]
        if u_list[i] < 1:
            D2 += 1
            if label_list[i] == label[0]:
                C21 += 1 - u_list[i]
            else:
                C22 += 1 - u_list[i]
    Gini1, Gini2 = 0, 0
    # print(f"D:{D}, D1:{D1}, D2:{D2}, C11:{C11}, C12:{C12}, C21:{C21}, C22:{C22}")
    if D1 != 0:
        Gini1 = (D1 / D) * 2 * (C11 / D1) * (C12 / D1)
        # a = (C11/D1)*log2(C11/D1) if C11 != 0 else 0
        # b = (C12/D1)*log2(C12/D1) if C12 != 0 else 0
        # Gini1 = -(D1/D) * (a + b)
    if D2 != 0:
        Gini2 = (D2 / D) * 2 * (C21 / D2) * (C22 / D2)
        # a = (C21/D2)*log2(C21/D2) if C21 != 0 else 0
        # b = (C22/D2)*log2(C22/D2) if C22 != 0 else 0
        # Gini2 = -(D2/D) * (a + b)
    Entropy = Gini1 + Gini2
    # print(f"Entropy:{Entropy}")
    return Entropy

# 选择最佳特征
def choose_best_feature(dataset):
    # 特征总数
    numFeatures = len(dataset.columns[:-1])
    # 当只有一个特征时，说明只有class列
    if numFeatures == 0:
        return 0
    # 初始化最佳熵
    bestEntropy = 1
    # 随机选择最优特征,初始化最优分割点c和模糊区间a
    index_of_best_feature = random.randint(0, numFeatures-1)
    best_c = -1
    best_a = -1
    # 寻找最优特征下的最优切分点
     # 记录当前特征名称
    name_feature = dataset.columns[index_of_best_feature]
    # 去重，每个属性值唯一
    uniqueVals = np.unique(np.array(dataset[name_feature]))
    mean = (np.max(np.array(dataset[name_feature])) - np.min(np.array(dataset[name_feature])))
    if len(uniqueVals) <= 2:
        best_c = uniqueVals[0]
        best_a = 0
    else:
        # 对于当前特征的每个取值
        for c in uniqueVals:
            # 确定参数a
            a = 0
            # 循环计算不同组合下的熵
            while a <= 0.3 * mean:
                # 计算的熵
                Entropy = calcEntropy(dataset, name_feature, c, a)
                # 更新最优特征和最优切分点
                if Entropy < bestEntropy:
                    bestEntropy = Entropy
                    best_c = c
                    best_a = a
                a = a + mean * 0.05
    # print("bestEntropy:", bestEntropy)
    # print("index_of_best_feature:", index_of_best_feature)
    # print("best_c:", best_c)
    # print("best_a:", best_a)
    return index_of_best_feature, best_c, best_a

def create_decision_tree(dataset, tree_type, deep=9):
    # 求出训练集所有样本的标签
    label_list = np.array(dataset[dataset.columns[-1]])
    # 有四个递归结束的情况：
    if dataset.empty:
        return -1 if tree_type == 'decision' else 0
    # 若当前集合的所有样本标签相等（即样本已被分“纯”）
    # 则直接返回该标签值作为一个叶子节点
    if len(np.unique(label_list)) <= 1:
        return label_list[0] if tree_type == 'decision' else 1
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
        if best_a == 0:
            sub_dataset1 = dataset[dataset[best_feature] <= best_c].drop(best_feature, axis=1)
        else:
            sub_dataset1 = dataset[dataset[best_feature] < (best_c + best_a)].drop(best_feature, axis=1)
        sub_dataset2 = dataset[dataset[best_feature] > best_c].drop(best_feature, axis=1)
    else:
        # 生成由最优切分点划分出来的二分子集
        if best_a == 0:
            sub_dataset1 = dataset[(dataset[best_feature] == best_c)].drop(best_feature, axis=1)
        else:
            sub_dataset1 = dataset[(dataset[best_feature] > (best_c - best_a)) & (dataset[best_feature] < (best_c + best_a))].drop(best_feature, axis=1)
        sub_dataset2 = dataset[dataset[best_feature] != best_c].drop(best_feature, axis=1)
    # 构造左子树
    decision_tree[best_feature]['left'] = create_decision_tree(sub_dataset1, tree_type, deep-1)
    # 构造右子树
    decision_tree[best_feature]['right'] = create_decision_tree(sub_dataset2, tree_type, deep-1)
    # 存储参数
    decision_tree[best_feature]['parameter'] = {'continuous': continuous, 'c': best_c, 'a': best_a}

    return decision_tree

# 用上面训练好的决策树对新样本分类
def predict(decision_tree, test_example, leaf_list, weight, weight_list):
    # 若决策树直接就是叶子结果，则直接返回结果
    if not isinstance(decision_tree, dict):
        # 则就是要找的标签值
        leaf_list.append(decision_tree)
        weight_list.append(weight)
        return leaf_list, weight_list
    # 根节点代表的属性
    first_feature = list(decision_tree.keys())[0]
    # second_dict是第一个分类属性的值（也是字典）
    second_dict = decision_tree[first_feature]
    # 获取参数值
    parameter = second_dict['parameter']
    # keys存储不同的路径，weights存储不同路径下的隶属度
    keys = []
    weights = {}
    # 若continuous=1则属于连续值，否则属于离散值
    if parameter['continuous'] == 1:
        if parameter['a'] == 0:
            # 走左子树
            if test_example[first_feature] <= parameter['c']:
                keys.append('left')
                weights['left'] = weight
            # 走右子树
            if test_example[first_feature] > parameter['c']:
                keys.append('right')
                weights['right'] = weight
        else:
            # 计算当前节点的隶属度
            u = membership_value(test_example[first_feature], parameter['c'], parameter['a'], 1)
            # 走左子树
            if test_example[first_feature] < parameter['c'] + parameter['a']:
                keys.append('left')
                weights['left'] = weight * u
            # 走右子树
            if test_example[first_feature] > parameter['c']:
                keys.append('right')
                weights['right'] = weight * (1 - u)
    else:
        if parameter['a'] == 0:
            # 走左子树
            if test_example[first_feature] == parameter['c']:
                keys.append('left')
                weights['left'] = weight
            # 走右子树
            if test_example[first_feature] != parameter['c']:
                keys.append('right')
                weights['right'] = weight
        else:
            # 计算当前节点的隶属度
            u = membership_value(test_example[first_feature], parameter['c'], parameter['a'], 0)
            # 走左子树
            if parameter['c'] - parameter['a'] < test_example[first_feature] < parameter['c'] + parameter['a']:
                keys.append('left')
                weights['left'] = weight * u
            # 走右子树
            if test_example[first_feature] != parameter['c']:
                keys.append('right')
                weights['right'] = weight * (1 - u)
    # 遍历存储的路径
    for key in keys:
        # 若当前second_dict的key的value是一个字典
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
    # 将训练数据转化为字典，注意去除标签列
    queries = training_data.iloc[:, :-1].to_dict(orient="records")
    # 遍历训练数据集
    for i in range(training_data.shape[0]):
        # 获取决策树分类结果
        predict_result_list, weight_list = predict(decision_tree, queries[i], [], 1, [])
        # 定义type为类别列表，temp为类别权重
        type = np.unique(np.array(predict_result_list))
        temp = [0]*len(type)
        # 计算每一类别的决策值
        for j in range(len(type)):
            for k in range(len(predict_result_list)):
                if predict_result_list[k] == type[j]:
                    temp[j] += weight_list[k]
        # 获取分类结果，即决策值最大的类别
        predict_result = type[np.argmax(np.array(temp))]
        # 获取训练集真实值
        true_result = training_data.loc[train_idx[i], training_data.columns[-1]]
        # 比较分类结果和真实值，若相同则分类正确标签改为1，否则分类错误标签改为0
        if predict_result == true_result:
            training_data.loc[train_idx[i], training_data.columns[-1]] = 1
        else:
            training_data.loc[train_idx[i], training_data.columns[-1]] = 0
            # print(type, temp, predict_result, true_result)
    return training_data

def classify(testing_data, tree, C, tree_num):
    # 将测试数据转换成字典，注意去除类别那一列
    queries = testing_data.iloc[:, :-1].to_dict(orient="records")
    # 定义一个空的列表，存储预测结果
    predicted = []
    # 依次为每个实例计算决策值
    for i in range(testing_data.shape[0]):
        predict_C = 0
        predict_not_C = 0
        # 计算当前类别的每棵树的决策值
        for j in range(tree_num):
            # 获取当前决策树的分类结果
            leaf_decision_tree, weight_decision_tree = predict(tree['decision_tree_'+str(j+1)], queries[i], [], 1, [])
            # 计算当前决策树的分类结果，即决策树的决策值
            result_decision_tree = 0
            for k in range(len(leaf_decision_tree)):
                if leaf_decision_tree[k] == C:
                    result_decision_tree += weight_decision_tree[k]
            result_decision_tree = result_decision_tree / sum(weight_decision_tree)
            # 获取错误树的分类结果
            leaf_error_tree, weight_error_tree = predict(tree['error_tree_'+str(j+1)], queries[i], [], 1, [])
            # 计算当前错误树的分类结果，即决策树的权重
            result_error_tree = 0
            for k in range(len(leaf_error_tree)):
                result_error_tree += weight_error_tree[k] * leaf_error_tree[k]
            # 计算该决策树最终的决策值
            predict_C += result_decision_tree * result_error_tree
            predict_not_C += (1 - result_decision_tree) * result_error_tree
        # 存储该实例在模糊随机森林中的分类结果
        s = C if predict_C >= predict_not_C else -1
        predicted.append(s)
        # print(f'实例{i+1}的决策值:', predict_C, predict_not_C, s)
    return predicted


# 生成模糊随机森林分类器模型
def FRF(dataset, dataset_idx, tree_num, tree_max_deep, test_idx):
    # 对训练数据根据类别进行分组
    classification = np.unique(dataset[dataset.columns[-1]])
    train_data = dataset.iloc[dataset_idx]
    testing1_data = dataset.iloc[test_idx]
    data_split = {}
    for C in classification:
        data_split[C] = list(train_data[train_data[dataset.columns[-1]] == C].index)

    # 训练模糊随机森林生成决策树
    frf_dict = {}
    for C in classification:
        # 若当前实例不属于当前类别标签改为-1
        testing1_data.loc[testing1_data['price_range'] != C, 'price_range'] = -1
        # 依次为每个类别训练tree_num课决策树
        temp = {}
        for i in range(tree_num):
            # 随机选取训练集,使当前类别占50%比例
            print(f"正在生成类别{C}的{i+1}号决策树训练集")
            if tree_num == 1:
                training_idx = dataset_idx
                training_data = dataset.iloc[training_idx]
                label_class = training_data.columns[-1]
                training_data.loc[training_data[label_class] != C, label_class] = -1
            else:
                training_idx, training_data = random_select(dataset, data_split, classification, C, len(dataset_idx)//tree_num)
            # 生成决策树
            print(f"正在生成类别{C}的{i+1}号模糊决策树")
            decision_tree = create_decision_tree(training_data, 'decision', deep=tree_max_deep)
            # print("决策树:", decision_tree)
            # 获取错误树训练数据
            print(f"正在生成类别{C}的{i+1}号错误树训练集")
            error_train_data = process_error_train_data(training_idx, training_data.copy(), decision_tree)
            print('训练集准确率为:', error_train_data[error_train_data['price_range'] == 1].shape[0] / error_train_data.shape[0])
            result_data = process_error_train_data(test_idx, testing1_data.copy(), decision_tree)
            print('测试集准确率为:', result_data[result_data['price_range'] == 1].shape[0] / result_data.shape[0])

            # 生成错误树
            print(f"正在生成类别{C}的{i+1}号错误树")
            error_tree = create_decision_tree(error_train_data, 'error', deep=tree_max_deep)
            # print("错误树:", error_tree)
            # 存储决策树和错误树
            temp['decision_tree_'+str(i+1)] = decision_tree
            temp['error_tree_'+str(i+1)] = error_tree
        print(f"已完成类别{C}的决策树和错误树的生成")

        t = classify(testing1_data, temp, C, tree_num)
        acc1 = accuracy_score(t, testing1_data['price_range'])
        print(f"类别{C}森林的准确率:{acc1}")

        print("="*40)
        # frf_dict = {类别1：{决策树1：{}, 错误树1：{}, 决策树2：{}, 错误树2：{},...}, 类别2：{...}, ...}
        frf_dict[C] = temp
    print(f"已完成模糊随机森林的生成")
    print("="*40)
    return frf_dict


import DatasetProcessing as DP
from sklearn.metrics import accuracy_score
import time

# 模糊随机森林分类
def classifyFRF(testing_data, frf, classification, tree_num):
    # 将测试数据转换成字典，注意去除类别那一列
    queries = testing_data.iloc[:, :-1].to_dict(orient="records")
    # 定义一个空的列表，存储预测结果
    predicted = []
    # 依次为每个实例计算决策值
    for i in range(testing_data.shape[0]):
        predict_dict = {}
        # 为该实例依次计算每个类别的决策值
        for C in classification:
            predict_dict[C] = 0
            # 计算当前类别的每棵树的决策值
            for j in range(tree_num):
                # 获取当前决策树的分类结果
                leaf_decision_tree, weight_decision_tree = predict(frf[C]['decision_tree_'+str(j+1)], queries[i], [], 1, [])
                # 计算当前决策树的分类结果，即决策树的决策值
                result_decision_tree = 0
                for k in range(len(leaf_decision_tree)):
                    if leaf_decision_tree[k] == C:
                        result_decision_tree += weight_decision_tree[k]
                result_decision_tree = result_decision_tree / sum(weight_decision_tree)
                # 获取错误树的分类结果
                leaf_error_tree, weight_error_tree = predict(frf[C]['error_tree_'+str(j+1)], queries[i], [], 1, [])
                # 计算当前错误树的分类结果，即决策树的权重
                result_error_tree = 0
                for k in range(len(leaf_error_tree)):
                    result_error_tree += weight_error_tree[k] * leaf_error_tree[k]
                # 计算该决策树最终的决策值
                predict_dict[C] += result_decision_tree * result_error_tree
        # print(f'实例{i+1}的决策值:\n', predict_dict)
        # 存储该实例在模糊随机森林中的分类结果
        predicted.append(max(predict_dict, key=predict_dict.get))
    return predicted

if __name__ == '__main__':
    start = time.perf_counter()
    tree_num_test = 5
    tree_deep_test = 15
    # 获取数据集
    dataset = DP.create_dataset()
    classification = np.unique(dataset[dataset.columns[-1]])
    # 8：2 划分数据集为训练集和测试集，这里划分的是index
    train_idx, test_idx = DP.dataset_split(dataset.shape[0], random_state=0)
    # 训练模糊随机森林分类器
    frf = FRF(dataset, train_idx, tree_num_test, tree_deep_test, test_idx)
    # 根据index获取测试集
    testing_data = dataset.iloc[test_idx]
    # 获取预测结果
    y_pred = classifyFRF(testing_data, frf, classification, tree_num_test)
    print('分类结果如下:\n', y_pred)
    # 获取真实值
    y_true = testing_data[testing_data.columns[-1]]
    print('真实值如下:\n', list(y_true))

    # 计算准确率
    acc = accuracy_score(y_pred, y_true)
    print(f"tree_num:{tree_num_test}, tree_max_deep:{tree_deep_test}, 准确率: {acc*100}%")
    end = time.perf_counter()
    print('程序运行时间为: %s Seconds' % (end-start))