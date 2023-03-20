import numpy as np
import math

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
    # 若当前实例不属于当前类别标签改为no
    for i in train_idx:
        if training_data.loc[i, label_class] != C:
            training_data.loc[i, label_class] = 'no'
    return training_data

# 判断该特征下的数据是否为连续值
def judge_continuous(data_feature):
    # 若数据是连续值就返回1，是离散值就返回0
    if len(np.unique(data_feature)) < len(data_feature) / 3:
        return 0
    else:
        return 1

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

# 计算当前集合的熵
def calcEntropy(dataset, name_feature, c, a):
    # 获取当前特征的数据
    data_features = np.array(dataset[name_feature])
    # 获取当前特征数据的结果标签
    label_list = np.array(dataset[dataset.columns[-1]])
    # 获取当前特征每条数据的隶属度
    u_list = membership_value_list(data_features, c, a)
    # N1 = u_list.sum()
    # N2 = len(data_features) - N1

    # 这里是一一对应的，label中下标为0的类别对应C1_list、C2_list中下标0的数据
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
    # 计算该组合下的熵
    Entropy = ((E1 + E2) / len(data_features))
    return Entropy

# 选择最佳特征
def choose_best_feature(dataset):
    # 特征总数
    numFeatures = len(dataset.columns[:-1])
    # 当只有一个特征时，说明只有class列
    if numFeatures == 0:
        return 0
    # 初始化最佳熵
    bestEntropy = 0
    # 初始化最优特征、最优分割点c和模糊区间a
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
            # 循环计算不同组合下的熵
            while b > 0.01 and a < 0.4*c:
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
        # 计算当前节点的隶属度
        u = membership_value(test_example[first_feature], parameter['c'], parameter['a'], 1)
        # 防止意外
        if parameter['a'] == 0:
            keys.append('left')
            weights['left'] = weight
        # 走左子树
        if test_example[first_feature] < parameter['c'] + parameter['a']:
            keys.append('left')
            weights['left'] = weight * u
        # 走右子树
        if test_example[first_feature] > parameter['c']:
            keys.append('right')
            weights['right'] = weight * (1 - u)
    else:
        # 计算当前节点的隶属度
        u = membership_value(test_example[first_feature], parameter['c'], parameter['a'], 0)
        # 防止意外
        if parameter['a'] == 0 and test_example[first_feature] == parameter['c']:
            keys.append('left')
            weights['left'] = weight
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
        for j in range(len(predict_result_list)):
            for k in range(len(type)):
                if predict_result_list[j] == type[k]:
                    temp[k] += weight_list[j]
        # 获取分类结果，即决策值最大的类别
        predict_result = type[np.argmax(np.array(temp))]
        # 获取训练集真实值
        true_result = training_data.loc[train_idx[i], training_data.columns[-1]]
        # 比较分类结果和真实值，若相同则分类正确标签改为1，否则分类错误标签改为0
        if predict_result == true_result:
            training_data.loc[train_idx[i], training_data.columns[-1]] = 1
        else:
            training_data.loc[train_idx[i], training_data.columns[-1]] = 0
    return training_data

# 生成模糊随机森林分类器模型
def FRF(dataset, dataset_idx, tree_num, tree_max_deep):
    # 对训练数据根据类别数量进行分组
    train_idx = fold_split(dataset_idx, n_fold=tree_num)
    # 训练模糊随机森林生成决策树
    frf = {}
    for C in np.unique(dataset[dataset.columns[-1]]):
        # 依次为每个类别训练tree_num课决策树
        temp = {}
        for i in range(tree_num):
            # 选取训练集
            print(f"正在生成类别{C}的{i+1}号决策树训练集")
            training_data = training_data_process(dataset, train_idx[i], C)
            # 生成决策树
            print(f"正在生成类别{C}的{i+1}号模糊决策树")
            decision_tree = create_decision_tree(training_data, 'decision', deep=tree_max_deep)
            # 获取错误树训练数据
            print(f"正在生成类别{C}的{i+1}号错误树训练集")
            error_train_data = process_error_train_data(train_idx[i], training_data.copy(), decision_tree)
            # 生成错误树
            print(f"正在生成类别{C}的{i+1}号错误树")
            error_tree = create_decision_tree(error_train_data, 'error', deep=tree_max_deep)
            # 存储决策树和错误树
            temp['decision_tree_'+str(i+1)] = decision_tree
            temp['error_tree_'+str(i+1)] = error_tree

        print(f"已完成类别{C}的决策树和错误树的生成")
        print("="*40)
        # frf = {类别1：{决策树1：{}, 错误树1：{}, 决策树2：{}, 错误树2：{},...}, 类别2：{...}, ...}
        frf[C] = temp
    print(f"已完成模糊随机森林的生成")
    print("="*40)
    return frf