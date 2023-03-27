import numpy as np

from FuzzyRandomForest import predict
from sklearn.metrics import accuracy_score

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
        # 存储该实例在模糊随机森林中的分类结果
        predicted.append(predict_dict)
    return predicted

# SVM分类
def classifySVM(SVM ,weigthSVM, testing_data, classification):
    # 用所有类别的SVM预测一个实例，获取决策值
    predicted = []
    for i in range(testing_data.shape[0]):
        # 依次取一个实例
        features_test = testing_data.iloc[i:i+1, :-1]
        predict_dict = {}
        # 依次获取该实例每个类别的决策值
        for C in classification:
            # 返回决策函数值
            de_f = SVM[C].decision_function(features_test)
            predict_dict[C] = de_f[0] * weigthSVM
        # 存储每个实例中每个类别SVM的决策值
        predicted.append(predict_dict)
    return predicted

# 自适应增强树分类
def classifyAda(AdaBoost, testing_data, classification):
    # 用所有类别的SVM预测一个实例，获取决策值
    predicted = []
    for i in range(testing_data.shape[0]):
        # 依次取一个实例
        features_test = testing_data.iloc[i:i+1, :-1]
        predict_dict = {}
        # 依次获取该实例每个类别的决策值
        for C in classification:
            # 返回决策函数值
            de_f = AdaBoost[C].decision_function(features_test)
            predict_dict[C] = de_f[0]
        # 存储每个实例中每个类别增强树的决策值
        predicted.append(predict_dict)
    return predicted

# 使用三大分类器预测，最后加权集成分类结果
def predicted(FRF, classification, tree_num, SVM, weigthSVM, AdaBoost, testing_data):
    # 获取模糊随机森林分类结果
    predictFRF = classifyFRF(testing_data, FRF, classification, tree_num)
    # 获取SVM分类结果
    predictSVM = classifySVM(SVM, weigthSVM, testing_data, classification)
    # 获取增强树分类结果
    predictAda = classifyAda(AdaBoost, testing_data, classification)

    # print('模糊随机森林分类结果如下:\n', predictFRF)
    # print('SVM支持向量机分类结果如下:\n', predictSVM)
    # print('自适应增强树分类结果如下:\n', predictAda)
    result_frf = ['error'] * testing_data.shape[0]
    result_svm = ['error'] * testing_data.shape[0]
    result_ada = ['error'] * testing_data.shape[0]
    result_frf_svm = ['error'] * testing_data.shape[0]
    result_frf_ada = ['error'] * testing_data.shape[0]
    result_svm_ada = ['error'] * testing_data.shape[0]
    result = ['error'] * testing_data.shape[0]
    # 获取真实值
    y_true = np.array(testing_data[testing_data.columns[-1]])
    # 依次得出测试实例的最终分类结果
    for i in range(testing_data.shape[0]):
        result_frf[i] = max(predictFRF[i], key=predictFRF[i].get)
        result_svm[i] = max(predictSVM[i], key=predictSVM[i].get)
        result_ada[i] = max(predictAda[i], key=predictAda[i].get)
        frf_svm = {}
        frf_ada = {}
        svm_ada = {}
        predict_dict = {}
        # 依次计算该实例每个类别的最后结果
        for C in classification:
            frf_svm[C] = predictFRF[i][C] + predictSVM[i][C]
            frf_ada[C] = predictFRF[i][C] + predictAda[i][C]
            svm_ada[C] = predictSVM[i][C] + predictAda[i][C]
            predict_dict[C] = predictFRF[i][C] + predictSVM[i][C] + predictAda[i][C]
        # print(f'第{i+1}个实例最终的决策值:', predict_dict)
        # 选取值最有可能的类别，即值最大的键名
        result_frf_svm[i] = max(frf_svm, key=frf_svm.get)
        result_frf_ada[i] = max(frf_ada, key=frf_ada.get)
        result_svm_ada[i] = max(svm_ada, key=svm_ada.get)
        result[i] = max(predict_dict, key=predict_dict.get)
        print(f'第{i+1}个实例真实值:', y_true[i])
        print(f'模糊随机森林:{predictFRF[i]}, \t结果：{result_frf[i]}')
        print(f'SVM支持向量机:{predictSVM[i]}, \t结果：{result_svm[i]}')
        print(f'自适应增强树:{predictAda[i]}, \t结果：{result_ada[i]}')
        print(f'frf_svm:{frf_svm}, \t结果：{result_frf_svm[i]}')
        print(f'frf_ada:{frf_ada}, \t结果：{result_frf_ada[i]}')
        print(f'svm_ada:{svm_ada}, \t结果：{result_svm_ada[i]}')
        print(f'最后结果:{predict_dict}, \t结果：{result[i]}')

    # 计算准确率
    acc_frf = accuracy_score(result_frf, y_true)
    print(f"frf准确率: {acc_frf*100}%")
    acc_svm = accuracy_score(result_svm, y_true)
    print(f"svm准确率: {acc_svm*100}%")
    acc_ada = accuracy_score(result_ada, y_true)
    print(f"ada准确率: {acc_ada*100}%")
    acc_frf_svm = accuracy_score(result_frf_svm, y_true)
    print(f"frf_svm准确率: {acc_frf_svm*100}%")
    acc_frf_ada = accuracy_score(result_frf_ada, y_true)
    print(f"frf_ada准确率: {acc_frf_ada*100}%")
    acc_svm_ada = accuracy_score(result_svm_ada, y_true)
    print(f"svm_ada准确率: {acc_svm_ada*100}%")

    # print('加权投票之后的分类结果如下:\n', result)
    return result
