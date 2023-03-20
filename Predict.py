from FuzzyRandomForest import predict

# 模糊随机森林分类
def classifyFRF(testing_data, frf, classification, tree_num):
    # Create new query instances by simply removing the target feature column from the original dataset and
    # convert it to a dictionary
    queries = testing_data.iloc[:, :-1].to_dict(orient="records")

    # Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = []

    # Calculate the prediction accuracy
    for i in range(testing_data.shape[0]):
        predict_dict = {}
        for C in classification:
            predict_dict[C] = 0
            for j in range(tree_num):
                leaf_decision_tree, weight_decision_tree = predict(frf[C]['decision_tree_'+str(j+1)], queries[i], [], 1, [])
                result_decision_tree = 0
                for k in range(len(leaf_decision_tree)):
                    if leaf_decision_tree[k] == C:
                        result_decision_tree += weight_decision_tree[k]

                leaf_error_tree, weight_error_tree = predict(frf[C]['error_tree_'+str(j+1)], queries[i], [], 1, [])
                result_error_tree = 0
                for k in range(len(leaf_error_tree)):
                    result_error_tree += weight_error_tree[k] * leaf_error_tree[k]

                predict_dict[C] += result_decision_tree * result_error_tree

        predicted.append(predict_dict)
        # print(f'predict_dict:', predict_dict)
    return predicted

# SVM分类
def classifySVM(SVM ,weigthSVM, testing_data, classification):
    predicted = []
    for i in range(testing_data.shape[0]):
        features_test = testing_data.iloc[i:i+1, :-1]
        predict_dict = {}
        for C in classification:
            # 返回决策函数值，大于0的就属于1
            de_f = SVM[C].decision_function(features_test)
            predict_dict[C] = de_f[0] * weigthSVM
        predicted.append(predict_dict)
    return predicted

# 自适应增强树分类
def classifyAda(AdaBoost, testing_data, classification):
    predicted = []
    for i in range(testing_data.shape[0]):
        features_test = testing_data.iloc[i:i+1, :-1]
        predict_dict = {}
        for C in classification:
            # 返回决策函数值，大于0的就属于1
            de_f = AdaBoost[C].decision_function(features_test)
            predict_dict[C] = de_f[0]
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

    print('模糊随机森林分类结果如下:\n', predictFRF)
    print('SVM支持向量机分类结果如下:\n:', predictSVM)
    print('自适应增强树分类结果如下:\n:', predictAda)
    result = ['error'] * testing_data.shape[0]
    for i in range(testing_data.shape[0]):
        predict_dict = {}
        for C in classification:
            predict_dict[C] = predictFRF[i][C] + predictSVM[i][C] + predictAda[i][C]

        print('predict_dict:', predict_dict)
        result[i] = max(predict_dict, key=predict_dict.get)
    print('加权投票之后的分类结果如下\n:', result)
    return result
