import numpy as np
import DatasetProcessing as DP
from FuzzyRandomForest import FRF
from SVM import SVM
from AdaBoost import AdaBoost
from Predict import predicted
from sklearn.metrics import accuracy_score
from ConfusionMatrix import calculate_metrics
from ConfusionMatrix import plot_confusion_matrix
import time

if __name__ == '__main__':
    start = time.perf_counter()
    # tree_num: 每个类别生成树的数量, tree_max_deep:数的深度, q_rounds:增强树循环的次数
    tree_num = 3
    tree_max_deep = 15
    q_rounds = 10
    # 获取数据集
    dataset = DP.create_dataset()
    classification = np.unique(dataset[dataset.columns[-1]])
    # 8：2 划分数据集为训练集和测试集，这里划分的是index
    train_idx, test_idx = DP.dataset_split(dataset.shape[0], random_state=0)

    # 训练模糊随机森林分类器
    frf = FRF(dataset, train_idx, tree_num, tree_max_deep, test_idx)
    # 训练一对多SVM分类器
    svm, weigthSVM = SVM(dataset, train_idx)
    # 训练自适应增强树分类器
    ada = AdaBoost(dataset, train_idx, q_rounds)

    # 根据index获取测试集
    testing_data = dataset.iloc[test_idx]
    # 获取预测结果
    y_pred = predicted(frf, classification, tree_num, svm, weigthSVM, ada, testing_data)
    # 获取真实值
    y_true = testing_data[testing_data.columns[-1]]

    # 计算准确率
    acc = accuracy_score(y_pred, y_true)
    print(f"准确率: {acc*100}%")
    # 计算总体指标，包括精度,召回率和f1-分数
    print("总体指标: ")
    calculate_metrics(y_true, y_pred)
    # 绘图混淆矩阵
    plot_confusion_matrix(np.array(y_true), np.array(y_pred))

    end = time.perf_counter()
    print('程序运行时间为: %s Seconds' % (end-start))

