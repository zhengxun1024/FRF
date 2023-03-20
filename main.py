import numpy as np
import DatasetProcessing as DP
from FuzzyRandomForest import FRF
from SVM import SVM
from AdaBoost import AdaBoost
from Predict import predicted
from sklearn.metrics import accuracy_score
from ConfusionMatrix import calculate_metrics
from ConfusionMatrix import plot_confusion_matrix

if __name__ == '__main__':
    # 获取数据集
    dataset = DP.create_dataset()
    classification = np.unique(dataset[dataset.columns[-1]])
    # 8：2 划分数据集为训练集和测试集，这里划分的是index
    train_idx, test_idx = DP.dataset_split(dataset.shape[0], random_state=0)
    # 训练模糊随机森林分类器
    frf = FRF(dataset, train_idx, tree_num=5, tree_max_deep=6)
    # 训练一对多SVM分类器
    svm, weigthSVM = SVM(dataset, train_idx)
    # 训练自适应增强树分类器
    ada = AdaBoost(dataset, train_idx, q_rounds=4)

    # 根据index获取测试集
    testing_data = dataset.iloc[test_idx]

    y_pred = predicted(frf, classification, 5, svm, weigthSVM, ada, testing_data)
    y_true = testing_data[testing_data.columns[-1]]

    acc = accuracy_score(y_pred, y_true)
    print("Accuracy:\n", acc)

    # Calculate Overall Metrics
    print("\nOverall Metrics")
    # calculate precision, recall and f1-score
    calculate_metrics(y_true, y_pred)
    # plot confusion matrix
    plot_confusion_matrix(np.array(y_true), np.array(y_pred))
