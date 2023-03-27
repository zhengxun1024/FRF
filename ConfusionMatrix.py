import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred):
    # unique classes
    conf_mat = {}
    classes = np.unique(y_true)
    # C is positive class while True class is y_true or temp_true
    for c in classes:
        temp_true = y_true[y_true == c]
        temp_pred = y_pred[y_true == c]
        conf_mat[c] = {pred: np.sum(temp_pred == pred) for pred in classes}
    print("混淆矩阵: \n", pd.DataFrame(conf_mat))

    # plot confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(data=pd.DataFrame(conf_mat), annot=True, cmap=plt.get_cmap("Blues"), fmt='d')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def calculate_metrics(y_true, y_pred):
    # convert to integer numpy array
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    pre_list = []
    rec_list = []
    f1_list = []
    # loop over unique classes
    for c in np.unique(y_true):
        # copy arrays
        temp_true = y_true.copy()
        temp_pred = y_pred.copy()

        # positive class
        temp_true[y_true == c] = 1
        temp_pred[y_pred == c] = 1

        # negative class
        temp_true[y_true != c] = 0
        temp_pred[y_pred != c] = 0

        # tp, fp and fn
        tp = np.sum(temp_pred[temp_pred == 1] == temp_true[temp_pred == 1])
        tn = np.sum(temp_pred[temp_pred == 0] == temp_true[temp_pred == 0])
        fp = np.sum(temp_pred[temp_pred == 1] != temp_true[temp_pred == 1])
        fn = np.sum(temp_pred[temp_pred == 0] != temp_true[temp_pred == 0])

        precision = tp / (tp + fp) * 100
        recall = tp / (tp + fn) * 100
        f1 = 2 * (precision * recall) / (precision + recall)

        pre_list.append(precision)
        rec_list.append(recall)
        f1_list.append(f1)
        print(
            "Class {}: 精确度 = {:0.3f}    召回率 = {:0.3f}    F1-分数 = {:0.3f}".format(c, precision, recall, f1))

    print("Average: 精确度 = {:0.3f}    召回率 = {:0.3f}    F1-分数 = {:0.3f}   准确率 = {:0.3f}".
          format(np.mean(pre_list),
                 np.mean(rec_list),
                 np.mean(f1_list),
                 np.sum(y_pred == y_true) / y_pred.shape[0] * 100))
