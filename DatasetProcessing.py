import random
import numpy as np
import pandas as pd

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

def dataset_split(n_samples, random_state=np.nan):
    """
    划分训练集和测试集
    """
    # 确定没有设置随机种子
    if np.isnan(random_state):
        np.random.seed(random_state)

    # 创建样本空间并随机打乱
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # 将样本 80% 划分给训练集
    n_train = round(n_samples * 0.8)
    train_indices = indices[:n_train]

    # 将20%的样本划分给测试集
    test_indices = indices[n_train:]

    return train_indices, test_indices
