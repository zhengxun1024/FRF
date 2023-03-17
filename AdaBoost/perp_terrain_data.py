import random

def makeTerrainData(n_points=1000):

    ### make the toy dataset
    random.seed(42)
    grade = [random.random() for ii in range(0 ,n_points)]
    bumpy = [random.random() for ii in range(0 ,n_points)]
    error = [random.random() for ii in range(0 ,n_points)]
    y = [round(grade[ii] * bumpy[ii] + 0.3 + 0.1 * error[ii]) for ii in range(0, n_points)]
    print(y)
    for ii in range(0, len(y)):
        if grade[ii] > 0.8 and bumpy[ii] > 0.8:
            y[ii] = 2
    ### split into train/test sets
    X = [[gg, ss] for gg, ss in zip(grade, bumpy)]
    split = int(0.8 * n_points)
    X_train = X[0:split]
    X_test  = X[split:]
    y_train = y[0:split]
    y_test  = y[split:]

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    features_train, labels_train, features_test, labels_test = makeTerrainData()
    print("features_train:\n", features_train)
    print("labels_train:\n", labels_train)
    print("labels_train:\n", len(labels_train))
    print("features_test:\n", features_test)
    print("labels_test:\n", labels_test)