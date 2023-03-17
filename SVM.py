import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
X = np.array([[10, 10],[8, 10],[-5, 5.5],[-5.4, 5.5],[-20, -20],[-15, -20]])
y = np.array([0, 0, 1, 1, 2, 2])
clf = OneVsRestClassifier(SVC(kernel='rbf')).fit(X, y)

predict = clf.predict(X)
acc = accuracy_score(predict, y)
print("acc:", acc)

de_f = clf.decision_function([[-19, -20], [9, 9], [-5, 5]])
print("decision_function:", de_f)
print("type(decision_function):", type(de_f))