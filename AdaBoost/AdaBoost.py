import matplotlib.pyplot as plt
from perp_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()

### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# n_estimators表示迭代的次数
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(features_train, labels_train)

# prettyPicture(clf, features_test, labels_test)

predict = clf.predict(features_test)
print("predict:\n", predict)

acc = accuracy_score(predict, labels_test)
print("acc:\n", acc)

# 返回决策函数值，大于0的就属于1
de_f = clf.decision_function(features_test)
print("de_f:\n", de_f)

