import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn import preprocessing
import csv
import numpy as np
from numpy import nan
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, log_loss, \
    classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
csvFile = open("数据集.csv", "r")
csv_data = csv.reader(csvFile)
cancer = np.array([i for i in csv_data])
names = ['小臂极值', '小腿极值', '小腿均值', '小腿加速度极值']
da=cancer[0:, :4]
data=[]
for i in da:
    temp = []
    for j in i:
        if j == '?':
            temp.append(nan)
        else:
            temp.append(float(j))
    data.append(temp)
target = []
for i in cancer[0:, 4]: #除去第一行的第九列
    if i == '0':
        target.append(0)
    if i == '1':
        target.append(1)
target_names = ['class1', 'class2']
# print(target)
#
from sklearn.datasets._base import Bunch
from sklearn.neighbors import KNeighborsClassifier
real_data = Bunch(data=data, target=target, feature_names=names, target_names=target_names)
# print(real_data)


X = real_data.data
y = real_data.target
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 构建KNN分类器
knn = KNeighborsClassifier()

# 设置超参数网格
param_grid = {
    'n_neighbors': range(1, 100),
    'weights': ['uniform', 'distance'],
    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
}

# 使用网格搜索和交叉验证进行超参数调优
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 输出最佳超参数组合
print("Best parameters: ", grid_search.best_params_)

# 使用最佳超参数组合对测试集进行预测
best_knn = grid_search.best_estimator_
best_knn.fit(X_train, y_train)
y_pred = best_knn.predict(X_test)

# 计算性能指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 计算ROC曲线和AUC
y_score = best_knn.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")


# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# 计算准确率、精确率、召回率、F1 分数
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)
# 输出性能指标
print("Accuracy: ".format(accuracy))
print("Precision: ".format(precision))
print("Recall: ".format(recall))
print("F1 Score: ".format(f1))
print("AUC: ".format(roc_auc))
plt.show()