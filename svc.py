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

# 定义要调优的参数范围
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear']}

# 创建 SVM 分类器
svm_clf = svm.SVC()

# 使用网格搜索进行参数调优
grid_search = GridSearchCV(svm_clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 打印最佳参数组合和对应的准确率
print("Best Parameters: ", grid_search.best_params_)
print("Best Accuracy: ", grid_search.best_score_)

# 使用最佳参数组合创建 SVM 分类器
best_svm_clf = svm.SVC(**grid_search.best_params_)

# 使用交叉验证计算模型在训练集上的性能
cross_val_scores = cross_val_score(best_svm_clf, X_train, y_train, cv=5)
print("Cross-validation Scores: ", cross_val_scores)
print("Mean CV Accuracy: ", cross_val_scores.mean())

# 在测试集上进行预测
best_svm_clf.fit(X_train, y_train)
y_pred = best_svm_clf.predict(X_test)

# 计算准确率、精确率、召回率、F1 分数
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)

# 绘制 ROC 曲线与计算 AUC
y_score = best_svm_clf.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
print("AUC: ", roc_auc)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")

# 计算并绘制混淆矩阵
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(conf_mat)
plt.show()