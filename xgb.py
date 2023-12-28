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
csvFile = open("数据集新.csv", "r")
csv_data = csv.reader(csvFile)
cancer = np.array([i for i in csv_data])
names = ['小臂极值', '小腿极值', '小腿加速度极值']
da=cancer[0:, :3]
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
for i in cancer[0:, 3]: #除去第一行的第九列
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
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.2, 0.3],
    'n_estimators': [50, 100, 150],
    'min_child_weight': [1, 3, 5]
}

# 创建 XGBoost 分类器
xgb_clf = xgb.XGBClassifier()

# 使用网格搜索进行参数调优
grid_search = GridSearchCV(xgb_clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 打印最佳参数组合和对应的准确率
print("Best Parameters: ", grid_search.best_params_)
print("Best Accuracy: ", grid_search.best_score_)

# 使用最佳参数组合创建 XGBoost 分类器
best_xgb_clf = xgb.XGBClassifier(**grid_search.best_params_)

# 使用交叉验证计算模型在训练集上的性能
cross_val_scores = cross_val_score(best_xgb_clf, X_train, y_train, cv=5)
print("Cross-validation Scores: ", cross_val_scores)
print("Mean CV Accuracy: ", cross_val_scores.mean())

# 在测试集上进行预测
# 使用最佳参数组合拟合模型
best_xgb_clf.fit(X_train, y_train)

y_pred = best_xgb_clf.predict(X_test)

# 计算准确率、精确率、召回率、F1 分数
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)

# 绘制 ROC 曲线与计算 AUC
y_pred_prob = best_xgb_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
print("ROC AUC: ", roc_auc)

# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, label='ROC Curve (AUC = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# y_pred_prob = best_xgb_clf.predict_proba(X_test)[:, 1]
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')


print("ROC AUC: ", roc_auc)
# 计算混淆矩阵
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(conf_mat)
plt.show()