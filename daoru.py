import csv, pandas as pd, numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn import preprocessing
import csv
import numpy as np
from numpy import nan
from sklearn.datasets._base import Bunch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, log_loss, \
    classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import  make_classification
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
csvFile = open("数据集新.csv", "r")
csv_data = csv.reader(csvFile)
cancer = np.array([i for i in csv_data])
# # 使用Pandas导入CSV数据
# filename = '数据集.csv'
names = ['小臂极值', '小腿极值',  '小腿加速度极值']
# data = pd.read_csv(filename, names=names)
# cancer=np.array([i for i in data])
# print(cancer.shape)
#只取第1行，前9列
# attribute_names=cancer[0,:9]
da=cancer[0:, :3]
# print(da)
data=[]
#data这个数据全是str
for i in da:
    temp = []
    for j in i:
        if j == '?':
            temp.append(nan)
        else:
            temp.append(float(j))
    data.append(temp)
# print(data)
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler() #实例化
# scaler = scaler.fit(data) #fit，在这里本质是生成min(x)和max(x)
# result = scaler.transform(data) #通过接口导出结果
# print(result)
# attribute_names=cancer[0,:8]
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

# --------opetuna调参模块-------------
def objective(trial):
    classifier_name = trial.suggest_categorical("classifier", ["KNN"])
    # 设置想要调优的模型与模块
    if classifier_name == "SVC":
        # 设置分类向量机的一些参数如核函数,gamma值,C容忍度.
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid'])
        gamma = trial.suggest_float('gamma', 1e-5, 1e5)
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e-2, log=True)
        classifier_obj = SVC(C=svc_c, kernel=kernel, gamma=gamma)
    elif classifier_name == "RandomForest":
        # 设置随机森林的深度,决策树的个数
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        rf_n_estimators=trial.suggest_int("rf_n_estimators",3,15)
        classifier_obj = RandomForestClassifier(max_depth=rf_max_depth, n_estimators=rf_n_estimators)
    elif classifier_name == "Ridge":
        # 设置岭回归的学习率,权重值
        C = trial.suggest_loguniform('alpha', 1e-7, 1e-2)
        b = trial.suggest_int('b', 1, 32)
        classifier_obj = RidgeClassifier(alpha=C, class_weight={0: 1, 1: b}, random_state=0)
    elif classifier_name == "KNN":
        # 设置kNN的邻居数量,第四个参数代表是步长为2,也就是都是奇数的邻居个数
        n_num = trial.suggest_int('n_neighbors', 1, 100,1)
        classifier_obj = KNeighborsClassifier(n_neighbors=n_num)
    return cross_val_score(
        # 根据交叉验证的平均值作为调优模型的调优方向,这里设置为五折的交叉验证
        classifier_obj , X_train, y_train, n_jobs=-1, cv=5).mean()


def objective_lgb(trial):
    dtrain = lgb.Dataset(X_train, label=y_train)
    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(X_test)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(y_test, pred_labels)
    return accuracy
# clf = SVC()
# clf.fit(X_train, y_train)
# print(clf.score(X_test, y_test))
# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)
# print(knn.score(X_test, y_test))
print('Start training...')
# # clf = SVC()
# # clf.fit(X_train, y_train)
from xgboost import XGBClassifier, plot_importance

# model = XGBClassifier(n_estimators=100,learning_rate=0.05)
# model = XGBClassifier(n_estimators=150,learning_rate=0.05)
#
# model.fit(X_train,y_train)
# model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160)
# model.fit(X_train, y_train)
#
# # knn = KNeighborsClassifier()
# # knn.fit(X_train, y_train)
#
# #
# # # 创建模型，训练模型
# gbm = lgb.LGBMClassifier()
# # # gbm.fit(X_train, y_train)
# y_pred=model.predict(X_test)
# cnt1 = 0
# cnt2 = 0
# for i in range(len(y_test)):
#     if y_pred[i] == y_test[i]:
#         cnt1 += 1
#     else:
#         cnt2 += 1
#
# print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
# plot_importance(model)
# plt.show()

# from sklearn.model_selection import GridSearchCV
# parameters = {'max_depth':[3, 5, 6, 7, 9, 12, 15, 17, 25],'n_estimators':[50,100,150],'learning_rate':[0.01,0.05,0.1,0.2]}
# # model = XGBClassifier()
# grid_search = GridSearchCV(model,parameters,scoring='roc_auc',cv=5)
# grid_search.fit(X_train,y_train)
# print(grid_search.best_params_)
# grid_search.predict(X_test)



# print('Light6BM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
# print("precision :", precision_score(y_test, y_pred),"\n")
# print("Recall :",recall_score(y_test, y_pred),"\n")
# print("f1 score:", f1_score(y_test, y_pred), "\n")
# print('Training set score: {:.4f}'.format(grid_search.score(X_train, y_train)))
# print('Test set score: {:.4f}'.format(grid_search.score(X_test,y_test)))

if __name__ == "__main__":
    # -------optuna调优以及可视化显示--------
    study = optuna.create_study(direction="maximize")
    #首先使用optuna进行参数调参来找到最好的模型
    study.optimize(objective, n_trials=200)
    # n_trials:测试200次找到最佳参数
    print("最佳参数：", study.best_params)
    # 最佳参数： {'classifier': 'RandomForest', 'rf_max_depth': 15, 'rf_n_estimators': 14}
    print("最佳trial：", study.best_trial)
    # optuna的可视化
    best=study.best_params;
    optuna.visualization.plot_optimization_history(study).show()
    # 使用最佳的参数去训练模型,然后获取其相关的性能指标
    best_param=study.best_params
    # params = {
    #     'lambda_l1': 9.415506474764248e-07,
    #     'lambda_l2': 1.6877710196978973e-07,
    #     'num_leaves': 22,
    #     'feature_fraction': 0.8380886049353398,
    #     'bagging_fraction': 0.9267085932787693,
    #     'bagging_freq': 2,
    #     'min_child_samples': 6,
    #     'boosting_type': 'gbdt'  # 通过字符串指定 boosting_type 类型
    # }

    # clf = SVC(**best)
    clf = lgb.LGBMClassifier(    lambda_l1=9.415506474764248e-07,
    lambda_l2=1.6877710196978973e-07,
    num_leaves=22,
    feature_fraction=0.8380886049353398,
    bagging_fraction=0.9267085932787693,
    bagging_freq=2,
    min_child_samples=6)
    # # # gbm.fit(X_train, y_train)
    # clf=KNeighborsClassifier(n_neighbors=3)
    # clf = RandomForestClassifier(max_depth=22, n_estimators=15)
    # 最佳trial： FrozenTrial(number=79, values=[0.7857062146892655]
    clf.fit(X_train,y_train)
    predicted=clf.predict(X_test)

    print('Light6BM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, predicted)))
    print("precision :", precision_score(y_test, predicted),"\n")
    print("Recall :",recall_score(y_test, predicted),"\n")
    print("f1 score:", f1_score(y_test, predicted), "\n")
    print('Training set score: {:.4f}'.format(clf.score(X_train, y_train)))
    print('Test set score: {:.4f}'.format(clf.score(X_test,y_test)))
    conf_matrix = confusion_matrix(y_test, predicted)
    print(conf_matrix)
    print(
        f"Classification report for classifier {study}:\n"
        f"{classification_report(y_test, predicted)}\n"
    )
    # -------------------------------------