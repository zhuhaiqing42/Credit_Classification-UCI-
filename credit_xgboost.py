
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE # 导入SMOTE算法模块
import matplotlib.pyplot as plt
# from credit_logistic import get_onehot_data
import xgboost as xgb
from sklearn import metrics
import time
from datetime import timedelta


def get_source_data():
	# 从txt中读取原始数据,数据与文件在同一级目录下
	# data = pd.read_table('german.data-numeric.txt',header=None,delim_whitespace=True)
	data = pd.read_excel('default_of_credit_card_clients.xls',header=0)
	# print(data)
	# data_array = data.values[:20, :]
	x, y = np.split(data, (23,), axis=1)

	return x, y


def xgboost(x, y):
	#处理不平衡数据
	sm = SMOTE(random_state=42)    # 处理过采样的方法
	X, Y = sm.fit_sample(x, y)

	X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1,train_size=0.8)
	dtrain=xgb.DMatrix(X_train,label=y_train)
	dtest=xgb.DMatrix(X_test)

	params={'booster':'gbtree',
    	'objective': 'binary:logistic',
    	'eval_metric': 'auc',
    	'max_depth':4,
    	'lambda':10,
    	'subsample':0.75,
    	'colsample_bytree':0.75,
    	'min_child_weight':2,
    	'eta': 0.025,
    	'seed':0,
    	'nthread':8,
    	'silent':1}
	watchlist = [(dtrain,'train')]
	
	start = time.time()
	bst=xgb.train(params,dtrain,num_boost_round=1000,evals=watchlist)
	end = time.time()
	print("Time usage:", timedelta(seconds=int(round(end - start))))
	
	ypred=bst.predict(dtest)

	# 设置阈值, 输出一些评价指标
	y_pred = (ypred >= 0.5)*1


	print ('AUC: %.4f' % metrics.roc_auc_score(y_test,ypred))
	print ('ACC: %.4f' % metrics.accuracy_score(y_test,y_pred))
	print ('Recall: %.4f' % metrics.recall_score(y_test,y_pred))
	print ('F1-score: %.4f' %metrics.f1_score(y_test,y_pred))
	print ('Precesion: %.4f' %metrics.precision_score(y_test,y_pred))
	cm = metrics.confusion_matrix(y_test,y_pred) # cm 为混淆矩阵
	return cm


def cm_plot(cm):
	# 混淆矩阵可视化
	plt.matshow(cm, cmap=plt.cm.Blues) 
	plt.colorbar() 

	for x in range(len(cm)): 
		for y in range(len(cm)):
			plt.annotate(cm[y,x], xy=(x, y), horizontalalignment='center', verticalalignment='center')

	plt.ylabel('True label') 
	plt.xlabel('Predicted label') 
	return plt


if __name__ == '__main__':
	x, y = get_source_data()
	cm = xgboost(x, y)
	cm_plot(cm).show() # 混淆矩阵可视化