from sklearn.ensemble import GradientBoostingClassifier
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
'''
author:朱海清
'''

def get_source_data():
	# 从txt中读取原始数据,数据与文件在同一级目录下
	# data = pd.read_table('german.data-numeric.txt',header=None,delim_whitespace=True)
	data = pd.read_excel('default_of_credit_card_clients.xls',header=0)
	# print(data)
	# data_array = data.values[:20, :]
	x, y = np.split(data, (23,), axis=1)

	return x, y


def GBDT_model(x,y):


	#处理不平衡数据
	sm = SMOTE(random_state=42)    # 处理过采样的方法
	X, Y = sm.fit_sample(x, y)

	X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1,train_size=0.8)
	start = time.time()
	clf_model = GradientBoostingClassifier(
    			loss='deviance',
    			learning_rate=0.01,
    			n_estimators=800,
    			subsample=0.8,
    			max_features=1,
    			max_depth=10,
    			verbose=2
				)

	# 训练模型
	clf_model.fit(X_train, y_train)
	end = time.time()
	print("Time usage:", timedelta(seconds=int(round(end - start))))
	# 评估模型
	prediction_train = clf_model.predict(X_train)
	cm_train = confusion_matrix(y_train, prediction_train)
	prediction_test = clf_model.predict(X_test)
	cm_test = confusion_matrix(y_test, prediction_test)

	print ("Confusion matrix for training dataset is \n%s\n for testing dataset is \n%s." \
		   % (cm_train, cm_test))

	print("train accuracy:", clf_model.score(X_train, y_train))
	print("test accuracy:", clf_model.score(X_test, y_test))

	precision_accuracy = cm_test[1,1]/(cm_test[1,1]+cm_test[0,1])
	print("Precision:", precision_accuracy)
	
	recall_accuracy = cm_test[1,1]/(cm_test[1,1]+cm_test[1,0])
	print("Recall:", recall_accuracy)
	return cm_test


def cm_plot(cm):
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
	# x, y = get_onehot_data()
	cm_test = GBDT_model(x,y)
	# xgboost_result = xgboost(x, y)
	cm_plot(cm_test).show() #显示混淆矩阵可视化结果

