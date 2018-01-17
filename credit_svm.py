 #-*- coding: utf-8 -*-
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# from sklearn import cross_validation
# from imblearn.over_sampling import SMOTE # 导入SMOTE算法模块
# import matplotlib.pyplot as plt
# from credit_logistic import get_onehot_data
'''
author:朱海清
注：由于原始样本有3万个,并且我们做了样本平衡,故此程序运行时间较长,大概要好几个小时

'''

def get_source_data():
	# 从txt中读取原始数据
	# data = pd.read_table('german.data-numeric.txt',header=None,delim_whitespace=True)
	# data = pd.read_excel('default_of_credit_card_clients.xls',header=0)
	data = pd.read_excel('/BIGDATA/sysu_tanjun_1/zhuhaiqing/default_of_credit_card_clients.xls',header=0)
	# print(data)
	data_array = data.values[:200, :]
	x, y = np.split(data_array, (23,), axis=1)
	return x,y


def svm_model(x, y):
	# 使用sklearn库构建SVM模型
	# sm = SMOTE(random_state=42)    # 处理过采样的方法
	# X, Y = sm.fit_sample(x, y)
	x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8,test_size=0.2)
	# print(y_train)

	# clf = svm.SVC(C=0.1, kernel='rbf', decision_function_shape='ovr')
	clf = svm.SVC(C=0.1, kernel='linear', gamma='auto', decision_function_shape='ovr',class_weight='balanced')
	clf.fit(x_train, y_train.ravel())

	print("train accuracy:", clf.score(x_train, y_train))  # 精度
	print("test_accuracy:", clf.score(x_test, y_test))
	# 10折交叉验证
	# y_tmp = y[:, 0]
	# scores = cross_validation.cross_val_score(clf, x, y_tmp, cv=10)
	# print(scores.mean())


	# 评估模型
	prediction_train = clf.predict(x_train)
	cm_train = confusion_matrix(y_train, prediction_train)
	prediction_test = clf.predict(x_test)
	cm_test = confusion_matrix(y_test, prediction_test)
	print ("Confusion matrix for training dataset is \n%s\n for testing dataset is \n%s."\
			 % (cm_train, cm_test)) # 混淆矩阵
	return cm_test


def cm_plot(cm):
	plt.matshow(cm, cmap=plt.cm.Greens) 
	plt.colorbar() 

	for x in range(len(cm)): 
		for y in range(len(cm)):
			plt.annotate(cm[y,x], xy=(x, y), horizontalalignment='center', verticalalignment='center')

	plt.ylabel('True label') 
	plt.xlabel('Predicted label') 
	return plt


if __name__ == '__main__':
	# x, y = get_onehot_data()
	x, y = get_source_data()
	cm_test = svm_model(x, y)
	# cm_plot(cm_test).show() #显示混淆矩阵可视化结果


