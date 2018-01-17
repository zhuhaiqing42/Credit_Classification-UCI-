# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 23:22:44 2017

@author: 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from time import time

def get_data():    
    data=pd.read_excel('D:\\default of credit card clients.xls', header=0,skiprows=[0],index_col=[0])
    x_feature = list(data.columns)
    x_feature.remove('default payment next month')
    x = data[x_feature]
    y = data['default payment next month']
    sm = SMOTE(random_state=42)    # 处理过采样的方法
    X, Y = sm.fit_sample(x, y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y,
                    test_size = 0.2, random_state = 0)
    return x_train, x_test, y_train, y_test

def RF_model(x_train, x_test, y_train, y_test):
    t1 = time()
    rf=RandomForestClassifier(n_estimators=30,max_depth=15)
    rf.fit(x_train,y_train)
    t2 = time()
    y_train_p=rf.predict(x_train)
    y_test_p=rf.predict(x_test)
    print ('----RandomForest----')
    print("Train set accuracy score: {:.5f}".format(accuracy_score(y_train_p, y_train)))
    print("Test set accuracy score: {:.5f}".format(accuracy_score(y_test_p, y_test)))
    print('Time: {:.1f} s'.format(t2 - t1))
    return y_test_p

def cm_plot(y_test, y_test_p,title):
    cm = confusion_matrix(y_test, y_test_p)
    print("Recall: ", cm[1,1]/(cm[1,0]+cm[1,1]))
    print("Precision: ",cm[1,1]/(cm[0,1]+cm[1,1]))
    plt.matshow(cm, cmap=plt.cm.Blues) 
    plt.colorbar() 
    for x in range(len(cm)): 
        for y in range(len(cm)):
            plt.annotate(cm[y,x], xy=(x,y),horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('%s'%title)
    plt.savefig('%s'%title,dpi=500)
    return plt

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_data()
    y_test_p = RF_model(x_train, x_test, y_train, y_test)
    cm_plot(y_test, y_test_p,'RandomForest').show() #显示混淆矩阵可视化结果











