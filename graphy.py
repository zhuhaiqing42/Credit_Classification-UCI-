# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:51:47 2017

@author: 朱浩
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_excel('D:\\default of credit card clients.xls', header=0,skiprows=[0],index_col=[0])
y_out=data.groupby('default payment next month').size()

data.loc[:,'default payment next month'][data['default payment next month']==1] = 'yes'
data.loc[:,'default payment next month'][data['default payment next month']==0] = 'no'

x1=data.loc[:,['LIMIT_BAL','default payment next month']]
x5=data.loc[:,['AGE','default payment next month']]
####################绘制连续变量的密度图#####################
import seaborn as sns
plt.show()
sns.distplot(x1['LIMIT_BAL'][x1['default payment next month']=='yes'],color='b',label='yes')
sns.distplot(x1['LIMIT_BAL'][x1['default payment next month']=='no'],color='r',label='no')
plt.legend()
plt.title('LIMIT_BAL')
plt.savefig('LIMIT_BAL.png',dpi=600)

plt.show()
sns.distplot(x5['AGE'][x5['default payment next month']=='yes'],color='b',label='yes')
sns.distplot(x5['AGE'][x5['default payment next month']=='no'],color='r',label='no')
plt.legend()
plt.title('AGE')
plt.savefig('AGE.png',dpi=600)

plt.show()
sns.distplot(x5['AGE'],color='b')
plt.legend()
plt.title('AGE')
plt.savefig('AGE_total.png',dpi=600)

#####################绘制离散变量的柱状图#####################
x2=data.groupby(['default payment next month','SEX']).size()
x2=x2.unstack()
x2.loc['yes']=x2.loc['yes']/6636
x2.loc['no']=x2.loc['no']/23364
x2.plot(kind='bar')
plt.title('Gender')
plt.legend(['male','female'])
plt.savefig('SEX.png',dpi=600)
plt.show()

data.loc[:,'EDUCATION'][data['EDUCATION']==0] = 4
data.loc[:,'EDUCATION'][data['EDUCATION']==5] = 4
data.loc[:,'EDUCATION'][data['EDUCATION']==6] = 4
x3=data.groupby(['default payment next month','EDUCATION']).size()
x3=x3.unstack()
x3.loc['yes']=x3.loc['yes']/6636
x3.loc['no']=x3.loc['no']/23364
x3.plot(kind='bar')
plt.title('Education')
plt.legend(['graduate school','university','high school','others'])
plt.savefig('Education.png',dpi=600)
plt.show()

x3_t=data.groupby(['EDUCATION']).size()
x3_t.plot(kind='bar')
plt.title('Education')
plt.legend(['graduate school','university','high school','others'])
plt.savefig('Education_total.png',dpi=600)
plt.show()

data.loc[:,'MARRIAGE'][data['MARRIAGE']==0] = 3
x4=data.groupby(['default payment next month','MARRIAGE']).size()
x4=x4.unstack()
x4.loc['yes']=x4.loc['yes']/6636
x4.loc['no']=x4.loc['no']/23364
x4.plot(kind='bar')
plt.title('Marriage')
plt.legend(['married','single','others'])
plt.savefig('Marriage.png',dpi=600)
plt.show()



