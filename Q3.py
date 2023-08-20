#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc
mpl.rcParams["font.sans-serif"] = ["kaiti"] # 设置中文字体
mpl.rcParams["axes.unicode_minus"] = False # 设置减号不改变


# In[18]:


data = pd.read_excel(r"D:\math\A题-附件\Q3是否患慢性病与居民信息饮食习惯.xlsx",index_col=0)


# In[19]:


Y = data.values[:,:2].copy()
X = data.values[:,2:].copy()


# In[20]:


Y[Y==1]=0
Y[Y==2]=1


# In[26]:


X = X
y = Y[:,0]

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立逻辑回归模型
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# 输出变量的系数
coefficients = model.coef_[0]
intercept = model.intercept_[0]
for i, coef in enumerate(coefficients):
    print(f'Variable {data.columns[2:][i]} coefficient: {coef:.4f}')

# 预测测试集
y_pred_prob = model.predict_proba(X_test)[:, 1]

# 计算ROC曲线和AUC值
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()


# In[34]:


X = X
y = Y[:,1]

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 建立逻辑回归模型
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# 输出变量的系数
coefficients = model.coef_[0]
intercept = model.intercept_[0]
for i, coef in enumerate(coefficients):
    print(f'Variable {data.columns[2:][i]} coefficient: {coef:.4f}')

# 预测测试集
y_pred_prob = model.predict_proba(X_test)[:, 1]

# 计算ROC曲线和AUC值
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()


# In[ ]:




