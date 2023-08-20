#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from scipy.stats import pearsonr
mpl.rcParams["font.sans-serif"] = ["kaiti"] # 设置中文字体
mpl.rcParams["axes.unicode_minus"] = False # 设置减号不改变


# In[2]:


# 导入数据
D1 = pd.read_excel(r"D:\math\A题-附件\Q2居民信息与饮食习惯.xlsx",index_col = 0)


# In[3]:


data = D1.values
# 将出生日期，转化成年龄
data[:,0] = 2023-data[:,0].copy()
# 将年龄进行分段
data[data[:,0]>65,0]=-4
data[data[:,0]>45,0]=-3
data[data[:,0]>18,0]=-2
data[data[:,0]>=0,0]=-1
data[:,0] = -data[:,0].copy()
print("0-18岁居民有:",np.sum(data[:,0]==1))
print("18-45岁居民有:",np.sum(data[:,0]==2))
print("45-65岁居民有:",np.sum(data[:,0]==3))
print("65+岁居民有:",np.sum(data[:,0]==4))
idx = D1.columns.values.copy()
idx.astype(np.str_)
for i in range(7,idx.shape[0]-2-7,3):
    idx[i+2] = idx[i][3:]+idx[i+2][:-2] if i>5 else idx[i][3:]+idx[i+2]


# In[17]:


# 对不同年龄段的居民在饮食习惯上进行方差检验
x= data[:,5:data.shape[1]-7]
y= data[:,0]
md1 = MANOVA(x,y)
print(md1.mv_test().results["x0"]["stat"])


# In[18]:


# 对不同性别的居民在饮食习惯上进行方差检验
x= data[:,5:data.shape[1]-7]
y= data[:,1]
md1 = MANOVA(x,y)
print(md1.mv_test().results["x0"]["stat"])


# In[19]:


# 对不同文化程度的居民在饮食习惯上进行方差检验
x= data[:,5:data.shape[1]-7]
y= data[:,2]
md1 = MANOVA(x,y)
print(md1.mv_test())


# In[21]:


# 对不同婚姻状况的居民在饮食习惯上进行方差检验
x= data[:,5:data.shape[1]-7]
y= data[:,3]
md1 = MANOVA(x,y)
print(md1.mv_test())


# In[22]:


# 对不同职业的居民在饮食习惯上进行方差检验
x= data[:,5:data.shape[1]-7]
y= data[:,4]
md1 = MANOVA(x,y)
print(md1.mv_test())


# In[11]:


##### 计算居民个人信息饮食习惯的相关性
svm = []
c = np.zeros((5,data.shape[1]-7-5))
for i in range(0,5):
    k = 0
    for j in range(5,data.shape[1]-7):
        x = data[:,i]
        y = data[:,j]
        # 执行 Pearson 相关性分析
        correlation_coefficient, p_value = pearsonr(x,y)
        alpha = 0.05 # 显著水平
        if p_value < alpha:
            print("在显著性水平 {} 下，[{}]和[{}]之间的相关性是显著的。".format(alpha,idx[i],idx[j]))
            svm += ["在显著性水平 {} 下，[{}]和[{}]之间的相关性是显著的。".format(alpha,idx[i],idx[j])]
        else:
            print("在显著性水平 {} 下，[{}]和[{}]之间的相关性不显著。".format(alpha,idx[i],idx[j]))
        c[i,k]=round(correlation_coefficient,4)
        k+=1


# In[15]:


for j in range(0,data.shape[1]-7-5,5):
    if j+5>data.shape[1]-7-5:
        break
    try:
        plt.figure(figsize = (6,6))
        plt.imshow(c[:,j:j+5])
        plt.xticks([0,1,2,3,4],idx[5:86][j:j+5],rotation=30)
        plt.yticks([0,1,2,3,4],idx[:5],rotation=0)
        plt.colorbar()
        plt.savefig(r"E:\zm\230726深圳杯\A2(2)\2t\{}.png".format(j),dpi=500)
        plt.show()
    except:pass


# In[ ]:




