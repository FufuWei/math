#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["kaiti"] # 设置中文字体
mpl.rcParams["axes.unicode_minus"] = False # 设置减号不改变


# In[2]:


# 数据导入 D4_30.xlsx
data = pd.read_excel(r"D:\math\A题-附件\Q1数据预处理.xlsx")


# In[3]:


# 缺失值填充-1,并提取数组
data1 = data.fillna(-1).values.copy() 


# In[4]:


# 缺失值填充（是否食用过食物）
j = 0
for i in range(data1.shape[0]):
    for j in range(0,data1.shape[1],5):
        if data1[i,j]>0:
            pass
        elif np.any(data1[i,j+1:j+5] >0):
            data1[i,j] = 1
        else:
            data1[i,j] = 2


# In[5]:


# 对于是否使用过食物,纠正错误值（即没有使用频率和量的选项改为2）
j = 0
for i in range(data1.shape[0]):
    for j in range(0,data1.shape[1],5):
        if np.all(data1[i,j+1:j+5]<0):
            data1[i,j] =2


# In[6]:


# 将使用频率统一为月
j = 0
for i in range(data1.shape[0]):
    for j in range(0,data1.shape[1],5):
        if data1[i,j]==2:
            data1[i,j+3] = 0
            data1[i,j+4] = 0
        else:
            if data1[i,j+3]>0:
                pass
            elif data1[i,j+2]>0:
                # 月频率为周的4倍
                data1[i,j+3] = 4*data1[i,j+2].copy()
            elif data1[i,j+1]>0:
                # 月频率为日的30倍
                data1[i,j+3] = 30*data1[i,j+1].copy()


# In[7]:


# 将使用频率的缺失值替换为平均值
for j in range(0,data1.shape[1],5):
    m = np.mean(data1[data1[:,j+3]>0,j+3])
    data1[data1[:,j+3]<0,j+3] = round(m,2)


# In[8]:


# 将平均使用量的缺失值使用同频率下的均值填充
for i in range(data1.shape[0]):
    for j in range(0,data1.shape[1],5):
        if data1[i,j+4]<0:
            data1[i,j+4] = round(np.mean(data1[data1[:,j+3]==data1[i,j+3],j+4]),2)
        if data1[i,j+4]<0:
            data1[i,j+4] = round(np.mean(data1[:,j+4]),2)


# In[9]:


# 去掉日,周两个频率的列
data2 = data1[:,np.array([0,3,4])+0*5].copy()
for j in range(5,data1.shape[1],5):
    data2 = np.c_["1",data2,data1[:,np.array([0,3,4])+j].copy()]


# In[10]:


#pd.DataFrame(data2,columns=[i for i in data.columns.values if i[0]!="U"]).to_excel(r"E:\zm\230726深圳杯\A1\1数据\D4_30(处理后).xlsx")


# In[11]:


Data1 = pd.DataFrame(data2,columns=[i for i in data.columns.values if i[0]!="U"])
Data1


# In[12]:


# 统计每种食物使用的人数占总人数的比值 
swzb = np.array([np.sum(data2[:,j]==1) for j in range(0,data2.shape[1],3)])/data2.shape[0]
print(Data1.columns[::3].values)
print(swzb)


# In[13]:


plt.figure(figsize = (9,4))
br = plt.bar(np.arange(swzb.shape[0]),swzb,edgecolor = "k")
for i in range(len(br.patches)):
    br.patches[i].set_fc(plt.cm.autumn_r(swzb[i]))
plt.xticks(np.arange(swzb.shape[0]),[f"D{i}"for i in range(4,31)])
plt.xlabel("问题编号D4-D30",fontsize = 14)
plt.ylabel("食物的食用占总人数比值",fontsize = 14)
for i in np.arange(swzb.shape[0]):
    plt.text(i,swzb[i]+0.02 ,round(swzb[i],2),ha="center",)
#plt.savefig(r"E:\zm\t\1.png",dpi = 1000 ,format = "png")
plt.show()


# In[14]:


# 统计每种食物使用的平均食用频率(月每次)
swpl = np.array([np.sum(data2[:,j]) for j in range(1,data2.shape[1],3)])/data2.shape[0]
print(Data1.columns[1::3].values)
print(swpl)


# In[15]:


plt.figure(figsize = (9,4))
br = plt.bar(np.arange(swpl.shape[0]),swpl,edgecolor = "k")
for i in range(len(br.patches)):
    br.patches[i].set_fc(plt.cm.autumn_r(swzb)[i])
for i in np.arange(swzb.shape[0]):
    plt.text(i,swpl[i]+1 ,round(swpl[i]),ha="center")
plt.xticks(np.arange(swzb.shape[0]),[f"D{i}"for i in range(4,31)])
plt.xlabel("问题编号D4-D30",fontsize = 14)
plt.ylabel("食物的食用频率(次/月)",fontsize = 14)
#plt.savefig(r"E:\zm\t\2.png",dpi = 1000 ,format = "png")
plt.show()


# In[18]:


# 统计每种食物使用的平均使用量
#pd.DataFrame(data2[:,[j for j in range(2,data2.shape[1],3)]],columns=[i[:-4]+"平均食用量" for i in Data1.columns[1::3].values]).describe().to_excel(r"E:\zm\230726深圳杯\A1\1数据\食物使用的平均使用量.xlsx")


# In[19]:


# 可视化统计每种食物使用的平均使用量分布.
plt.figure(figsize = (9,4))
bx = plt.boxplot(data2[:,[j for j in range(2,data2.shape[1],3)]],showfliers = 0)
plt.xticks(np.arange(1,swzb.shape[0]+1),[f"D{i}"for i in range(4,31)])
plt.xlabel("问题编号D4-D30",fontsize = 14)
plt.ylabel("平均每次使用量",fontsize = 14)
# plt.savefig(r"E:\zm\t\3.png",dpi = 1000 ,format = "png")
plt.show()


# In[20]:


# 导入 D31-37
Data3 = pd.read_excel(r"D:\math\A题-附件\Q1_2调味料预处理.xlsx")
#Data3.describe().to_excel(r"E:\zm\230726深圳杯\A1\1数据\D31_37(统计学描述).xlsx")


# In[22]:


plt.figure(figsize = (5,4))
bx = plt.boxplot(Data3.fillna(0).values,showfliers = 0)
plt.xticks(np.arange(1,8),Data3.columns)
plt.ylabel("平均每月食用量(两)")
#plt.savefig(r"E:\zm\230726深圳杯\A1\1t\4.png",dpi = 2000 ,format = "png")
plt.show()


# In[ ]:




