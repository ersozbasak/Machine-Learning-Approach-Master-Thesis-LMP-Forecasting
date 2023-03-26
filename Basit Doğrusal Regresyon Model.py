#!/usr/bin/env python
# coding: utf-8

# In[141]:


import pandas as pd
df = pd.read_csv("Advertising.csv")
#iloc ile index hatası giderildi
df = df.iloc[:,1:len(df)]
df.head()


# In[143]:


import seaborn as sns
sns.jointplot(x = "TV", y = "sales", data = df, kind = "reg" );


# In[170]:


#MODELLEME/ MODEL KURMA
from sklearn.linear_model import LinearRegression
X = df[["TV"]]
X.head()


# In[171]:


y = df[["sales"]]
y.head()


# In[151]:


reg = LinearRegression()
model = reg.fit(X, y)
reg = LinearRegression()


# In[153]:


model = reg.fit(X, y)
model


# In[154]:


model.intercept_


# In[155]:


model.coef_


# In[156]:


model.score(X, y)


# In[29]:


#TAHMİN sklearn ile (ML için özelleştirilmiştir.)

import matplotlib.pyplot as plt


# In[160]:


g = sns.regplot(df["TV"], df["sales"], ci = None, scatter_kws={'color':'r', 's':9})
g.set_title("Model Denklemi: Sales 7.03 + TV*0.05")
plt.xlim(-10,310)
plt.ylim(bottom=0)


# In[172]:


model.predict([[165]])


# In[173]:


yeni_veri = [[5],[15],[30]]


# In[163]:


model.predict(yeni_veri)


# In[174]:


model.intercept_ + model.coef_*165


# In[45]:


#HATA kısmı MSE ve RMSE
model.predict(X)[0:6]


# In[103]:


gercek_y = y[0:10]


# In[106]:


tahmin_edilen_y = pd.DataFrame(model.predict(X)[0:10])


# In[114]:


hatalar = pd.concat([gercek_y, tahmin_edilen_y], axis=1)
hatalar


# In[119]:


hatalar.columns = ["gercek_y", "tahmin_edilen_y"]
hatalar


# In[120]:


hatalar["hata"] = hatalar["gercek_y"] - hatalar["tahmin_edilen_y"]
hatalar


# In[124]:


#HATA KARE
hatalar["hata_kare"] = hatalar["hata"]**2
hatalar


# In[136]:


#HATA KARELER ORT
np.mean(hatalar["hata_kare"])


# In[ ]:




