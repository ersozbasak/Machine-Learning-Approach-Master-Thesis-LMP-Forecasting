#!/usr/bin/env python
# coding: utf-8

# In[17]:


#ElasticNet Regresyon Model ve Tahmini
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV


# In[5]:


df = pd.read_csv("Hitters.csv")
df = df.dropna()
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df['Salary']
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


# In[7]:


enet_model = ElasticNet ().fit(X_train, y_train)


# In[8]:


enet_model.coef_


# In[9]:


enet_model.intercept_


# In[11]:


#tahmin
enet_model.predict(X_test)[0:10]


# In[15]:


y_pred = enet_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# In[16]:


r2_score(y_test,y_pred)


# In[18]:


#ModelTuning
enet_cv_model = ElasticNetCV(cv = 10).fit(X_train, y_train)


# In[19]:


enet_cv_model.alpha_


# In[20]:


enet_cv_model.intercept_


# In[21]:


enet_cv_model.coef_


# In[27]:


#final modeli; alpha parametresini kullanarak model fit edilmiştir
#alpas = 10**np.linspace(10,-2,100)*0,5
#enet_cv_model = ElasticNetCV(alphas = alphas, cv = 10).fit(X_train, y_train)
enet_tuned = ElasticNet(alpha = enet_cv_model.alpha_).fit(X_train, y_train)


# In[23]:


y_pred = enet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# In[24]:


get_ipython().run_line_magic('pinfo', 'ElasticNet')


# In[25]:





# In[26]:





# In[ ]:




