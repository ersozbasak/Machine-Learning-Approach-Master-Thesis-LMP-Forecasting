#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV


# In[37]:


df = pd.read_csv("Hitters.csv")
df = df.dropna()
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df['Salary']
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


# In[38]:


df.head()


# In[15]:


df.shape


# In[16]:


ridge_model = Ridge(alpha = 0.1).fit(X_train, y_train)


# In[20]:


ridge_model.coef_


# In[21]:


ridge_model.intercept_


# In[27]:


np.linspace(10,-2,100)


# In[30]:


lambdalar = 10**np.linspace(10,-2,100)*0.5


# In[31]:


lambdalar


# In[39]:


ridge_model = Ridge()
katsayılar = []
for i in lambdalar:
    ridge_model.set_params(alpha = i)
    ridge_model.fit(X_train, y_train)
    katsayılar.append(ridge_model.coef_)


# In[33]:


katsayılar


# In[40]:


ax = plt.gca()
ax.plot(lambdalar, katsayılar)
ax.set_xscale("log")


# In[45]:


#TAHMİN
ridge_model
ridge_model = Ridge().fit(X_train, y_train)


# In[46]:


y_pred = ridge_model.predict(X_train)


# In[47]:


y_pred[0:10]


# In[48]:


y_train[0:10]


# In[51]:


RMSE = np.sqrt(mean_squared_error(y_train, y_pred))
RMSE


# In[55]:


from sklearn.model_selection import cross_val_score
np.sqrt(np.mean(-cross_val_score(ridge_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error" )))


# In[57]:


#test hatası/test setinde başarımızı deneriz
y_pred = ridge_model.predict(X_test)
#X_test te bulunan bağımsız değişkenleri kullanarak bağımlı değişkenlerin tahminidir.


# In[58]:


RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
RMSE


# In[ ]:




