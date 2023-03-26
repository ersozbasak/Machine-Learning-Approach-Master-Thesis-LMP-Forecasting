#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV


# In[38]:


#Veri Seti
df = pd.read_csv("Hitters.csv")
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df["Salary"]
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, test_size = 0.25, 
                                                    random_state = 42)


# In[34]:


df.head()


# In[28]:


df.shape


# In[43]:


lasso_model = Lasso().fit(X_train, y_train)
lasso_model.intercept_


# In[41]:


lasso_model.coeff_


# In[42]:


lasso = Lasso()
coefs = []
alphas = np.random.randint(0,1000,10)
for a in alphas:
    lasso.set_params(alpha = a)
    lasso.fit(X_train, y_train)
    coefs.append()lasso.coef_


# In[46]:


ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale("log")


# In[ ]:




