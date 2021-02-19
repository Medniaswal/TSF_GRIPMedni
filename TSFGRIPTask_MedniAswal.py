#!/usr/bin/env python
# coding: utf-8

# # TASK1 - Prediction using Supervised ML
# 

# ## Student's Percentage Prediction Model by Medni Aswal

# ### Problem - Predicting percentage of student by the number of hours of studying. What will be the score is the students studies for 9.25 hours per day?

# In[22]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn


# In[23]:


url='http://bit.ly/w-data'
data=pd.read_csv(url)
data.head()


# In[24]:


len(data)


# In[25]:


data.info()


# In[26]:


data.describe()


# In[27]:


data.isnull().sum()


# In[28]:


data.plot(x="Hours",y="Scores",style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Score')
plt.show()


# In[29]:


plt.hist(data.Scores)


# In[30]:


sn.pairplot(data)


# In[31]:


data.corr()


# In[32]:


X=data["Hours"]
X


# In[33]:


Y=data["Scores"]
Y


# In[34]:


X=X.values.reshape(-1,1)
X.shape


# In[35]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=5)
len(X_train),len(X_test),len(Y_train),len(Y_test)


# In[36]:


#X-attribute(input),y-label(output)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
LinearRegression(copy_X=True, fit_intercept=True,n_jobs=None,normalize=False)
print(X_test)
Y_pred=reg.predict(X_test)


# #### VISUALIZATION

# In[37]:


plt.figure(figsize=(15,7))
plt.scatter(X_test,Y_test,color='magenta',s=70,label='Actual Score')
plt.scatter(X_test,Y_pred,color='yellow',s=70,label='Predicted Score')
plt.plot(X_test,Y_pred,color='green',label='Line of best fit')
plt.title('Number of Hours Studied v/s Percentage of Marks ')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.show()


# In[38]:


error=pd.DataFrame({'Actual':Y_test,'Predicted':Y_pred,'Absolute Error':abs(Y_test-Y_pred)})
error


# #### Finding the ROOT MEAN SQUARED ERROR(RMSE) & R-SQUARED VALUE

# In[39]:


from sklearn.metrics import r2_score,mean_squared_error
rmse=np.sqrt(mean_squared_error(Y_test,Y_pred))
print("RSMSE=",rmse)


# In[40]:


r2=r2_score(Y_test,Y_pred)
print("The regressor score is =",r2)


# In[41]:


test_Hours=np.array([9.25]).reshape(-1,1) #Creating a numpy array of the test independent variable 'Hours'
pred_Scores=reg.predict(test_Hours)
print("The predicted percentge score of a student studying for 9 hours 25 min a day is {} % ".format(np.round(pred_Scores[0],2)))


# ## THANK YOU
