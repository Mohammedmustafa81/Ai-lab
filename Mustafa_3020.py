#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


pip install pandas


# In[3]:


df=pd.read_csv("diabetes.csv")


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[9]:


x=df.iloc[:,:-1].to_numpy()
y=df.iloc[:,-1].to_numpy()


# In[10]:


df.iloc[:,:-1]


# In[11]:


x


# In[12]:


y


# In[13]:


x=df.iloc[:,:-1].to_numpy()
y=df.iloc[:,-1].to_numpy()


# In[15]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[17]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(criterion="entropy",random_state=0)
clf.fit(x_train,y_train)


# In[18]:


import matplotlib.pylot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.tree import plot_tree
plt.figure(figsize=(20,10))


# In[19]:


pip install matplotlib


# In[21]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.tree import plot_tree
plt.figure(figsize=(20,10))


# In[27]:


plot_tree(clf,feature_names=['Glucose','BMI'],class_names=['No','Yes'])
plt.show()


# In[31]:


clf.set_params(max_depth=2)


# In[32]:


clf.set_params(max_depth=2)


# In[33]:


clf.fit(x_train,y_train)
plt.figure(figsize=(20,10))
plot_tree(clf,feature_names=['Glucose','BMI'],class_names=['No','Yes'])
plt.show()


# In[34]:


predictions=clf.predict(x_test)


# In[36]:


predictions


# In[35]:


clf.predict([[90,20],[200,30]])


# In[38]:


from sklearn import metrics
cf=metrics.confusion_matrix(y_test,predictions)
cf


# In[ ]:




