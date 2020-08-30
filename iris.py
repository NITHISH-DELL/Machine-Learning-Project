#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
from pandas.plotting import scatter_matrix 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier


# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


data=pd.read_csv("iris.csv")


# In[19]:


print(data.shape)


# In[20]:


print(data.head(10))


# In[21]:


print(data.describe())


# In[25]:


print(data.groupby("Species").size())


# In[30]:


data.plot(kind="box",layout=(2,2),sharex=False,sharey=False)
plt.show()


# In[31]:


data.hist()
plt.show()


# In[33]:


scatter_matrix(data)
plt.show()


# In[36]:


array=data.values
x=array[:,0:4]
y=array[:,4]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[56]:


model=[]
model.append(("LR",LogisticRegression(solver="liblinear",multi_class="over")))
model.append(("LDA",LinearDiscriminantAnalysis()))
model.append(("KNN",KNeighborsClassifier()))
model.append(("NB",GaussianNB()))
model.append(("svm",SVC(gamma="auto")))


# In[ ]:





# In[ ]:





# In[62]:


result=[]
names=[]
for name,models in model:
    kfold=StratifiedKFold(n_splits=10,random_state=1)
    cv_result=cross_val_score(models,x_train,y_train,cv=kfold,scoring="accuracy")
    result.append(name)
    names.append(name)
    print("%s:%f(%f)"% (name,cv_result.mean(),cv_result.std()))


# In[66]:


plt.boxplot(result,labels=names)
plt.title("algorithm Comparision")
plt.show()


# In[67]:


model=svm(gamma="auto")
model.fit(x_train,y_train)
pred=model.predict(x_test)
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,prediction))
print(Classification_report(y_test,pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




