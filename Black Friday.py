#!/usr/bin/env python
# coding: utf-8

# # STEP 1 READ DATA

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


# # STEP 2 HIGH LEVEL ANALYSIS

# In[2]:


#df.head()
#df.info()
#df.describe()
train.describe(include=['O'])
#test.describe(include=['O'])


# # STEP 3 NULL CHECK

# In[3]:


percentage=(train.isnull().sum()/train.shape[0]*100).sort_values(ascending=False)
print(percentage)
#percentage=(test.isnull().sum()/test.shape[0]*100).sort_values(ascending=False)
#print(percentage)


# # STEP 4 NULL CORRECTION

# In[4]:


train.fillna(0,inplace=True)
test.fillna(0,inplace=True)


# # STEP 5 LABEL ENCODING OR ONE HOT ENCODING

# In[5]:


import numpy as np
from sklearn import preprocessing
la = preprocessing.LabelEncoder()
#from sklearn.preprocessing import LableEncoder
#la=LableEncoder()
train['Product_ID']=la.fit_transform(train['Product_ID'])
train['Gender']=np.where(train['Gender']=='M',1,0)
train=pd.get_dummies(train)
train
train.info()


# In[6]:


import numpy as np
from sklearn import preprocessing
la = preprocessing.LabelEncoder()
#from sklearn.preprocessing import LableEncoder
#la=LableEncoder()
test['Product_ID']=la.fit_transform(test['Product_ID'])
test['Gender']=np.where(test['Gender']=='M',1,0)
test=pd.get_dummies(train)
test
test.info()


# # STEP 6 Feature enggby corr heatmap

# In[7]:


import seaborn as sns
corr=train.corr()
corr=np.abs(corr)
sns.set(rc={'figure.figsize':(20,16)})
heatmap=sns.heatmap(corr,annot=True)
heatmap


# # STEP 7 SEPERATE X AND Y

# In[8]:


x=train.drop(['Purchase'],axis=1)
y=train['Purchase']


# In[9]:


x=test.drop(['Purchase'],axis=1)
y=test['Purchase']


# # STEP 8 TRAIN AND TEST SPLIT

# In[10]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=100)


# # STEP 9 SCALLING

# In[11]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
scaler=sc.fit(x)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


# # STEP 10 BASLINE MODEL

# In[12]:


from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(x_train,y_train)
model.score(x_test,y_test)


# # STEP 11 HYPERPARA TUNING

# In[13]:


from sklearn.model_selection import RandomizedSearchCV
n_estimators =[int(x)for x in np.linspace(start=1,stop=200,num=4)]
max_features=['auto','sqrt']
max_depth=[int(x)for x in np.linspace(1,50,num=11)]
max_depth.append(None)
min_samples_split=[2,3,5,10]
min_samples_leaf=[1,2,4]
bootstrap=[True,False]
random_grid={'n_estimators':n_estimators,
             'max_features':max_features,
             'max_depth':max_depth,
             'min_samples_split':min_samples_split,
             'min_samples_leaf':min_samples_leaf,
             'bootstrap':bootstrap}
rf_random=RandomizedSearchCV(estimator = model,param_distributions=random_grid,cv=3)
rf_random.fit(x_train,y_train)


# # STEP 12 TESTING THE MODEL

# In[16]:


model=rf_random.best_estimator_
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
print(y_test[:10])
print(model.predict(x_test))


# In[18]:


Purchase=test['Purchase'] 
submission_df_1 = pd.DataFrame({
                  "Purchase": Purchase, 
                  "Purchase_Status":model.predict})


# In[20]:


submission_df_1.to_csv('submission_1.csv', index=False)


# In[ ]:




