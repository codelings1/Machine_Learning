#!/usr/bin/env python
# coding: utf-8

# In[94]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn


# In[95]:


trainData = pd.read_csv(r"C:\Users\sarthak.agarwal\Downloads\titanic\titanic\train.csv")
trainData.head()


# In[96]:


testData = pd.read_csv(r"C:\Users\sarthak.agarwal\Downloads\titanic\titanic\test.csv")
testData.head()


# In[97]:


print(trainData.shape)


# In[98]:


trainData.hist(bins=50,figsize=[10,14])


# In[99]:


trainData.info()


# In[100]:


trainData.describe()


# In[101]:


trainData2 = pd.cut(trainData['Age'],bins=[0,20,40,60,80,np.inf],labels=[1,2,3,4,5])
trainData2.hist()


# In[102]:


trainData.select_dtypes(include='O').head()


# In[103]:


trainData.describe(include=['O'])


# In[104]:


trainData.select_dtypes(include=['float','int', 'int64']).head()


# In[105]:


type(trainData['PassengerId'][0])


# In[106]:


nullObjectCount = trainData.select_dtypes(include='object').isnull().sum()


# In[107]:


nullObjectCount


# In[108]:


nullObjectCountNumerical = trainData.select_dtypes(include=['int','float','int64']).isnull().sum()
nullObjectCountNumerical


# In[109]:


trainData[trainData['Age'].isnull()].head()


# In[110]:


meanAge = trainData[trainData['Age'].notnull()]['Age'].mean()
print(meanAge)
print(trainData['Age'].mean())


# # Handling all null values

# In[111]:


trainData.dropna(subset=['Survived','Pclass','Fare','Sex'], axis = 0, inplace = True)


# In[112]:


trainData.shape


# In[113]:


embarkedMode = trainData['Embarked'].mode()[0]
embarkedMode


# In[114]:


trainData['Name'].fillna("Name", inplace=True)
trainData['Sex'].fillna("Sex", inplace=True)
trainData['Ticket'].fillna("0000", inplace=True)
trainData['SibSp'].fillna(0, inplace=True)
trainData['Parch'].fillna(0, inplace=True)
trainData['Embarked'].fillna(embarkedMode, inplace=True)
# Same should be done for test data


# In[115]:


trainData['Embarked'].value_counts()


# # Replacing null values in age and Embarked by mean age and mode embarked

# In[116]:


for col in ['Age']:
    trainData['Age'] = trainData[col].fillna(meanAge)
    testData['Age'] = testData[col].fillna(meanAge)
trainData['Age'].isnull().sum()


# In[117]:


modeEmbark = trainData['Embarked'].mode()
modeEmbark.values[0]


# In[118]:


for col in ['Embarked']:
    trainData['Embarked'] = trainData[col].fillna(modeEmbark.values[0])
    testData['Embarked'] = testData[col].fillna(modeEmbark.values[0])
trainData['Embarked'].isnull().sum()


# In[119]:


trainData.head()


# # As cabin has many null values, remove that column

# In[120]:


nullObjectCount.sort_values(ascending=False)


# In[121]:


transformedtrainData = trainData.drop("Cabin", axis  = 1)
transformedtestData = testData.drop("Cabin", axis  = 1)
transformedtrainData.head()


# In[122]:


corrMatrix = trainData.corr()
corrMatrix


# In[123]:


plt.figure(figsize=[15,8])
sbn.heatmap(corrMatrix, annot=True)
#plt.plot(corrHeatmap)
plt.show()


# # We can see that Pclass and Fare are inversely related, also PcLass and Survuved are also related

# In[124]:


plt.plot(transformedtrainData['Fare']);
plt.show()


# In[125]:


transformedtrainData.head()


# # Removing columns that are not needed : PassengerId, Name, SibSp, Parch, Embarked, Ticket

# In[126]:


transformedtrainData = transformedtrainData.drop(['PassengerId','Name','Ticket'], axis = 1)
transformedtestData = transformedtestData.drop(['PassengerId','Name','Ticket'], axis = 1)


# # Converting Categorical data to numerical

# In[127]:


transformedtrainData = pd.get_dummies(transformedtrainData, columns=['Sex','Embarked'])
transformedtestData=pd.get_dummies(transformedtestData, columns=['Sex','Embarked'])
transformedtrainData.head()


# In[128]:


transformedtrainDataCorr = transformedtrainData.corr()
plt.figure(figsize=[15,8])
sbn.heatmap(transformedtrainDataCorr, annot=True)
#plt.plot(corrHeatmap)
plt.show()


# # Now we can drop one column from all categorical columns created

# In[129]:


transformedtrainData = transformedtrainData.drop(['Sex_male','Embarked_C'], axis = 1)


# In[130]:


transformedtrainDataCorr = transformedtrainData.corr()
plt.figure(figsize=[15,8])
sbn.heatmap(transformedtrainDataCorr, annot=True)
#plt.plot(corrHeatmap)
plt.show()


# In[131]:


from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


# In[132]:


# from sklearn.model_selection import StratifiedShuffleSplit
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 
# for train_index, test_index in split.split(housing, housing["income_cat"]): 
#     strat_train_set = housing.loc[train_index]   
#     strat_test_set = housing.loc[test_index]


# # Create age and fare category and on basis of these variables do better model fitting

# In[133]:


transformedtrainData.head()
transformedtrainData


# In[134]:


transformedtrainData['Age'].hist()


# In[135]:


ageCat = transformedtrainData[((transformedtrainData['Age'] >= 50) & (transformedtrainData['Age'] < np.inf) & 
                               (transformedtrainData['Survived'] == 0))].count()
ageCat


# In[136]:


ageCat = transformedtrainData[((transformedtrainData['Age'] >= 50) & (transformedtrainData['Age'] < np.inf) & 
                               (transformedtrainData['Survived'] == 0))]


# In[137]:


plt.plot(ageCat['Age'], ageCat['Survived'] )
plt.show()


# In[138]:


transformedtrainData['AgeCat'] = pd.cut(transformedtrainData['Age'],bins=[0,9,20,25,30,40,50,200], labels=[1,2,3,4,5,6,7])
transformedtrainData['AgeCat'].head()


# In[139]:


transformedtrainData['AgeCat'].value_counts()


# In[140]:


transformedtrainData[transformedtrainData['Survived'] == 1]['AgeCat'].value_counts()/ transformedtrainData['AgeCat'].value_counts()


# In[141]:


transformedtrainData[transformedtrainData['Survived'] == 0]['AgeCat'].value_counts()/ transformedtrainData['AgeCat'].value_counts()


# In[142]:


transformedtrainData['fareCat'] = pd.cut(transformedtrainData['Fare'],bins=[-1,20,100,263,10000], labels=[1,2,3,4])
transformedtrainData['fareCat'].value_counts()


# In[143]:


transformedtrainData[transformedtrainData['fareCat'].isnull()]


# In[144]:


transformedtrainData[transformedtrainData['Survived'] == 1]['fareCat'].value_counts() / transformedtrainData['fareCat'].value_counts()


# # Dropping columns Age and Fare

# In[145]:


transformedtrainData2 = transformedtrainData.drop(['Age','Fare'] ,axis = 1)


# In[146]:


transformedtrainData2.head()

print(transformedtrainData2.shape)
print(train_index.max())
print(test_index.max())
# In[147]:


transformedtrainData2['fareCat'].value_counts()

from sklearn.model_selection import StratifiedShuffleSplit
shuffleSplitStrategy = StratifiedShuffleSplit(n_splits=1, test_size = 0.2, random_state = 42)
for train_index, test_index in shuffleSplitStrategy.split(transformedtrainData2 ,transformedtrainData2['fareCat']):
    print(train_index)
    print(test_index)
    strat_train_set1 = transformedtrainData2.loc[train_index]
    strat_test_set1 = transformedtrainData2.loc[test_index]
    
# In[148]:


transformedtrainData2.isnull().sum()


# In[149]:


X_train, X_test, Y_train, Y_test = train_test_split(transformedtrainData2.drop('Survived',axis = 1),transformedtrainData2['Survived'],test_size = 0.1)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[150]:


X_train.head()


# In[151]:


scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[152]:


X_train


# In[153]:


X_test


# # Scaling the data

# In[154]:


X_train_pd = pd.DataFrame(X_train)

X_train_pd.isnull().sum()


# # Comparing more than one models

# In[155]:


ClassificationModel = LogisticRegressionCV().fit(X_train,Y_train)


# # Working on self created test data

# In[156]:


ClassificationModel.score(X_test,Y_test)


# In[163]:


from sklearn.tree import DecisionTreeClassifier


# In[164]:


decision_tree = DecisionTreeClassifier()


# In[167]:


decision_tree.fit(X_train,Y_train)


# In[168]:


decision_tree.score(X_test,Y_test)


# In[ ]:










# In[174]:


from sklearn.linear_model import SGDClassifier
sgdClassifier = SGDClassifier().fit(X_train,Y_train)
sgdClassifier.score(X_test, Y_test)


# In[173]:


from sklearn.linear_model import Perceptron
pcpClassifier = Perceptron().fit(X_train,Y_train)
pcpClassifier.score(X_test, Y_test)


# In[172]:


from sklearn.naive_bayes import GaussianNB
gnbClassifier = GaussianNB().fit(X_train,Y_train)
gnbClassifier.score(X_test, Y_test)


# In[169]:


from sklearn.svm import SVC, LinearSVC
svmClassifier = SVC().fit(X_train,Y_train)
svmClassifier.score(X_test, Y_test)


# In[176]:


from sklearn.ensemble import RandomForestClassifier
rfClassifier = RandomForestClassifier(n_estimators=100).fit(X_train, Y_train)
rfClassifier.score(X_test, Y_test)


# In[199]:


from sklearn.model_selection import GridSearchCV

param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
             {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}]

grid_search = GridSearchCV(rfClassifier, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train, Y_train)


# In[190]:


grid_search.best_params_


# In[189]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params) 
    


# In[204]:


X_train_pd = pd.DataFrame(X_train) 
Y_train_pd = pd.DataFrame(Y_train) 
type(Y_train_pd)


# In[203]:


X_train_pd.head()


# In[220]:


from sklearn.model_selection import RandomizedSearchCV
param_dist = {'n_estimators': [4,6,10], 'max_features': [2, 4, 6, 8],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
              
             

rand_search = RandomizedSearchCV(rfClassifier, param_distributions=param_dist, n_iter = 10, cv=5, return_train_score=True,iid = False)
rand_search.fit(X_train, Y_train)


# In[221]:


rand_search.best_params_


# In[171]:


from sklearn.neighbors import KNeighborsClassifier
knClassifier = KNeighborsClassifier().fit(X_train, Y_train)
knClassifier.score(X_test, Y_test)


# In[85]:


transformedtrainData.drop('Survived',axis=1).head()


# In[180]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(svmClassifier,X_train,Y_train,scoring="neg_mean_squared_error", cv=10)
scores


# In[179]:


get_ipython().run_line_magic('pinfo', 'cross_val_score')


# In[86]:


transformedtestData.isnull().sum()

transformedtestData['Fare'] = transformedtestData['Fare'].fillna(transformedtestData['Fare'].mean())


# In[87]:


preds = ClassificationModel.predict(transformedtestData)


# In[89]:


preds.shape


# In[177]:


get_ipython().run_line_magic('pinfo', 'transformedtestData.groupby')

