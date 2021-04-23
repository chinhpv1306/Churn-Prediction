#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


from collections import Counter
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from xgboost import XGBClassifier 

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection  import train_test_split,cross_val_predict
from sklearn.metrics import recall_score, precision_score, confusion_matrix, accuracy_score, classification_report, precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

def ModelClassification(X,y,option):
    if option == 0:
        model = SVC( C=1, probability=True, random_state=42)
        name = 'SVM model'
    elif option == 1:
        model = DecisionTreeClassifier(min_samples_leaf=8, random_state=42)
        name = 'DecisionTree model'
    elif option == 2:
        model = KNeighborsClassifier(n_neighbors=7)
        name = "KNN model"
    elif option == 3:
        model = GaussianNB()
        name = 'NaiveBayes model'
    elif option == 4:
        model = RandomForestClassifier(n_estimators=1000,
                                      bootstrap=True,
                                      random_state=42,
                                      min_samples_split=2)
        name = 'RandomForest model'
    elif option == 5:
        model = XGBClassifier()
        name = 'XGBoot'
    elif option == 6:
        level0 = list()
        level0.append(('knn', KNeighborsClassifier()))
        level0.append(('cart', DecisionTreeClassifier()))
        level0.append(('svm', SVC()))
        level0.append(('bayes', GaussianNB()))
        # define meta learner model
        level1 = DecisionTreeClassifier()
        # define the stacking ensemble
        model = StackingClassifier(estimators=level0, final_estimator=level1, cv=3)
        name = 'StackClassifier'

    y_predict = cross_val_predict(model,X,y,cv=3)
    
    return y_predict, name, model

data_1 = pd.read_csv(r'E:\Tài Liệu\Tài liệu Khóa Luận\Data\ContractData.csv')
data_2 = pd.read_excel(r'C:\Users\Administrator\knime-workspace\01_Training_a_Churn_Predictor\data\CallsData.xls')

data = pd.merge(data_1,data_2, on = ['Phone','Area Code'], how='inner')

data.pop('State')
data.pop('Phone')

y = data['Churn'].values
data.pop('Churn')

X = data.values
X = X[:,:20]

#print(data.info())

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=42)

# print(Counter(y[:100]))
print(data.head(3))



print("Before OverSampling, counts of label '1': {}".format(sum(y==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y==0)))
#print(X_train.shape)

ov = RandomOverSampler(sampling_strategy='minority', random_state=42)
sm = SMOTE(sampling_strategy='minority',random_state=2, k_neighbors=4)

X_sm, y_sm = sm.fit_resample(X, y)
X_ov, y_ov = ov.fit_resample(X, y)

print('After OverSampling, the shape of train_X: {}'.format(X_sm.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_sm.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_sm==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_sm==0)))

for i in range(8):
#    model, name = ModelClassification(X_train, y_train, i)
    
#    y_predic = model.predict(X_test)
#    print (name)
#    print(confusion_matrix(y_test,y_predic))
#    print(classification_report(y_test,y_predic))
    
#    print ('*'*100)
    y_predict, name, model = ModelClassification(X_sm,y_sm,i)
    print (name, 'SMOTE')
    print(confusion_matrix(y_sm,y_predict))
    print(classification_report(y_sm,y_predict))
    print(precision_recall_curve(y_sm,y_predict))
    print ('*'*100)
    y_predict1, name1, model1 = ModelClassification(X,y,i)
    print (name1, 'Imbalance')
#     print(confusion_matrix(y,y_predict1))
#     print(classification_report(y,y_predict1))
#     print ('*'*100)
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




