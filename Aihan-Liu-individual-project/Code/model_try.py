import pandas as pd
import numpy as np
from scipy.linalg import svd
import random
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,recall_score, precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score


import warnings
warnings.filterwarnings("ignore")

'''
################################ loading data ################################
'''
import os
# os.chdir(r'D:\GWU\Aihan\DATS 6103 Data Mining\Quiz')
os.chdir(r'D:\GWU\Aihan\DATS 6103 Data Mining\Final Project\Code\lt-vehicle-loan-default-prediction')
# read csv
df_original = pd.read_csv(r"final_train.csv")
df_original.shape
df_original.info()


'''
################################ data cleaning ################################
'''
# ## null value check
df_original.isnull().sum()
ds = df_original.dropna()
# print("The total number of data-points after removing the rows with missing values are:", len(df))
#
# ## Checking for the duplicates
ds.duplicated().sum()

# df = ds.drop(['comfortability'], axis=1)
df = ds.drop(['loan_default'], axis=1)
y = ds['loan_default']

sm = SMOTE(random_state=0)
df, y = sm.fit_resample(df, y)

'''
################################ Classification ################################
'''
F1 = []
model_names =[]

categorical = ['Employment.Type', 'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH',
               'Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag', 'Passport_flag']
numerical = ['disbursed_amount', 'asset_cost', 'ltv', 'PERFORM_CNS.SCORE', 'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS',
             'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT', 'PRI.DISBURSED.AMOUNT',
             'SEC.NO.OF.ACCTS','SEC.ACTIVE.ACCTS', 'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE',
             'SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT','SEC.INSTAL.AMT',
             'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS', 'NO.OF_INQUIRIES',
             'Age', 'Disbursal_months']


scalar = StandardScaler()

X_train_std = scalar.fit_transform(df) # normalizing the features
df_temp = pd.DataFrame(X_train_std)
df_temp.columns = df.columns
y = pd.DataFrame({'loan_default': y})
X_train, X_test, y_train, y_test = train_test_split(df_temp, y, test_size=0.3, random_state=1)

testing = pd.concat([X_test, y_test], axis=1)
testing.to_csv(r"final_test2.csv", index=False)

'''
################## Baseline Model + scale data ######################
'''
'''
######## Logistic - scale#############
'''
lr = LogisticRegression(solver='liblinear')

lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

accuracy_testing=accuracy_score(y_test,y_pred_lr)
f1_score_testing = f1_score(y_test,y_pred_lr)
precision_score_testing = precision_score(y_test,y_pred_lr)
recall_score_testing = recall_score(y_test,y_pred_lr)

print('###############Logistic Regression')
print("Accuracy")
print("Testing")
print(accuracy_testing)

print("F1_score")
print("Testing")
print(f1_score_testing)

print("Precision_score")
print("Testing")
print(precision_score_testing)

print("Recall")
print("Testing")
print(recall_score_testing)

import pickle
# filename = 'lr_finalized_model2.sav'
# pickle.dump(lr, open(filename, 'wb'))

filename2 = 'lr_finalized_model2.sav'
clf_entropy = pickle.load(open(filename2, 'rb'))
y_pred_entropy = clf_entropy.predict(X_test)
accuracy_score = accuracy_score(y_test, y_pred_entropy)
f1_score= f1_score(y_test,y_pred_entropy)
precision = precision_score(y_test,y_pred_entropy)
recall = recall_score(y_test,y_pred_entropy)
print("Accuracy")
print("Accuracy: ", accuracy_score)
print("Accuracy: ", precision)
print("Accuracy: ",  recall)
print("Accuracy: ",  f1_score)

dt = DecisionTreeClassifier(max_depth=5,min_samples_leaf=0.01,criterion='gini',class_weight='balanced',random_state=123)

dt.fit(X_train, y_train)
y_pred_lr = dt.predict(X_test)

accuracy_testing=accuracy_score(y_test,y_pred_lr)
f1_score_testing = f1_score(y_test,y_pred_lr)
precision_score_testing = precision_score(y_test,y_pred_lr)
recall_score_testing = recall_score(y_test,y_pred_lr)

print('#################Decision Tree')
print("Accuracy")
# print("Testing")
print(accuracy_testing)

print("F1_score")
# print("Testing")
print(f1_score_testing)

print("Precision_score")
# print("Testing")
print(precision_score_testing)

print("Recall")
# print("Testing")
print(recall_score_testing)

import pickle
filename = 'dt_finalized_model2.sav'
pickle.dump(dt, open(filename, 'wb'))


rf = RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=0.01,class_weight='balanced',random_state=123)

rf.fit(X_train, y_train)
y_pred_lr = rf.predict(X_test)

accuracy_testing=accuracy_score(y_test,y_pred_lr)
f1_score_testing = f1_score(y_test,y_pred_lr)
precision_score_testing = precision_score(y_test,y_pred_lr)
recall_score_testing = recall_score(y_test,y_pred_lr)
print('###############Random Forest')
print("Accuracy")
# print("Testing")
print(accuracy_testing)

print("F1_score")
# print("Testing")
print(f1_score_testing)

print("Precision_score")
# print("Testing")
print(precision_score_testing)

print("Recall")
# print("Testing")
print(recall_score_testing)


filename = 'rf_finalized_model2.sav'
pickle.dump(rf, open(filename, 'wb'))

from sklearn.ensemble import GradientBoostingClassifier
modelGB = GradientBoostingClassifier()

modelGB.fit(X_train, y_train)
y_pred_lr = modelGB.predict(X_test)

accuracy_testing=accuracy_score(y_test,y_pred_lr)
f1_score_testing = f1_score(y_test,y_pred_lr)
precision_score_testing = precision_score(y_test,y_pred_lr)
recall_score_testing = recall_score(y_test,y_pred_lr)

print('######################Gradient Boosting')
print("Accuracy")
# print("Testing")
print(accuracy_testing)

print("F1_score")
print(f1_score_testing)

print("Precision_score")
# print("Testing")
print(precision_score_testing)

print("Recall")
# print("Testing")
print(recall_score_testing)

filename = 'gb_finalized_model2.sav'
pickle.dump(modelGB, open(filename, 'wb'))