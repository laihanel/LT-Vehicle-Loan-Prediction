import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn import feature_selection
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv") #uploaded to Google Colab directly

# Looking at the data headers, these values aren't required

#feature to drop here
train = train.drop(['UniqueID', 'supplier_id', 'Current_pincode_ID', 'Date.of.Birth', 'DisbursalDate', 'Employee_code_ID'], axis = 1)

test = test.drop(['UniqueID', 'supplier_id', 'Current_pincode_ID', 'Date.of.Birth', 'DisbursalDate', 'Employee_code_ID'], axis = 1)



print(train.shape)
print(test.shape)


Y = train.iloc[:, -1] #last column is the the prediction in the training set

Y.shape

X = train.drop(['loan_default'], axis = 1)

X.shape

test_X = test.iloc[:,:]

X.sample(3) # Checking whether irrelevant rows are dropped or not

X['Employment.Type'].fillna('Self employed', inplace = True)
test_X['Employment.Type'].fillna('Self employed', inplace = True)

X['Employment.Type'].value_counts()

X['Employment.Type'] = X['Employment.Type'].replace(('Unemployed', 'Salaried', 'Self employed'), (0, 1, 2))
test_X['Employment.Type'] = test_X['Employment.Type'].replace(('Unemployed', 'Salaried', 'Self employed'), (0, 1, 2))

X['Employment.Type'].value_counts() #Converted irrelevant strings to numbers for computations while training

X['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()

X['PERFORM_CNS.SCORE.DESCRIPTION'] = X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('No Bureau History Available',
                                     'Not Scored: Sufficient History Not Available','Not Scored: Not Enough Info available on the customer',
                                     'Not Scored: No Activity seen on the customer (Inactive)',
                                     'Not Scored: No Updates available in last 36 months', 'Not Scored: Only a Guarantor',
                                     'Not Scored: More than 50 active Accounts found'),(0, 0, 0, 0, 0, 0, 0))

X['PERFORM_CNS.SCORE.DESCRIPTION'] = X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('L-Very High Risk', 'M-Very High Risk'), (1, 1))

X['PERFORM_CNS.SCORE.DESCRIPTION'] = X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('J-High Risk', 'K-High Risk'), (2, 2))

X['PERFORM_CNS.SCORE.DESCRIPTION'] = X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('H-Medium Risk', 'I-Medium Risk'), (3, 3))

X['PERFORM_CNS.SCORE.DESCRIPTION'] = X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('E-Low Risk', 'F-Low Risk', 'G-Low Risk'), (4, 4, 4))

X['PERFORM_CNS.SCORE.DESCRIPTION'] = X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('A-Very Low Risk', 'B-Very Low Risk',
                                      'C-Very Low Risk', 'D-Very Low Risk'), (5, 5, 5, 5))

X['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()

test_X['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()

test_X['PERFORM_CNS.SCORE.DESCRIPTION'] = test_X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('No Bureau History Available',
                                     'Not Scored: Sufficient History Not Available','Not Scored: Not Enough Info available on the customer',
                                     'Not Scored: No Activity seen on the customer (Inactive)',
                                     'Not Scored: No Updates available in last 36 months', 'Not Scored: Only a Guarantor',
                                     'Not Scored: More than 50 active Accounts found'),(0, 0, 0, 0, 0, 0, 0))

test_X['PERFORM_CNS.SCORE.DESCRIPTION'] = test_X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('L-Very High Risk', 'M-Very High Risk'), (1, 1))

test_X['PERFORM_CNS.SCORE.DESCRIPTION'] = test_X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('J-High Risk', 'K-High Risk'), (2, 2))

test_X['PERFORM_CNS.SCORE.DESCRIPTION'] = test_X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('H-Medium Risk', 'I-Medium Risk'), (3, 3))

test_X['PERFORM_CNS.SCORE.DESCRIPTION'] = test_X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('E-Low Risk', 'F-Low Risk', 'G-Low Risk'), (4, 4, 4))

test_X['PERFORM_CNS.SCORE.DESCRIPTION'] = test_X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('A-Very Low Risk', 'B-Very Low Risk',
                                      'C-Very Low Risk', 'D-Very Low Risk'), (5, 5, 5, 5))

test_X['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()

import re
def toMonths(str):
  cache = []
  for k in X[str]:
    temp = int(re.split("[yrs mon]+", k)[0]) * 12 + int(re.split("[yrs mon]+", k)[1])
    cache.append(temp)
  return cache

def toMonthstest(str):
  cache = []
  for k in test_X[str]:
    temp = int(re.split("[yrs mon]+", k)[0]) * 12 + int(re.split("[yrs mon]+", k)[1])
    cache.append(temp)
  return cache

X['CREDIT.HISTORY.LENGTH'] = toMonths('CREDIT.HISTORY.LENGTH')
X['CREDIT.HISTORY.LENGTH'][:5]

X['AVERAGE.ACCT.AGE'] = toMonths('AVERAGE.ACCT.AGE')

X['AVERAGE.ACCT.AGE'][:5]

test_X['CREDIT.HISTORY.LENGTH'] = toMonthstest('CREDIT.HISTORY.LENGTH')
test_X['AVERAGE.ACCT.AGE'] = toMonthstest('AVERAGE.ACCT.AGE')
test_X['AVERAGE.ACCT.AGE'][0:5]






from imblearn.over_sampling import SMOTE

oversample = SMOTE()
x_train, y_train = oversample.fit_resample(X, Y.values.ravel())

print(x_train.shape)
print(y_train.shape)


# pca = PCA(n_components=7).fit(X)
# X = pca.fit_transform(X)
# X = pd.DataFrame(X, columns = ['p1','p2','p3','p4','p5','p6','p7'])
# test_df = pd.DataFrame(pca.fit_transform(train.iloc[:, -1]), columns = ['p1','p2','p3','p4','p5','p6','p7'])
# #Plotting the Cumulative Summation of the Explained Variance
# plt.figure(figsize=(15,5))
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Components')
# plt.ylabel('Variance (%)') #for each component
# plt.title('Pulsar Dataset Explained Variance')
# plt.show()

# import numpy as np
from sklearn.model_selection import train_test_split

#splitting training data into train and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)

print(X_train.shape)
print(Y_train.shape)

print(X_valid.shape)
print(Y_valid.shape)
#
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_valid = scalar.transform(X_valid)
test_X = scalar.transform(test_X)


from sklearn.metrics import roc_auc_score

modelXG = DecisionTreeClassifier(max_depth=3,random_state=100,criterion='entropy',min_samples_leaf=5)
modelXG.fit(X_train, Y_train)

Y_predXG = modelXG.predict(X_valid)

print("Train Accuracy: ", modelXG.score(X_train, Y_train))
print("Validation Accuracy: ", modelXG.score(X_valid, Y_valid))

print("AUROC Score of decision = ", roc_auc_score(Y_valid, Y_predXG))

from sklearn.ensemble import RandomForestClassifier

modelRF = RandomForestClassifier(max_depth=3,random_state=500)
modelRF.fit(X_train, Y_train)

Y_predRF = modelRF.predict(X_valid)

print("Train Accuracy: ", modelRF.score(X_train, Y_train))
print("Validation Accuracy: ", modelRF.score(X_valid, Y_valid))

print("AUROC Score of Random Forest = ", roc_auc_score(Y_valid, Y_predRF))



modelAB = LogisticRegression()
modelAB.fit(X_train, Y_train)

Y_predAB = modelAB.predict(X_valid)

print("Train Accuracy: ", modelAB.score(X_train, Y_train))
print("Validation Accuracy: ", modelAB.score(X_valid, Y_valid))

print("AUROC Score of logistic = ", roc_auc_score(Y_valid, Y_predAB))

from sklearn.ensemble import GradientBoostingClassifier

modelGB = GradientBoostingClassifier()
modelGB.fit(X_train, Y_train)

Y_predGB = modelGB.predict(X_valid)

print("Training Accuracy: ", modelGB.score(X_train, Y_train))
print('Testing Accuarcy: ', modelGB.score(X_valid, Y_valid))

print("AUROC Score of Gradient Boosting = ", roc_auc_score(Y_valid, Y_predGB))


test_Y_RF = modelRF.predict(test_X)
test_Y_XG = modelXG.predict(test_X)
test_Y_AB = modelAB.predict(test_X)
test_Y_GB = modelGB.predict(test_X)
test_Y_pred = []
#
for i in range(len(test_Y_RF)):
  k = 0.25 * test_Y_RF[i] + 0.175 * test_Y_GB[i] + 0.125 * test_Y_XG[i] + 0.1 * test_Y_AB[i] # weighted averaging
  test_Y_pred.append(k)


