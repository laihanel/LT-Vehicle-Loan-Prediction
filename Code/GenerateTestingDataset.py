import pandas as pd
import xlwt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

'''
############################# Final Result ##############################
################################ loading data ################################
'''
df_original = pd.read_csv(r"lt-vehicle-loan-default-prediction\final_train.csv")
df_original.shape
df_original.info()
'''
################################ data cleaning ################################
'''
df = df_original.drop(['loan_default'], axis=1)
y = df_original['loan_default']

sm = SMOTE(random_state=0)
df, y = sm.fit_resample(df, y)

'''
################################ Classification ################################
'''
F1 = []
model_names =[]
scalar = StandardScaler()

X_train_std = scalar.fit_transform(df) # normalizing the features
df_temp = pd.DataFrame(X_train_std)
df_temp.columns = df.columns
y = pd.DataFrame({'loan_default': y})
X_train, X_test, y_train, y_test = train_test_split(df_temp, y, test_size=0.3, random_state=1)

testing = pd.concat([X_test, y_test], axis=1)
testing.to_csv(r"final_test2.csv", index=False)