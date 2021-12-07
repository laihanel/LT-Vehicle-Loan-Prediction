import pandas as pd
import xlwt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
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
import random
from sklearn.model_selection import train_test_split

# '''
# train = pd.read_csv("train.csv")
# test  = pd.read_csv("test.csv") #uploaded to Google Colab directly
#
# # Looking at the data headers, these values aren't required
#
# #feature to drop here
# train = train.drop(['UniqueID', 'supplier_id', 'Current_pincode_ID', 'Date.of.Birth', 'DisbursalDate', 'Employee_code_ID'], axis = 1)
#
# test = test.drop(['UniqueID', 'supplier_id', 'Current_pincode_ID', 'Date.of.Birth', 'DisbursalDate', 'Employee_code_ID'], axis = 1)
#
# print(train.shape)
# print(test.shape)
#
# Y = train.iloc[:, -1] #last column is the the prediction in the training set
#
# Y.shape
#
# X = train.drop(['loan_default'], axis = 1)
#
# X.shape
#
# test_X = test.iloc[:,:]
#
# X.sample(3) # Checking whether irrelevant rows are dropped or not
#
# X['Employment.Type'].fillna('Self employed', inplace = True)
# test_X['Employment.Type'].fillna('Self employed', inplace = True)
#
# X['Employment.Type'].value_counts()
#
# X['Employment.Type'] = X['Employment.Type'].replace(('Unemployed', 'Salaried', 'Self employed'), (0, 1, 2))
# test_X['Employment.Type'] = test_X['Employment.Type'].replace(('Unemployed', 'Salaried', 'Self employed'), (0, 1, 2))
#
# X['Employment.Type'].value_counts() #Converted irrelevant strings to numbers for computations while training
#
# X['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()
#
# X['PERFORM_CNS.SCORE.DESCRIPTION'] = X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('No Bureau History Available',
#                                      'Not Scored: Sufficient History Not Available','Not Scored: Not Enough Info available on the customer',
#                                      'Not Scored: No Activity seen on the customer (Inactive)',
#                                      'Not Scored: No Updates available in last 36 months', 'Not Scored: Only a Guarantor',
#                                      'Not Scored: More than 50 active Accounts found'),(0, 0, 0, 0, 0, 0, 0))
#
# X['PERFORM_CNS.SCORE.DESCRIPTION'] = X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('L-Very High Risk', 'M-Very High Risk'), (1, 1))
#
# X['PERFORM_CNS.SCORE.DESCRIPTION'] = X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('J-High Risk', 'K-High Risk'), (2, 2))
#
# X['PERFORM_CNS.SCORE.DESCRIPTION'] = X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('H-Medium Risk', 'I-Medium Risk'), (3, 3))
#
# X['PERFORM_CNS.SCORE.DESCRIPTION'] = X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('E-Low Risk', 'F-Low Risk', 'G-Low Risk'), (4, 4, 4))
#
# X['PERFORM_CNS.SCORE.DESCRIPTION'] = X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('A-Very Low Risk', 'B-Very Low Risk',
#                                       'C-Very Low Risk', 'D-Very Low Risk'), (5, 5, 5, 5))
#
# X['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()
#
# test_X['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()
#
# test_X['PERFORM_CNS.SCORE.DESCRIPTION'] = test_X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('No Bureau History Available',
#                                      'Not Scored: Sufficient History Not Available','Not Scored: Not Enough Info available on the customer',
#                                      'Not Scored: No Activity seen on the customer (Inactive)',
#                                      'Not Scored: No Updates available in last 36 months', 'Not Scored: Only a Guarantor',
#                                      'Not Scored: More than 50 active Accounts found'),(0, 0, 0, 0, 0, 0, 0))
#
# test_X['PERFORM_CNS.SCORE.DESCRIPTION'] = test_X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('L-Very High Risk', 'M-Very High Risk'), (1, 1))
#
# test_X['PERFORM_CNS.SCORE.DESCRIPTION'] = test_X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('J-High Risk', 'K-High Risk'), (2, 2))
#
# test_X['PERFORM_CNS.SCORE.DESCRIPTION'] = test_X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('H-Medium Risk', 'I-Medium Risk'), (3, 3))
#
# test_X['PERFORM_CNS.SCORE.DESCRIPTION'] = test_X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('E-Low Risk', 'F-Low Risk', 'G-Low Risk'), (4, 4, 4))
#
# test_X['PERFORM_CNS.SCORE.DESCRIPTION'] = test_X['PERFORM_CNS.SCORE.DESCRIPTION'].replace(('A-Very Low Risk', 'B-Very Low Risk',
#                                       'C-Very Low Risk', 'D-Very Low Risk'), (5, 5, 5, 5))
#
# test_X['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()
#
# import re
# def toMonths(str):
#   cache = []
#   for k in X[str]:
#     temp = int(re.split("[yrs mon]+", k)[0]) * 12 + int(re.split("[yrs mon]+", k)[1])
#     cache.append(temp)
#   return cache
#
# def toMonthstest(str):
#   cache = []
#   for k in test_X[str]:
#     temp = int(re.split("[yrs mon]+", k)[0]) * 12 + int(re.split("[yrs mon]+", k)[1])
#     cache.append(temp)
#   return cache
#
# X['CREDIT.HISTORY.LENGTH'] = toMonths('CREDIT.HISTORY.LENGTH')
# X['CREDIT.HISTORY.LENGTH'][:5]
#
# X['AVERAGE.ACCT.AGE'] = toMonths('AVERAGE.ACCT.AGE')
#
# X['AVERAGE.ACCT.AGE'][:5]
#
# test_X['CREDIT.HISTORY.LENGTH'] = toMonthstest('CREDIT.HISTORY.LENGTH')
# test_X['AVERAGE.ACCT.AGE'] = toMonthstest('AVERAGE.ACCT.AGE')
# test_X['AVERAGE.ACCT.AGE'][0:5] '''


'''
############################# Final Result ##############################
################################ loading data ################################
'''
import os
# os.chdir(r'D:\GWU\Aihan\DATS 6103 Data Mining\Final Project\Code\lt-vehicle-loan-default-prediction')
# read csv
df_original = pd.read_csv(r"lt-vehicle-loan-default-prediction\final_train.csv")
df_original.shape
df_original.info()

'''
################################ data cleaning ################################
'''
# # ## null value check
# df_original.isnull().sum()
# ds = df_original.dropna()
# # print("The total number of data-points after removing the rows with missing values are:", len(df))
# #
# # ## Checking for the duplicates
# ds.duplicated().sum()

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
testing.to_csv(r"lt-vehicle-loan-default-prediction/final_test2.csv", index=False)

'''
################## Modelling, please comment this part if necessary ######################
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
filename = 'lr_finalized_model2.sav'
pickle.dump(lr, open(filename, 'wb'))

# filename2 = 'lr_finalized_model2.sav'
# clf_entropy = pickle.load(open(filename2, 'rb'))
# y_pred_entropy = clf_entropy.predict(X_test)
# accuracy_score = accuracy_score(y_test, y_pred_entropy)
# f1_score= f1_score(y_test,y_pred_entropy)
# precision = precision_score(y_test,y_pred_entropy)
# recall = recall_score(y_test,y_pred_entropy)
#
# print("Accuracy")
# print(accuracy_score)
# print("F1_score")
# print(f1_score)
# print("Precision_score")
# print(precision)
# print("Recall")
# print(recall)



dt = DecisionTreeClassifier(max_depth=5,min_samples_leaf=0.01,criterion='gini',class_weight='balanced',random_state=123)

dt.fit(X_train, y_train)
y_pred_lr = dt.predict(X_test)

accuracy_testing=accuracy_score(y_test,y_pred_lr)
f1_score_testing = f1_score(y_test,y_pred_lr)
precision_score_testing = precision_score(y_test,y_pred_lr)
recall_score_testing = recall_score(y_test,y_pred_lr)

print('#################Decision Tree')
print("Accuracy")
print(accuracy_testing)

print("F1_score")
print(f1_score_testing)

print("Precision_score")
print(precision_score_testing)

print("Recall")
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
print(accuracy_testing)

print("F1_score")
print(f1_score_testing)

print(precision_score_testing)

print("Recall")
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
print(accuracy_testing)

print("F1_score")
print(f1_score_testing)

print("Precision_score")
print(precision_score_testing)

print("Recall")
print(recall_score_testing)

filename = 'gb_finalized_model2.sav'
pickle.dump(modelGB, open(filename, 'wb'))

'''
train = pd.read_csv(r"lt-vehicle-loan-default-prediction/final_train.csv") # Change this according to your data location

Z1=[12311,12312,12374,12534,12539,12797,13131,13317,13448,13512,13890,13928,13929,13937,13941,13942,13947,13948,13960,13984,13993,14002,14004,14011,14021,14074,14078,14088,14100,14108,14115,14123,14133,14134,14136,14144,14151,14166,14176,14178,14180,14189,14198,14214,14231,14237,14241,14255,14271,14292,14300,14320,14331,14336,14343,14360,14368,14370,14375,14419,14427,14432,14436,14441,14443,14444,14462,14522,14545,14557,14573,14574,14582,14592,14601,14606,14622,14625,14630,14643,14656,14657,14668,14690,14697,14698,14701,14711,14716,14723,14754,14764,14790,14795,14804,14823,14834,14835,14848,14854,14882,14958,14975,14982,15047,15049,15062,15070,15075,15082,15090,15123,15129,15142,15179,15183,15194,15196,15198,15199,15200,15201,15202,15209,15212,15217,15218,15227,15230,15231,15235,15242,15271,15272,15296,15304,15306,15309,15336,15352,15357,15366,15383,15387,15398,15410,15411,15413,15425,15438,15452,15457,15458,15460,15483,15484,15490,15511,15513,15616,15617,15630,15631,15642,15678,15680,15685,15700,15710,15733,15743,15744,15762,15775,15779,15781,15803,15804,15805,15808,15809,15838,15848,15884,15887,15888,15893,15897,15899,15905,15909,15911,15912,15916,15920,15925,15932,15939,15941,15943,15974,15979,15997,16027,16053,16057,16065,16080,16092,16120,16138,16140,16156,16167,16169,16172,16212,16239,16244,16249,16252,16270,16273,16277,16291,16292,16309,16322,16364,16365,16388,16392,16430,16437,16460,16499,16504,16505,16513,16528,16531,16539,16540,16556,16562,16578,16587,16599,16600,16612,16624,16630,16639,16652,16677,16679,16680,16699,16704,16712,16714,16717,16754,16759,16760,16763,16775,16785,16787,16792,16806,16809,16815,16835,16838,16849,16861,16932,16938,16940,16956,16964,16982,17000,17004,17032,17039,17094,17126,17140,17169,17189,17190,17198,17217,17231,17267,17270,17315,17323,17327,17355,17364,17383,17394,17398,17400,17401,17411,17413,17416,17431,17433,17450,17451,17473,17477,17507,17514,17528,17530,17560,17563,17610,17627,17654,17668,17731,17737,17741,17746,17768,17790,17795,17803,17827,17859,17878,17901,17904,17906,17916,17920,17935,17954,17962,17964,17966,17981,18001,18006,18007,18029,18033,18045,18052,18059,18060,18063,18066,18068,18077,18080,18082,18087,18089,18100,18107,18110,18111,18123,18130,18142,18144,18154,18165,18166,18170,18175,18176,18185,18207,18214,18217,18218,18228,18234,18238,18255,18258,18268,18269,18276,18294,18295,18306,18309,18310,18312,18321,18327,18332,18342,18356,18371,18373,18374,18375,18388,18399,18409,18410,18412,18422,18435,18437,18450,18472,18502,18526,18528,18535,18540,18557,18559,18563,18565,18578,18588,18605,18606,18631,18633,18636,18648,18651,18661,18674,18680,18686,18692,18703,18706,18719,18730,18731,18733,18752,18753,18756,18757,18758,18774,18786,18794,18795,18823,18838,19528,20180,20285,20286,20289,20294,20297,20328,20335,20343,20379,20390,20432,20438,20442,20475,20478,20480,20535,20538,20563,20570,20571,20582,20613,20620,20639,20640,20644,20647,20674,20676,20685,20690,20700,20704,20710,20724,20739,20757,20767,20787,20789,20790,20794,20795,20796,20806,20869,20873,20878,20940,20959,20973,20975,20987,21013,21022,21032,21033,21054,21064,21097,21106,21140,21142,21149,21153,21167,21188,21198,21202,21204,21206,21215,21221,21223,21237,21239,21247,21264,21276,21305,21314,21321,21327,21335,21352,21366,21386,21388,21413,21422,21423,21435,21441,21475,21478,21495,21528,21553,21573,21579,21597,21601,21617,21635,21680,21702,21703,21704,21709,21713,21722,21735,21746,21772,21773,21838,21843,21846,21854,21855,21857,21860,21872,21884,21899,21904,21910,21932,21946,21949,21952,21955,21958,21963,21966,21976,21987,21991,22002,22003,22015,22030,22032,22045,22056,22059,22070,22079,22084,22090,22123,22124,22125,22163,22173,22200,22221,22222,22239,22243,22275,22276,22285,22329,22335,22350,22358,22369,22388,22398,22420,22421,22430,22451,22452,22454,22460,22493,22499,22504,22515,22516,22522,22524,22537,22547,22554,22561,22562,22565,22568,22570,22572,22574,22585,22594,22595,22598,22607,22610,22611,22628,22629,22632,22633,22634,22635,22637,22638,22647,22656,22658,22664,22671,22681,22685,22694,22699,22702,22703,22709,22713,22716,22719,22720,22727,22731,22732,22740,22744,22745,22746,22748,22752,22753,22761,22762,22763,22765,22769,22770,22772,22773,22774,22777,22778,22782,22786,22787,22789,22796,22798,22802,22806,22808,22811,22812,22815,22831,22839,22844,22850,22853,22856,22862,22865,22869,22870,22871,22872,22873,22875,22877,22879,22888,22889,22890,22891,22892,22898,22900,22901,22905,22907,22917,22919,22922,22924,22928,22933,22938,22942,22943,22945,22949,22951,22952,22954,22955,22956,22972,22979,22980,22985,22986,22989,22992,22996,23000,23001,23003,23004,23005,23006,23007,23009,23013,23014,23022,23024,23025,23027,23028,23037,23038,23041,23042,23043,23046,23048,23049,23051,23053,23057,23063,23069,23070,23075,23083,23085,23090,23099,23104,23112,23114,23115,23122,23125,23127,23129,23130,23132,23133,23134,23136,23141,23146,23147,23148,23152,23153,23158,23160,23163,23166,23169,23170,23175,23178,23180,23183,23185,23192,23195,23197,23199,23200,23202,23203,23205,23209,23210,23212,23213,23216,23217,23222,23227,23228,23232,23233,23237,23239,23240,23243,23244,23246,23247,23248,23253,23263,23266,23267,23268,23273,23275,23276,23282,23291,23294,23295,23296,23299,23301,23303,23304,23308,23309,23313,23314,23316,23317,23318,23323,23328,23329,23330,23331,23336,23345,23349,23350,23351,23355,23360,23371,23372,23377,23383,23385,23386,23387,23389,23390,23393,23397,23400,23404,23405,23406,23408,23410,23411,23413,23414,23415,23417,23421,23422,23423,23425,23426,23428,23430,23431,23434,23436,23438,23440,23441,23442,23443,23445,23446,23449,23459,23460,23463,23466,23469,23470,23471,23474,23478,23483,23486,23488,23490,23499,23504,23506,23507,23508,23513,23514,23517,23520,23522,23525,23529,23532,23539,23542,23544,23546,23549,23551,23552,23555,23557,23558,23563,23565,23566,23567,23568,23569,23570,23571,23572,23573,23575,23582,23583,23587,23588,23592,23594,23600,23607,23613,23616,23617,23619,23620,23622,23624,23629,23630,23634,23638,23639,23640,23649,23650,23658,23660,23670,23671,23678,23683,23686,23687,23689,23698,23699,23701,23704,23706,23707,23712,23713,23716,23717,23718,23719,23720,23736,23738,23740,23744,23747,23748,23757,23760,23764,23766,23768,23770,23772,23774,23775,23778,23780,23784,23785,23786,23787,23790,23791,23793,23795,23797,23799,23800,23801,23805,23806,23810,23811,23813,23815,23818,23820,23822,23824,23825,23828,23831,23833,23837,23851,23852,23857,23858,23863,23867,23868,23869,23871,23874,23876,23877,23878,23882,23885,23887,23888,23894,23903,23904,23906,23907,23908,23916,23923,23928,23930,23934,23937,23943,23944,23945,23954,23956,23959,23960,23963,23964,23965,23972,23973,23977,23980,23982,23984,23985,23986,23995,24001,24005,24007,24014,24021,24024,24025,24029,24031,24036,24038,24040,24041,24052,24053,24054,24055,24061,24062,24063,24064,24065,24069,24070,24072,24076,24077,24078,24080,24084,24089,24099,24100,24101,24105,24106,24115,24120,24121,24123,24126,24127,24138,24141,24145,24147,24148,24153,24159,24163,24170,24175,24176,24177,24191,24192,24198,24199,24206,24210,24217,24218,24219,24223,24224,24229,24231,24234,24235,24242,24243,24246,24248,24253,24254,24256,24257,24258,24268,24271,24274,24275,24282,24283,24284,24285,24286,24287,24291,24293,24295,24297,24300,24301,24304,24305,24306,24309,24311,24314,24315,24325,24329,24330,24332,24338,24339,24342,24343,24345,24348,24349,24353,24354,24356,24360,24361,24374,24375,24378,24379,24385,24386,24389,24393,24395,24396,24402,24414,24416,24419,24421,24425,24426,24429,24430,24432,24434,24435,24439,24441,24447,24449,24457,24460,24461,24462,24463,24464,24469,24471,24473,24474,24476,24479,24484,24485,24487,24492,24496,24498,24500,24508,24511,24515,24518,24520,24522,24525,24529,24530,24531,24533,24542,24546,24554,24559,24561,24564,24565,24567,24568,24570,24572,24573,24576,24577,24581,24582,24592,24593,24600,24604,24606,24612,24615,24619,24624,24625,24627,24628,24629,24632,24634,24636,24638,24645,24646,24648,24650,24653,24656,24659,24660,24662,24664,24666,24670,24673,24676,24677,24684,24685,24687,24694,24701,24704,24706,24710,24712,24718,24719,24720,24736,24739,24740,24743,24744,24745,24749,24754,24756,24761,24762,24769,24778,24780,24781,24785,24787,24794,24799,24802]
Z2=[12441,12456,12500,12842,12878,13295,13612,13895,13913,13926,13931,13968,13971,13972,13975,13978,13988,13990,13992,14015,14028,14046,14070,14081,14094,14095,14112,14114,14124,14132,14142,14143,14145,14147,14148,14152,14158,14159,14181,14203,14213,14234,14293,14305,14313,14347,14356,14364,14387,14430,14458,14464,14470,14518,14539,14541,14558,14591,14600,14602,14610,14614,14621,14628,14636,14710,14734,14761,14770,14791,14810,14811,14851,14871,14878,14927,14929,14930,14987,15011,15068,15077,15078,15091,15097,15105,15116,15117,15118,15141,15152,15163,15222,15299,15300,15312,15354,15359,15364,15369,15372,15406,15449,15475,15503,15523,15529,15532,15564,15614,15636,15651,15661,15662,15663,15686,15693,15694,15705,15706,15732,15770,15772,15777,15792,15793,15798,15828,15852,15857,15861,15879,15889,15894,15898,15900,15907,15919,15996,16010,16016,16023,16031,16051,16052,16067,16103,16146,16159,16166,16171,16190,16196,16199,16204,16219,16234,16251,16264,16310,16335,16344,16350,16419,16428,16431,16445,16450,16461,16471,16483,16486,16487,16535,16555,16565,16573,16576,16583,16594,16596,16603,16605,16609,16610,16611,16613,16616,16620,16631,16633,16637,16638,16640,16646,16647,16676,16682,16686,16689,16691,16692,16693,16694,16700,16721,16730,16736,16739,16742,16764,16776,16794,16798,16801,16803,16805,16807,16817,16833,16845,16846,16848,16865,16866,16870,16873,16920,16985,17011,17014,17020,17022,17023,17038,17049,17058,17066,17075,17078,17113,17116,17119,17138,17139,17141,17142,17143,17145,17170,17175,17181,17207,17223,17224,17227,17242,17255,17272,17285,17292,17308,17310,17318,17339,17389,17399,17408,17412,17417,17466,17467,17468,17485,17501,17502,17509,17551,17628,17635,17684,17687,17694,17705,17725,17732,17740,17742,17752,17771,17783,17786,17789,17804,17809,17831,17847,17896,17921,17980,17995,18002,18025,18039,18050,18062,18065,18072,18103,18105,18112,18125,18129,18136,18149,18150,18153,18155,18169,18174,18178,18180,18219,18223,18231,18233,18252,18257,18264,18265,18271,18274,18291,18296,18302,18315,18317,18326,18344,18348,18354,18384,18385,18387,18392,18393,18396,18397,18398,18400,18401,18404,18408,18415,18429,18441,18449,18451,18459,18461,18466,18471,18473,18486,18489,18501,18505,18508,18510,18519,18520,18532,18534,18546,18549,18551,18564,18599,18625,18643,18652,18654,18655,18658,18663,18675,18676,18696,18702,18704,18714,18715,18722,18729,18732,18804,18822,19525,20292,20316,20321,20333,20348,20391,20408,20441,20470,20473,20486,20490,20495,20498,20512,20514,20518,20520,20547,20576,20619,20672,20673,20675,20677,20691,20706,20716,20786,20804,20858,20868,20891,20912,20913,20933,20981,20983,20984,20996,21003,21019,21043,21044,21046,21074,21081,21082,21091,21108,21124,21156,21177,21180,21216,21240,21251,21273,21282,21308,21312,21333,21344,21347,21426,21429,21445,21465,21479,21498,21514,21531,21537,21556,21557,21592,21602,21619,21626,21668,21669,21674,21684,21715,21716,21721,21739,21745,21750,21752,21758,21788,21795,21796,21824,21836,21866,21879,21880,21883,21891,21895,21908,21909,21911,21977,21980,21983,21988,21995,22004,22012,22017,22023,22026,22031,22073,22076,22085,22111,22137,22168,22177,22186,22192,22198,22229,22264,22278,22286,22288,22289,22293,22304,22315,22317,22320,22322,22333,22354,22401,22408,22412,22417,22418,22438,22439,22440,22441,22442,22446,22447,22450,22457,22463,22480,22482,22483,22486,22489,22500,22506,22521,22523,22529,22535,22536,22542,22558,22567,22569,22593,22599,22601,22604,22609,22631,22639,22644,22645,22657,22668,22676,22684,22689,22692,22697,22705,22707,22708,22712,22721,22724,22728,22729,22736,22739,22742,22754,22755,22756,22764,22766,22768,22775,22776,22781,22785,22805,22807,22809,22810,22817,22829,22830,22832,22834,22836,22841,22842,22847,22851,22852,22857,22858,22860,22861,22866,22867,22874,22881,22882,22885,22887,22895,22902,22908,22909,22910,22913,22914,22915,22918,22923,22927,22929,22930,22932,22934,22935,22936,22937,22939,22944,22957,22958,22959,22960,22961,22964,22967,22969,22971,22973,22974,22976,22978,22990,22991,22993,22995,22998,22999,23002,23008,23010,23012,23015,23019,23020,23021,23026,23029,23030,23032,23033,23036,23039,23044,23045,23056,23062,23065,23066,23071,23074,23076,23079,23080,23082,23084,23089,23091,23092,23094,23098,23103,23106,23107,23108,23113,23116,23120,23124,23126,23131,23135,23137,23138,23140,23142,23143,23144,23145,23149,23151,23155,23156,23161,23162,23167,23168,23172,23173,23174,23182,23191,23193,23194,23196,23201,23207,23208,23214,23219,23221,23224,23225,23238,23251,23252,23254,23257,23270,23274,23277,23279,23280,23283,23284,23288,23289,23292,23293,23302,23305,23306,23310,23311,23319,23320,23321,23322,23325,23332,23337,23339,23341,23342,23343,23344,23348,23352,23353,23354,23357,23359,23362,23364,23365,23373,23374,23376,23378,23381,23382,23384,23391,23394,23396,23399,23401,23403,23407,23409,23412,23416,23418,23427,23432,23435,23447,23448,23450,23451,23452,23453,23455,23461,23462,23464,23465,23467,23468,23472,23473,23475,23476,23477,23479,23480,23482,23484,23489,23491,23492,23494,23495,23496,23498,23501,23502,23503,23505,23509,23510,23511,23512,23515,23516,23519,23524,23526,23530,23531,23533,23535,23536,23537,23538,23540,23543,23545,23547,23550,23556,23564,23574,23576,23579,23584,23585,23589,23593,23597,23599,23601,23603,23605,23606,23609,23610,23611,23614,23618,23621,23623,23626,23631,23641,23643,23645,23647,23651,23652,23654,23657,23659,23663,23667,23668,23669,23672,23674,23675,23676,23682,23684,23688,23690,23691,23692,23695,23697,23703,23709,23710,23711,23722,23726,23727,23728,23730,23731,23737,23742,23745,23746,23750,23759,23762,23763,23765,23767,23769,23771,23777,23779,23781,23782,23792,23796,23798,23807,23808,23812,23819,23821,23823,23826,23827,23836,23849,23853,23854,23855,23861,23862,23865,23866,23870,23872,23873,23879,23880,23881,23883,23889,23890,23891,23892,23893,23896,23898,23899,23901,23902,23905,23909,23912,23913,23914,23915,23917,23918,23919,23921,23922,23924,23927,23929,23933,23936,23938,23939,23940,23941,23942,23946,23949,23950,23952,23961,23962,23966,23967,23970,23971,23974,23975,23976,23978,23979,23981,23987,23988,23989,23991,23993,23998,23999,24000,24002,24003,24004,24008,24009,24012,24013,24015,24018,24019,24027,24028,24030,24032,24033,24034,24035,24037,24039,24042,24043,24044,24046,24047,24048,24050,24059,24060,24066,24067,24073,24074,24082,24083,24085,24086,24087,24088,24091,24092,24093,24096,24098,24102,24108,24110,24113,24117,24119,24122,24124,24129,24130,24131,24132,24139,24140,24144,24149,24150,24151,24155,24157,24160,24162,24165,24168,24169,24171,24172,24173,24174,24178,24180,24182,24184,24185,24186,24187,24188,24194,24195,24197,24201,24203,24204,24208,24209,24211,24214,24215,24220,24221,24225,24226,24227,24230,24232,24236,24241,24244,24247,24249,24251,24255,24259,24260,24261,24262,24266,24267,24269,24270,24273,24276,24279,24280,24281,24290,24312,24319,24321,24323,24324,24333,24334,24341,24344,24347,24364,24365,24366,24367,24369,24370,24372,24373,24376,24377,24380,24382,24383,24384,24387,24388,24391,24392,24394,24397,24398,24400,24401,24405,24406,24408,24410,24411,24415,24418,24428,24431,24433,24442,24444,24445,24446,24448,24450,24452,24453,24455,24456,24458,24465,24466,24467,24468,24477,24488,24490,24491,24494,24495,24499,24504,24506,24507,24512,24516,24517,24521,24523,24526,24532,24536,24545,24547,24549,24551,24556,24560,24562,24563,24571,24574,24585,24586,24594,24602,24610,24620,24622,24630,24640,24641,24642,24644,24647,24649,24651,24654,24657,24667,24668,24672,24675,24683,24686,24688,24689,24691,24693,24707,24709,24714,24721,24723,24727,24728,24729,24732,24751,24758,24759,24760,24763,24767,24770,24773,24777]
Z3=[10524,13924,14005,14188,14383,14617,14847,15034,15408,15442,15699,16294,16332,16340,16498,16747,16756,17097,17284,17436,17448,17565,17853,17961,18049,18083,18106,18132,18161,18182,18222,18226,18300,18423,18469,18556,18792,18802,18829,20302,20347,20569,20597,20684,20687,21242,21345,21466,21638,21832,21873,21968,22127,22321,22374,22586,22600,22655,22826,22828,22884,22894,22912,22963,22994,23016,23017,23034,23047,23061,23086,23087,23093,23100,23102,23118,23119,23123,23150,23187,23211,23220,23234,23285,23334,23369,23388,23395,23424,23429,23437,23485,23487,23497,23500,23518,23578,23580,23586,23598,23642,23648,23656,23702,23705,23714,23721,23723,23733,23743,23752,23753,23755,23758,23773,23776,23788,23816,23829,23830,23834,23835,23886,23895,23910,23947,23957,23969,23983,23994,23997,24006,24017,24022,24045,24051,24068,24071,24075,24094,24111,24118,24143,24161,24179,24189,24190,24202,24205,24216,24228,24233,24245,24250,24264,24265,24272,24292,24296,24307,24310,24313,24352,24355,24357,24363,24381,24399,24407,24409,24412,24423,24424,24427,24438,24472,24483,24486,24509,24510,24513,24514,24535,24538,24566,24578,24584,24587,24588,24591,24603,24605,24611,24621,24626,24631,24635,24637,24663,24669,24674,24692,24696,24698,24708,24711,24747,24752,24753,24766,24779,24797,24803]
Z4=[13909,15704,15921,16225,16226,16644,20972,22028,22880,23097,23356,23591,23696,23708,23817,23951,24023,24112,24166,24200,24303,24308,24317,24337,24534,24541,24569,24579,24607,24623,24639,24697,24699,24716,24737]
Z5=[15045,15186,16788,17183,17228,17865,18079,18099,18102,18513,20315,20763,20931,20943,21511,21847,21981,22474,22552,22630,22751,22840,22845,23088,23111,23171,23188,23189,23541,23635,23661,23685,23741,23802,23932,24109,24222,24252,24294,24443,24552,24555,24598,24599,24616,24679,24700,24713,24715,24724,24742,24789,24790,24793]
train.loc[train.supplier_id.isin(Z1),'cat_supplier_id']='G1'
train.loc[train.supplier_id.isin(Z2),'cat_supplier_id']='G2'
train.loc[train.supplier_id.isin(Z3),'cat_supplier_id']='G3'
train.loc[train.supplier_id.isin(Z4),'cat_supplier_id']='G4'
train.loc[train.supplier_id.isin(Z5),'cat_supplier_id']='G5'

#train.isnull().sum()
train = train.drop('supplier_id', axis=1)

train["branch_id"].replace({152:'A',8:'A',17:'A',
                            1:'B',100:'B',19:'B',104:'B',142:'B',162:'B',
                            66:'C',3:'C',15:'C',135:'C',34:'C',42:'C',103:'C',82:'C',2:'C',63:'C',160:'C',70:'C',
                            67:'D',9:'D',77:'D',121:'D',207:'D',20:'D',48:'D',84:'D',68:'D',257:'D',130:'D',73:'D',7:'D',258:'D',138:'D',72:'D',11:'D',62:'D',
                            43:'E',29:'E',250:'E',255:'E',159:'E',136:'E',202:'E',79:'E',5:'E',261:'E',165:'E',249:'E',61:'E',13:'E',
                            259:'F',18:'F',101:'F',111:'F',76:'F',64:'F',217:'F',248:'F',69:'F',
                            85:'G',14:'G',120:'G',74:'G',147:'G',260:'G',10:'G',65:'G',105:'G',158:'G',117:'G',35:'G',153:'G',
                            146:'H',16:'H',78:'H',36:'H',97:'H',254:'H',251:'H'},inplace=True)

train["manufacturer_id"].replace({152:'A',156:'A',145:'A',
                                  86:'B',51:'B',
                                  67:'C',49:'C',120:'C',
                                  45:'D',48:'D',153:'D'},inplace=True)

train["State_ID"].replace({22:'A',20:'A',10:'A',1:'A',
                           16:'B',19:'B',3:'B',
                           21:'C',5:'C',7:'C',11:'C',6:'C',
                           4:'D',15:'D',9:'D',
                           18:'E',8:'E',17:'E',12:'E',
                           2:'F',14:'F',13:'F'},inplace=True)

# train["Age"].replace({23:'[23,29]',24:'[23,29]',25:'[23,29]',26:'[23,29]',27:'[23,29]',28:'[23,29]',29:'[23,29]',
#                       30:'[30,36]',31:'[30,36]',32:'[30,36]',33:'[30,36]',34:'[30,36]',35:'[30,36]',36:'[30,36]',
#                       37:'[37,45]',38:'[37,45]',39:'[37,45]',40:'[37,45]',41:'[37,45]',42:'[37,45]',43:'[37,45]',44:'[37,45]',45:'[37,45]',
#                       46:'[46,57]',47:'[46,57]',48:'[46,57]',49:'[46,57]',50:'[46,57]',51:'[46,57]',52:'[46,57]',53:'[46,57]',54:'[46,57]',55:'[46,57]',56:'[46,57]',57:'[46,57]',
#                       58:'[58,67]',59:'[58,67]',60:'[58,67]',61:'[58,67]',62:'[58,67]',63:'[58,67]',64:'[58,67]',65:'[58,67]',66:'[58,67]',67:'[58,67]'},inplace=True)


#train['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()

# train.info()
# train=pd.get_dummies(train, columns=["State_ID",
#                                       "manufacturer_id",
#                                       "branch_id",
#                                       "cat_supplier_id"])

train.to_csv(r"lt-vehicle-loan-default-prediction/final_train1.csv",index=False)
train = pd.read_csv(r"lt-vehicle-loan-default-prediction/final_train1.csv") # Change this according to your data location


def split_data(inpath, target_name, test_size):
  df = pd.read_csv(inpath)
  y = df[target_name]
  # x = df1.loc[:,df1.columns!='loan_default']
  x = df.drop(target_name, axis=1)
  class_le = LabelEncoder()
  for le_val in ["State_ID", "manufacturer_id", "branch_id", "cat_supplier_id"]:
      x[le_val] = class_le.fit_transform(x[le_val])
  y = class_le.fit_transform(y)
  # set a random seed for the data, so that we could get the same train and test set
  random.seed(12345)
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1, stratify=y)

  # With stratify, we make sure to have the same default rate for both df
  return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = split_data(r"lt-vehicle-loan-default-prediction/final_train1.csv", 'loan_default', 0.3)
y_test2 = pd.DataFrame({'loan_default':y_test})
testing = pd.concat([X_test.reset_index(), y_test2], axis=1)
testing.to_csv(r"lt-vehicle-loan-default-prediction/final_test.csv",index=False)


# # For Imbalance Classification
# from imblearn.over_sampling import SMOTE
#
# oversample = SMOTE()
# X_train, y_train = oversample.fit_resample(X_train, y_train.values.ravel())

#print(X_train.shape)
#print(y_train.shape)


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

#splitting training data into train and validation set
# X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)
#
# print(X_train.shape)
# print(Y_train.shape)
#
# print(X_valid.shape)
# print(Y_valid.shape)
#
'''
'''
#### save the data for the feature selection
### This data will be used at the GUI part


#%----------------- STANDARIZATION
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train_std = scalar.fit_transform(X_train) # normalizing the features
X_train_std=pd.DataFrame(X_train_std,columns=X_train.columns)
#%--------------------- Fetuare Selection
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

estimator = RandomForestClassifier(n_estimators=50,random_state=123,n_jobs=-1)

selector = RFE(estimator, n_features_to_select=35,step=0.05, verbose=1)
selector = selector.fit(X_train_std, y_train)

list_rf=selector.support_

X_train.iloc[:,list_rf].info()

#X_valid = scalar.transform(X_valid)
#test_X = scalar.transform(test_X)


from sklearn.metrics import roc_auc_score

#%----------------- Decision Tree -----------------
dt = DecisionTreeClassifier(max_depth=5,min_samples_leaf=0.01,criterion='gini',class_weight='balanced',random_state=123)
#modelXG.fit(X_train, y_train)

#Y_predXG = modelXG.predict(X_valid)
#print("Train Accuracy: ", modelXG.score(X_train, y_train))
#print("Validation Accuracy: ", modelXG.score(X_valid, Y_valid))
#print("AUROC Score of decision = ", roc_auc_score(Y_valid, Y_predXG))

dt.fit(X_train.iloc[:,list_rf], y_train)

df_training_pred_dt = pd.DataFrame({'actual':y_train,'predicted': dt.predict(X_train.iloc[:,list_rf]),
                                 'Non_Target':dt.predict_proba(X_train.iloc[:,list_rf])[:,0],
                                 'Target':dt.predict_proba(X_train.iloc[:,list_rf])[:,1],
                                })

df_testing_pred_dt = pd.DataFrame({'actual':y_test,'predicted': dt.predict(X_test.iloc[:,list_rf]),
                                'Non_Target':dt.predict_proba(X_test.iloc[:,list_rf])[:,0],
                                'Target':dt.predict_proba(X_test.iloc[:,list_rf])[:,1],
                               })

from sklearn import metrics
import matplotlib.pyplot as plt

accuracy_training_dt=metrics.accuracy_score(df_training_pred_dt.actual,df_training_pred_dt.predicted)
accuracy_testing_dt=metrics.accuracy_score(df_testing_pred_dt.actual,df_testing_pred_dt.predicted)

f1_score_training_dt=metrics.f1_score(df_training_pred_dt.actual,df_training_pred_dt.predicted)
f1_score_testing_dt=metrics.f1_score(df_testing_pred_dt.actual,df_testing_pred_dt.predicted)

auc_score_training_dt = metrics.roc_auc_score(df_training_pred_dt.actual, df_training_pred_dt.predicted)
auc_score_testing_dt = metrics.roc_auc_score(df_testing_pred_dt.actual, df_testing_pred_dt.predicted)

print("Accuracy")
print("Training")
print(accuracy_training_dt)
print("Testing")
print(accuracy_testing_dt)
print("\n")

print("F1_score")
print("Training")
print(f1_score_training_dt)
print("Testing")
print(f1_score_testing_dt)
print("\n")

print("AUC_Score")
print("Training")
print(auc_score_training_dt)
print("Testing")
print(auc_score_testing_dt)

#### To sove the numbers to be used in the pyqt5
import pickle
filename = 'dt_finalized_model.sav'
pickle.dump(dt, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result=loaded_model.score(X_test.iloc[:,list_rf],y_test)
print(result)

# to graph the tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
dot_data = export_graphviz(dt, filled=True, rounded=True, class_names=["No","Yes"], feature_names=X_test.iloc[:,list_rf].columns, out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree_entropy.pdf")

#%----------------- Random Forest -----------------
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=0.01,class_weight='balanced',random_state=123)
#modelRF.fit(X_train, y_train)
#Y_predRF = modelRF.predict(X_valid)

# calculate parameters for logistic Model on Training
rf.fit(X_train.iloc[:,list_rf], y_train)

#print("Train Accuracy: ", modelRF.score(X_train, y_train))
#print("Validation Accuracy: ", modelRF.score(X_valid, Y_valid))
#print("AUROC Score of Random Forest = ", roc_auc_score(Y_valid, Y_predRF))


df_training_pred_rf = pd.DataFrame({'actual':y_train,'predicted': rf.predict(X_train.iloc[:,list_rf]),
                                 'Non_Target':rf.predict_proba(X_train.iloc[:,list_rf])[:,0],
                                 'Target':rf.predict_proba(X_train.iloc[:,list_rf])[:,1],
                                })

df_testing_pred_rf = pd.DataFrame({'actual':y_test,'predicted': rf.predict(X_test.iloc[:,list_rf]),
                                'Non_Target':rf.predict_proba(X_test.iloc[:,list_rf])[:,0],
                                'Target':rf.predict_proba(X_test.iloc[:,list_rf])[:,1],
                               })

from sklearn import metrics
import matplotlib.pyplot as plt

accuracy_training_rf=metrics.accuracy_score(df_training_pred_rf.actual,df_training_pred_rf.predicted)
accuracy_testing_rf=metrics.accuracy_score(df_testing_pred_rf.actual,df_testing_pred_rf.predicted)

f1_score_training_rf=metrics.f1_score(df_training_pred_rf.actual,df_training_pred_rf.predicted)
f1_score_testing_rf=metrics.f1_score(df_testing_pred_rf.actual,df_testing_pred_rf.predicted)

auc_score_training_rf = metrics.roc_auc_score(df_training_pred_rf.actual, df_training_pred_rf.predicted)
auc_score_testing_rf = metrics.roc_auc_score(df_testing_pred_rf.actual, df_testing_pred_rf.predicted)

print("Accuracy")
print("Training")
print(accuracy_training_rf)
print("Testing")
print(accuracy_testing_rf)
print("\n")

print("F1_score")
print("Training")
print(f1_score_training_rf)
print("Testing")
print(f1_score_testing_rf)
print("\n")

print("AUC_Score")
print("Training")
print(auc_score_training_rf)
print("Testing")
print(auc_score_testing_rf)

feat_importances = pd.Series(rf.feature_importances_, index=X_train.iloc[:,list_rf].columns)
feat_importances.nlargest(45).sort_values().plot(kind='barh')
plt.show()

#### To sove the numbers to be used in the pyqt5
import pickle
filename = 'rf_finalized_model.sav'
pickle.dump(rf, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result=loaded_model.score(X_test.iloc[:,list_rf],y_test)
print(result)

#%----------------- Logistic Regression -----------------

from sklearn.linear_model import LogisticRegression

# creating the classifier object
lr = LogisticRegression(penalty='l2',fit_intercept=True,random_state=123,class_weight='balanced')

# calculate parameters for logistic Model on Training
lr.fit(X_train.iloc[:,list_rf], y_train)

df_training_pred_lr = pd.DataFrame({'actual':y_train,'predicted': lr.predict(X_train.iloc[:,list_rf]),
                                 'Non_Target':lr.predict_proba(X_train.iloc[:,list_rf])[:,0],
                                 'Target':lr.predict_proba(X_train.iloc[:,list_rf])[:,1],
                                })

df_testing_pred_lr = pd.DataFrame({'actual':y_test,'predicted': lr.predict(X_test.iloc[:,list_rf]),
                                'Non_Target':lr.predict_proba(X_test.iloc[:,list_rf])[:,0],
                                'Target':lr.predict_proba(X_test.iloc[:,list_rf])[:,1],
                               })

from sklearn import metrics
import matplotlib.pyplot as plt

accuracy_training=metrics.accuracy_score(df_training_pred_lr.actual,df_training_pred_lr.predicted)
accuracy_testing=metrics.accuracy_score(df_testing_pred_lr.actual,df_testing_pred_lr.predicted)

f1_score_training=metrics.f1_score(df_training_pred_lr.actual,df_training_pred_lr.predicted)
f1_score_testing=metrics.f1_score(df_testing_pred_lr.actual,df_testing_pred_lr.predicted)

auc_score_training = metrics.roc_auc_score(df_training_pred_lr.actual, df_training_pred_lr.predicted)
auc_score_testing = metrics.roc_auc_score(df_testing_pred_lr.actual, df_testing_pred_lr.predicted)

print("Accuracy")
print("Training")
print(accuracy_training)
print("Testing")
print(accuracy_testing)
print("\n")

print("F1_score")
print("Training")
print(f1_score_training)
print("Testing")
print(f1_score_testing)
print("\n")

print("AUC_Score")
print("Training")
print(auc_score_training)
print("Testing")
print(auc_score_testing)

#Y_predAB = modelAB.predict(X_valid)
#print("Train Accuracy: ", modelAB.score(X_train, y_train))
#print("Validation Accuracy: ", modelAB.score(X_valid, Y_valid))
#print("AUROC Score of logistic = ", roc_auc_score(Y_valid, Y_predAB))

#### To sove the numbers to be used in the pyqt5
import pickle
filename = 'lr_finalized_model.sav'
pickle.dump(lr, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result=loaded_model.score(X_test.iloc[:,list_rf],y_test)
print(result)

#%----------------- Gradient Boosting -----------------

from sklearn.ensemble import GradientBoostingClassifier
modelGB = GradientBoostingClassifier()
modelGB.fit(X_train.iloc[:,list_rf], y_train)
Y_predGB = modelGB.predict(X_test.iloc[:,list_rf])
print("Training Accuracy: ", modelGB.score(X_train.iloc[:,list_rf], y_train))
print('Testing Accuarcy: ', modelGB.score(X_test.iloc[:,list_rf], y_test))
print("AUROC Score of Gradient Boosting = ", roc_auc_score(y_test, Y_predGB))

#### To sove the numbers to be used in the pyqt5
import pickle
filename = 'gb_finalized_model.sav'
pickle.dump(modelGB, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result=loaded_model.score(X_test.iloc[:,list_rf],y_test)
print(result)
'''