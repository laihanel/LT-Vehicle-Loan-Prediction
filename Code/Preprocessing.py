import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import os
import re
from sklearn.model_selection import train_test_split
import random
import scorecardpy as sc

# Split train into train data and test data
# os.chdir(r'D:\GWU\Aihan\DATS 6103 Data Mining\Final Project\Code')


# def split_data(inpath, target_name, test_size):
#     df = pd.read_csv(inpath)
#     y = df[target_name]
#     #x = df1.loc[:,df1.columns!='loan_default']
#     x=df.drop(target_name,axis=1)
#     # set a random seed for the data, so that we could get the same train and test set
#     random.seed(12345)
#     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1, stratify=y)
#     #With stratify, we make sure to have the same default rate for both df
#
#     training = pd.concat([X_train, y_train], axis=1)
#     testing = pd.concat([X_test, y_test], axis=1)
#     return training, testing

class PreProcessing():
    def __init__(self, df):
        self.Title = "Preprocessing Start"
        self.df = df
# checking the null value and drop the null value
    def Null_value(self):
        self.df.isnull().sum()
        self.df_new = self.df.dropna()
        return self.df_new

    # convert the format of 'AVERAGE.ACCT.AGE' and 'CREDIT.HISTORY.LENGTH' from 'xyrs xmon' to numbers that represent month.
    def find_number(self, text):
        num = re.findall(r'[0-9]+',text)
        return int(num[0])*12 + int(num[1])

    def comvert_format(self, colname):
        colname_new = self.df[colname].apply(lambda x: self.find_number(x))
        self.df[colname] = colname_new


    # convert categorical string to numbers
    def convert_cate_to_num(self, colname_list):
        for colname in colname_list:
            self.df[colname] = self.df[colname].astype('category')
        cat_columns = self.df.select_dtypes(['category']).columns
        self.df[cat_columns] = self.df[cat_columns].apply(lambda x: x.cat.codes)

    def format_date(self, colname_list):
        for colname in colname_list:
            self.df[colname] = pd.to_datetime(self.df[colname], format = "%d-%m-%y",infer_datetime_format=True)

    def format_age_disbursal(self):
        self.df['Date.of.Birth'] = self.df['Date.of.Birth'].where(self.df['Date.of.Birth'] < pd.Timestamp('now'),
                                                        self.df['Date.of.Birth'] - np.timedelta64(100, 'Y'))
        self.df['Age'] = (pd.Timestamp('now') - self.df['Date.of.Birth']).astype('<m8[Y]').astype(int)
        self.df['Disbursal_months'] = ((pd.Timestamp('now') - self.df['DisbursalDate']) / np.timedelta64(1, 'M')).astype(int)


    def bin_cutpoint(self, target_name, colname_list):
        for colname in colname_list:
            bins_disbursed_amount = sc.woebin(self.df, y=target_name, x=[colname])
            sc.woebin_plot(bins_disbursed_amount)

            pd.concat(bins_disbursed_amount)
            list_break = pd.concat(bins_disbursed_amount).breaks.astype('float').to_list()
            list_break.insert(0, float('-inf'))
            # list_break

            self.df[colname] = pd.cut(self.df[colname], list_break)

    def delet_columns(self, delete_list):
        df_new = self.df.drop(delete_list, axis=1)
        return df_new

    def save_csv(self, outpath):
        self.df.to_csv(outpath,index=False)

'''
# format the date variable
training['Date.of.Birth'] = pd.to_datetime(training['Date.of.Birth']).dt.strftime('%d/%m/%Y')
training['DisbursalDate'] = pd.to_datetime(training['DisbursalDate'], format = "%d-%m-%y",infer_datetime_format=True)
# covert Date of birth to age

def age(born):
    born_date = datetime.strptime(born, "%d/%m/%Y").date()
    today = datetime.now()
    return relativedelta(today, born_date).years

training['Age'] = training['Date.of.Birth'].apply(age)
training['Disbursal_months'] = ((pd.Timestamp('now') - training['DisbursalDate'])/np.timedelta64(1,'M')).astype(int)



'''
if __name__ == "__main__":
    inpath = r'lt-vehicle-loan-default-prediction/train.csv'
    target_name = 'loan_default'
    outpath_train = r'lt-vehicle-loan-default-prediction/final_train.csv'
    #outpath_test = r'lt-vehicle-loan-default-prediction/final_test.csv'
    # training, testing = split_data(inpath, target_name, test_size=0.3)
    # checking the format of each variable
    training = pd.read_csv(inpath)
    print(training.dtypes)

    print(PreProcessing(training).Title)
    df_new = PreProcessing(training).Null_value()


    # There are 5375 missing value

    PreProcessing(df_new).comvert_format('AVERAGE.ACCT.AGE')
    PreProcessing(df_new).comvert_format('CREDIT.HISTORY.LENGTH')
    # comvert_format(training, 'AVERAGE.ACCT.AGE')
    # comvert_format(training, 'CREDIT.HISTORY.LENGTH')

    #PreProcessing(df_new).convert_cate_to_num(['Employment.Type'])
    PreProcessing(df_new).convert_cate_to_num(['Employment.Type', 'PERFORM_CNS.SCORE.DESCRIPTION'])



    # Create Age and Disbursal_months
    PreProcessing(df_new).format_date(['Date.of.Birth', 'DisbursalDate'])
    PreProcessing(df_new).format_age_disbursal()
    df_all = PreProcessing(df_new).delet_columns(['UniqueID', 'Date.of.Birth', 'DisbursalDate', 'PERFORM_CNS.SCORE.DESCRIPTION', 'Employee_code_ID', 'Current_pincode_ID'])
    #df_all = PreProcessing(df_new).delet_columns(['UniqueID', 'Date.of.Birth', 'DisbursalDate', 'Employee_code_ID', 'Current_pincode_ID'])
    # Traditional Credit Scoring
    # PreProcessing(df_new).bin_cutpoint(target_name, ["disbursed_amount", "asset_cost", "ltv", "PERFORM_CNS.SCORE", "PRI.NO.OF.ACCTS",\
    #                                                  "PRI.ACTIVE.ACCTS", "PRI.OVERDUE.ACCTS", "PRI.CURRENT.BALANCE", "PRI.SANCTIONED.AMOUNT",\
    #                                                  "PRI.DISBURSED.AMOUNT", "PRIMARY.INSTAL.AMT", "NEW.ACCTS.IN.LAST.SIX.MONTHS", \
    #                                                  "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS", "AVERAGE.ACCT.AGE", "CREDIT.HISTORY.LENGTH",\
    #                                                  "Age", "Disbursal_months"])

    PreProcessing(df_all).save_csv(outpath_train)


####


'''
# FINISH FOR NOW
'''

'''

# merge_df.to_csv(r'lt-vehicle-loan-default-prediction/merge.csv')

merge_df = pd.read_csv(r'lt-vehicle-loan-default-prediction/merge.csv')
print(merge_df.dtypes)

# fig, axes = plt.subplots(7, 2, figsize=(18, 10))
#
# fig.suptitle('Pokemon Stats by Generation')
#
# sns.boxplot(ax=axes[0, 0], data=merge_df, x='disbursed_amount')
# sns.boxplot(ax=axes[0, 1], data=merge_df, x='asset_cost')
# sns.boxplot(ax=axes[1, 0], data=merge_df, x='ltv')
# sns.boxplot(ax=axes[1, 1], data=merge_df, x='PERFORM_CNS.SCORE')
# sns.boxplot(ax=axes[2, 1], data=merge_df, x='PRI.CURRENT.BALANCE')
# sns.boxplot(ax=axes[3, 0], data=merge_df, x='PRI.SANCTIONED.AMOUNT')
# sns.boxplot(ax=axes[3, 1], data=merge_df, x='PRI.DISBURSED.AMOUNT')
# sns.boxplot(ax=axes[4, 0], data=merge_df, x='SEC.CURRENT.BALANCE')
# sns.boxplot(ax=axes[4, 1], data=merge_df, x='SEC.SANCTIONED.AMOUNT')
# sns.boxplot(ax=axes[5, 0], data=merge_df, x='SEC.DISBURSED.AMOUNT')
# sns.boxplot(ax=axes[5, 1], data=merge_df, x='PRIMARY.INSTAL.AMT')
# sns.boxplot(ax=axes[6, 0], data=merge_df, x='SEC.INSTAL.AMT')
# sns.boxplot(ax=axes[6, 1], data=merge_df, x='AVERAGE.ACCT.AGE')
# plt.show()



# Continuous variable vs categorical variables
score_ranking = ["A-Very Low Risk", "B-Very Low Risk", "C-Very Low Risk", "D-Very Low Risk", \
                 "E-Low Risk", "F-Low Risk", "G-Low Risk",  "H-Medium Risk", "I-Medium Risk", "J-High Risk", "K-High Risk",\
                 "L-Very High Risk", "M-Very High Risk", "No Bureau History Available", "Not Scored: No Activity seen on the customer (Inactive)", \
                 "Not Scored: Not Enough Info available on the customer", "Not Scored: Sufficient History Not Available", "Not Scored: Only a Guarantor",\
                "Not Scored: No Updates available in last 36 months1", "Not Scored: More than 50 active Accounts found"]



# sns.boxplot(x="PERFORM_CNS.SCORE.DESCRIPTION", y="PERFORM_CNS.SCORE", color="b", data=df_subset)
# plt.show()

def df_boxplot(df, xstr, ystr):
    sns.boxplot(x=xstr, y=ystr, palette=sns.color_palette(), data=df)
    plt.show()


# continuous variable vs target
df_subset = merge_df[merge_df['PERFORM_CNS.SCORE.DESCRIPTION'] < 13]
df_boxplot(df_subset, "PERFORM_CNS.SCORE.DESCRIPTION", "PERFORM_CNS.SCORE")


#continuous variable vs target
df_boxplot(merge_df, "loan_default", y="PERFORM_CNS.SCORE")


# t-test
# stats.ttest_ind(a, b, equal_var = False)

'''