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

# split train into train data and test data
# os.chdir(r'D:\GWU\Aihan\DATS 6103 Data Mining\Final Project\Code')


# def split_data(inpath, target_name, test_size):
#     df = pd.read_csv(inpath)
#     y = df[target_name]
#     #x = df1.loc[:,df1.columns!='loan_default']
#     x=df.drop(target_name,axis=1)
#     # set a random seed for the data, so that we could get the same train and test set
#     random.seed(12345)
#     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1, stratify=y)
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
        self.df.to_csv(outpath, index=False)



if __name__ == "__main__":
    inpath = r'lt-vehicle-loan-default-prediction/train.csv'
    target_name = 'loan_default'
    outpath_train = r'lt-vehicle-loan-default-prediction/final_train.csv'
    # outpath_test = r'lt-vehicle-loan-default-prediction/final_test.csv'
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

    PreProcessing(df_new).convert_cate_to_num(['Employment.Type', 'PERFORM_CNS.SCORE.DESCRIPTION'])

    # Create Age and Disbursal_months
    PreProcessing(df_new).format_date(['Date.of.Birth', 'DisbursalDate'])
    PreProcessing(df_new).format_age_disbursal()
    df_all = PreProcessing(df_new).delet_columns(['UniqueID', 'Date.of.Birth', 'DisbursalDate', 'PERFORM_CNS.SCORE.DESCRIPTION', 'Employee_code_ID', 'Current_pincode_ID'])

    # Traditional Credit Scoring
    # PreProcessing(df_new).bin_cutpoint(target_name, ["disbursed_amount", "asset_cost", "ltv", "PERFORM_CNS.SCORE", "PRI.NO.OF.ACCTS",\
    #                                                  "PRI.ACTIVE.ACCTS", "PRI.OVERDUE.ACCTS", "PRI.CURRENT.BALANCE", "PRI.SANCTIONED.AMOUNT",\
    #                                                  "PRI.DISBURSED.AMOUNT", "PRIMARY.INSTAL.AMT", "NEW.ACCTS.IN.LAST.SIX.MONTHS", \
    #                                                  "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS", "AVERAGE.ACCT.AGE", "CREDIT.HISTORY.LENGTH",\
    #                                                  "Age", "Disbursal_months"])

    PreProcessing(df_all).save_csv(outpath_train)

