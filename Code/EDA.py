#########
11/15/2021

#Generating a dataset for only categorical
df_categorical = df.select_dtypes(exclude=['number'])
df_categorical=df_categorical.drop(['Date.of.Birth','DisbursalDate'],axis=1)
df_categorical.head()

#Building a Dataset for numerical (continous)
df_continuous = df.select_dtypes(include=['number'])
df_continuous=df_continuous.drop(['UniqueID'],axis=1)
df_continuous.head()

#Univariate Analysis
import matplotlib.pyplot as plt # Charge matplotlib
import seaborn as sns   # Charge seaborn

#To obtain the basic statistics
df_continuous.describe()

#Get the List of all Column Names
continuous_list = list(df_continuous)

# Plot for all continous
#1
sns.displot(df['disbursed_amount'][df['disbursed_amount'] < df['disbursed_amount'].quantile(.99)],kind='hist',kde=True)
plt.show()

#2
sns.displot(df['asset_cost'][df['asset_cost'] < df['asset_cost'].quantile(.99)],kind='hist',kde=True)
plt.show()

#3
sns.displot(df['ltv'][df['ltv'] < df['ltv'].quantile(.99)],kind='hist',kde=True)
plt.show()

#4
sns.displot(df['PERFORM_CNS.SCORE'][df['PERFORM_CNS.SCORE'] < df['PERFORM_CNS.SCORE'].quantile(.99)],kind='hist',kde=True)
plt.show()

#5
sns.displot(df['PRI.NO.OF.ACCTS'][df['PRI.NO.OF.ACCTS'] < df['PRI.NO.OF.ACCTS'].quantile(.99)],kind='hist',kde=True)
plt.show()

#6
sns.displot(df['PRI.ACTIVE.ACCTS'][df['PRI.ACTIVE.ACCTS'] < df['PRI.ACTIVE.ACCTS'].quantile(.99)],kind='hist',kde=True)
plt.show()

#6
sns.displot(df['PRI.OVERDUE.ACCTS'][df['PRI.OVERDUE.ACCTS'] < df['PRI.OVERDUE.ACCTS'].quantile(.99)],kind='hist',kde=True)
plt.show()

#7
sns.displot(df['PRI.CURRENT.BALANCE'][df['PRI.CURRENT.BALANCE'] < df['PRI.CURRENT.BALANCE'].quantile(.99)],kind='hist',kde=True)
plt.show()

#8
sns.displot(df['PRI.SANCTIONED.AMOUNT'][df['PRI.SANCTIONED.AMOUNT'] < df['PRI.SANCTIONED.AMOUNT'].quantile(.99)],kind='hist',kde=True)
plt.show()

#9
sns.displot(df['PRI.DISBURSED.AMOUNT'][df['PRI.DISBURSED.AMOUNT'] < df['PRI.DISBURSED.AMOUNT'].quantile(.99)],kind='hist',kde=True)
plt.show()

#10
sns.displot(df['SEC.NO.OF.ACCTS'][df['SEC.NO.OF.ACCTS'] < df['SEC.NO.OF.ACCTS'].quantile(.99)],kind='hist',kde=True)
plt.show()

#11
sns.displot(df['SEC.ACTIVE.ACCTS'][df['SEC.ACTIVE.ACCTS'] < df['SEC.ACTIVE.ACCTS'].quantile(.99)],kind='hist',kde=True)
plt.show()

#12
sns.displot(df['SEC.OVERDUE.ACCTS'][df['SEC.OVERDUE.ACCTS'] < df['SEC.OVERDUE.ACCTS'].quantile(.99)],kind='hist',kde=True)
plt.show()

#13
sns.displot(df['SEC.CURRENT.BALANCE'][df['SEC.CURRENT.BALANCE'] < df['SEC.CURRENT.BALANCE'].quantile(.99)],kind='hist',kde=True)
plt.show()

#14
sns.displot(df['SEC.SANCTIONED.AMOUNT'][df['SEC.SANCTIONED.AMOUNT'] < df['SEC.SANCTIONED.AMOUNT'].quantile(.99)],kind='hist',kde=True)
plt.show()

#15
sns.displot(df['SEC.DISBURSED.AMOUNT'][df['SEC.DISBURSED.AMOUNT'] < df['SEC.DISBURSED.AMOUNT'].quantile(.99)],kind='hist',kde=True)
plt.show()

#16
sns.displot(df['PRIMARY.INSTAL.AMT'][df['PRIMARY.INSTAL.AMT'] < df['PRIMARY.INSTAL.AMT'].quantile(.99)],kind='hist',kde=True)
plt.show()

#17
sns.displot(df['NEW.ACCTS.IN.LAST.SIX.MONTHS'][df['NEW.ACCTS.IN.LAST.SIX.MONTHS'] < df['NEW.ACCTS.IN.LAST.SIX.MONTHS'].quantile(.99)],kind='hist',kde=True)
plt.show()

#18
sns.displot(df['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS'][df['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS'] < df['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS'].quantile(.99)],kind='hist',kde=True)
plt.show()

#19
sns.displot(df['AVERAGE.ACCT.AGE'][df['AVERAGE.ACCT.AGE'] < df['AVERAGE.ACCT.AGE'].quantile(.99)],kind='hist',kde=True)
plt.show()

#20
sns.displot(df['CREDIT.HISTORY.LENGTH'][df['CREDIT.HISTORY.LENGTH'] < df['CREDIT.HISTORY.LENGTH'].quantile(.99)],kind='hist',kde=True)
plt.show()

#21
sns.displot(df['CREDIT.HISTORY.LENGTH'][df['CREDIT.HISTORY.LENGTH'] < df['CREDIT.HISTORY.LENGTH'].quantile(.99)],kind='hist',kde=True)
plt.show()

#22
sns.displot(df['NO.OF_INQUIRIES'][df['NO.OF_INQUIRIES'] < df['NO.OF_INQUIRIES'].quantile(.99)],kind='hist',kde=True)
plt.show()

#23
sns.displot(df['Age'][df['Age'] < df['Age'].quantile(.99)],kind='hist',kde=True)
plt.show()

#23
sns.displot(df['Disbursal_months'][df['Disbursal_months'] < df['Disbursal_months'].quantile(.99)],kind='hist',kde=True)
plt.show()

########Multivariate Analysis

plt.rcParams["figure.figsize"] = (10,7)
sns.heatmap(df_continuous.corr())
plt.show()

#Heat map
sns.heatmap(df_continuous.corr(), cmap="YlGnBu", annot=False,mask=np.triu(df_continuous.corr()))
plt.show()

#Heat map that highligts if the correlation is greater than 0.6
sns.heatmap(df_continuous.corr().abs()>0.6, cmap="YlGnBu", annot=False,mask=np.triu(df_continuous.corr()))
plt.show() # black are with the highest correlation

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

list1=get_top_abs_correlations(df_continuous,n=9)
print(list1)


'''
11/22 Aihan added, file name unchanged
test needed
'''
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