import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

from sklearn.cross_validation import train_test_split

warnings.filterwarnings('ignore')
#matplotlib inline
df_train = pd.read_csv('../input/train.csv')


total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'])
missing_cols=missing_data[missing_data['Total']>1].index
print(missing_cols)
df_train = df_train.drop(missing_data[missing_data['Total']>1].index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)


df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
#print("SalePrice")
#print(df_train['SalePrice'])
df_train['SalePrice'] = np.log(df_train['SalePrice'])
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

#print(np.exp(df_train['SalePrice']))


df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1

df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

df_train['MSSubClass'] = df_train['MSSubClass'].apply(str)
df_train['OverallCond'] = df_train['OverallCond'].astype(str)
df_train['YrSold'] = df_train['YrSold'].astype(str)
df_train['MoSold'] = df_train['MoSold'].astype(str)

cols = (   'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 
         'Functional',  'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

# apply sklearn.preprocessing.LabelEncoder to each categorical feature
import sklearn.preprocessing

for c in cols:
    lbl = sklearn.preprocessing.LabelEncoder() 
    lbl.fit(list(df_train[c].values)) 
    df_train[c] = lbl.transform(list(df_train[c].values))

#df_train = pd.get_dummies(df_train)

train, test = train_test_split(df_train, test_size=0.25)

train_y = train['SalePrice']
test_y = test['SalePrice']
print(train)
print(test)
train_x = pd.DataFrame(train.drop('SalePrice',axis=1,inplace=False))
test_x = pd.DataFrame(test.drop('SalePrice', axis=1, inplace=False))




m_col = ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage',
       'GarageCond', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual',
       'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual',
       'MasVnrArea', 'MasVnrType')


df_test = pd.read_csv('../input/test.csv')
print(df_test.dtypes)
df_test = df_test.drop(missing_cols,1,inplace=False)
print(df_test.dtypes)

total = df_test.isnull().sum().sort_values(ascending=False)
percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'])
print(missing_data)



#df_test.fillna(df_test.mean(),inplace=True)
df_test['MSZoning'] = df_test['MSZoning'].fillna(df_test['MSZoning'].mode()[0])
df_test['Utilities'] = df_test['Utilities'].fillna(df_test['Utilities'].mode()[0])
df_test['Functional'] = df_test['Functional'].fillna("Typ")
df_test['Exterior2nd'] = df_test['Exterior2nd'].fillna(df_test['Utilities'].mode()[0])
df_test['KitchenQual'] = df_test['KitchenQual'].fillna(df_test['KitchenQual'].mode()[0])
df_test['GarageCars'] = df_test['GarageCars'].fillna(df_test['GarageCars'].mode()[0])

cols_fill = ('BsmtHalfBath','BsmtFullBath','GarageArea','BsmtFinSF1','SaleType','TotalBsmtSF','BsmtUnfSF','BsmtFinSF2','Exterior1st')

for col in cols_fill:
    df_test[col] = df_test[col].fillna(df_test[col].mode()[0])
total = df_test.isnull().sum().sort_values(ascending=False)
print(total)
#df_test['SalePrice'] = np.log(df_test['SalePrice'])
df_test['GrLivArea'] = np.log(df_test['GrLivArea'])

df_test['HasBsmt'] = pd.Series(len(df_test['TotalBsmtSF']), index=df_test.index)
df_test['HasBsmt'] = 0 
df_test.loc[df_test['TotalBsmtSF']>0,'HasBsmt'] = 1

df_test.loc[df_test['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_test['TotalBsmtSF'])

df_test['MSSubClass'] = df_test['MSSubClass'].apply(str)
df_test['OverallCond'] = df_test['OverallCond'].astype(str)
df_test['YrSold'] = df_test['YrSold'].astype(str)
df_test['MoSold'] = df_test['MoSold'].astype(str)

import sklearn.preprocessing

for c in cols:
    lbl = sklearn.preprocessing.LabelEncoder() 
    lbl.fit(list(df_test[c].values)) 
    df_test[c] = lbl.transform(list(df_test[c].values))

df_train['label'] = 'train'
df_test['label'] = 'test'

print(df_train)
concat_df = pd.concat([df_train, df_test])
concat_df = pd.get_dummies(concat_df)
print("concat")
print(concat_df)
df_train = concat_df[concat_df['label_train']==1]
df_test = concat_df[concat_df['label_test']==1]

df_train = df_train.drop(['label_train','label_test'], axis=1)
df_test = df_test.drop(['label_train','label_test','SalePrice'], axis=1)
#df_test = pd.get_dummies(df_test)
df_train_x = pd.DataFrame(df_train.drop('SalePrice', axis=1, inplace=False))


print("column names")
print(df_test['Id'])
print(df_test.columns)
total = df_test.isnull().sum().sort_values(ascending=False)
print(total)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_jobs=-1)
#print(train_x)
#print(train_y)
model.fit(df_train_x,df_train['SalePrice'])
pred = model.predict(df_test)



#from sklearn.metrics import mean_squared_error
#from math import sqrt

#rms = sqrt(mean_squared_error(test_y, pred))

sub = pd.DataFrame({'Id':df_test['Id'],'SalePrice':np.exp(pred)})
#sub['Id'] = df_test['Id']
#sub['SalePrice'] = pred
sub.to_csv('housing_submission.csv',index=False)

#from sklearn.metrics import accuracy_score
#accuracy = accuracy_score(test_y, pred)
#print(rms)
#print(df_train.head())