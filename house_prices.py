import numpy as np
import pandas as pd
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import RobustScaler

df_train = pd.read_csv('data/hp_train.csv')
df_test = pd.read_csv('data/hp_test.csv')
X_test = df_test
X_train = df_train.drop('SalePrice', 1)
y_train = df_train['SalePrice']


def remove_nan(X_train):
    df = X_train
    dt = X_train.dtypes
    dfM = dt[dt == 'object'].index
    dfN = dt[dt != 'object'].index
    df[dfN] = df[dfN].fillna(0)
    df[dfM] = df[dfM].fillna('missing')
    return df


def feature_en(df: pd.DataFrame) -> pd.DataFrame:
    totalArea = pd.DataFrame(df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] + df['GrLivArea'] + df['GarageArea'],
                             columns=['TotalArea'])
    yearAvg = pd.DataFrame((df['YearRemodAdd'] + df['YearBuilt']) / 2, columns=['YearAverage'])
    liveAreaQual = pd.DataFrame(df['OverallQual'] * df['GrLivArea'], columns=['LiveAreaQual'])
    df = pd.concat([df, liveAreaQual, totalArea, yearAvg], axis=1)
    return df


def scaling(df: pd.DataFrame) -> np.array:
    scaler = RobustScaler()
    scaler.fit(df)
    return scaler.transform(df)


def catfeatures(X_train: pd.DataFrame, y_train: pd.Series):
    cat_features_index = np.where(X_train.dtypes != float)[0]
    train_pool = Pool(data=X_train,
                      label=y_train,
                      cat_features=cat_features_index,
                      feature_names=list(X_train.columns))
    param = {   'depth': np.arange(10) + 1,
                               'iterations': np.arange(10) * 20,
                               'learning_rate': np.arange(100) / 100,
                               'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
                               }
    # rs = CatBoostRegressor().randomized_search(param_distributions=param,X=train_pool,
    #                                               n_iter=20, cv=6,
    #                                               plot=True)
    param = {'depth': 9,
  'iterations': 60,
  'learning_rate': 0.09,
  'grow_policy': 'Lossguide',
  'nan_mode': 'Max'}
    return CatBoostRegressor(**param).fit(train_pool)


X_train, X_test = remove_nan(X_train), remove_nan(X_test)
X_train, X_test = feature_en(X_train), feature_en(X_test)
dfN = X_train.dtypes[X_train.dtypes != 'object'].index
X_train[dfN], X_test[dfN] = scaling(X_train[dfN]), scaling(X_test[dfN])
model = catfeatures(X_train,y_train)
y_pred = model.predict(X_test)
y_pred = y_pred.rename_axis('Id').reset_index()
y_pred['Id'] = y_pred['Id']+1461
y_pred = y_pred.set_index('Id')
y_pred.to_csv('data/hp_submission.csv')