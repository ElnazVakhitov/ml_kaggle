import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

df_train = pd.read_csv('data/t_train.csv')
df_test = pd.read_csv('data/t_test.csv')
X_test = df_test
X_train = df_train.drop('Survived', 1)
y_train = df_train['Survived']


def remove_nan(df):
    dt = df.dtypes
    dfM = dt[dt == 'object'].index
    dfN = dt[dt != 'object'].index
    df[dfN] = df[dfN].fillna(-1)
    df[dfM] = df[dfM].fillna('missing')
    return df


def feature_en(df: pd.DataFrame) -> pd.DataFrame:
    relatives = pd.DataFrame(df['SibSp'] + df['Parch'], columns=['relatives'])
    df = pd.concat([df, relatives], axis=1)
    return df


def catfeatures(X_train: pd.DataFrame, y_train: pd.Series):
    cat_features_index = np.where(X_train.dtypes != float)[0]
    train_pool = Pool(data=X_train,
                      label=y_train,
                      cat_features=cat_features_index,
                      feature_names=list(X_train.columns))
    # param = {'depth': np.arange(10) + 1,
    #          'iterations': np.arange(10) * 20,
    #          'learning_rate': np.arange(100) / 100,
    #          'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
    #          }
    # rs = CatBoostClassifier().randomized_search(param_distributions=param, X=train_pool,
    #                                             n_iter=20, cv=6)
    param = {
     'iterations': 180,
     'grow_policy': 'Depthwise',
     'l2_leaf_reg': 3,
     'random_seed': 0,
     'depth': 8,
     'min_data_in_leaf': 1,
     'loss_function': 'Logloss',
     'learning_rate': 0.17000000178813934,
     'max_leaves': 256}
    # param = rs['params']
    return CatBoostClassifier(**param).fit(train_pool)


X_train, X_test = remove_nan(X_train), remove_nan(X_test)
X_train, X_test = feature_en(X_train), feature_en(X_test)
dfN = X_train.dtypes[X_train.dtypes != 'object'].index
model_cat = catfeatures(X_train, y_train)
y_pred = model_cat.predict(X_test)
y_pred = pd.DataFrame(y_pred)
y_pred = y_pred.rename_axis('PassengerId').reset_index()
y_pred['PassengerId'] = y_pred['PassengerId']+892
y_pred = y_pred.set_index('PassengerId')
y_pred.to_csv('data/t_submission.csv')