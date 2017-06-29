import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime

from preprocess import preprocess
import output
import large


def first_model(path_train, path_test, istrain=False):
    train, test, id_test = preprocess(path_train, path_test)
    print('price changed done')

    y_train = train["price_doc"]
    x_train = train.drop(["id", "timestamp", "price_doc", "average_q_price"], axis=1)
    x_test = test.drop(["id", "timestamp", "average_q_price"], axis=1)

    num_train = len(x_train)
    x_all = pd.concat([x_train, x_test])

    for c in x_all.columns:
        if x_all[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_all[c].values))
            x_all[c] = lbl.transform(list(x_all[c].values))

    x_train = x_all[:num_train]
    x_test = x_all[num_train:]

    xgb_params = {
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.6,
        'colsample_bytree': 1,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }

    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test)

    if istrain == True:
        num_boost_rounds = 422
        model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
        model.save_model('0001.model')

    model = xgb.Booster(dict(xgb_params, silent=0))
    model.load_model('0001.model')

    y_predict = model.predict(dtest)
    return pd.DataFrame({'id': id_test, 'price_doc': y_predict})



def second_model(path_train, path_test, istrain=False):

    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)
    id_test = test.id

    mult = .969

    y_train = train["price_doc"] * mult + 10
    x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
    x_test = test.drop(["id", "timestamp"], axis=1)

    for c in x_train.columns:
        if x_train[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_train[c].values))
            x_train[c] = lbl.transform(list(x_train[c].values))

    for c in x_test.columns:
        if x_test[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_test[c].values))
            x_test[c] = lbl.transform(list(x_test[c].values))

    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }

    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test)

    if istrain == True:
        num_boost_rounds = 385
        model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)
        model.save_model('0002.model')

    model = xgb.Booster(dict(xgb_params, silent=0))
    model.load_model('0002.model')


    y_predict = model.predict(dtest)
    return pd.DataFrame({'id': id_test, 'price_doc': y_predict})


def third_model(path_train, path_test, path_macro, istrain=False):
    # Any results you write to the current directory are saved as output.
    df_train = pd.read_csv(path_train, parse_dates=['timestamp'])
    df_test = pd.read_csv(path_test, parse_dates=['timestamp'])
    df_macro = pd.read_csv(path_macro, parse_dates=['timestamp'])

    mult = 0.969
    df_train.drop(df_train[df_train["life_sq"] > 7000].index, inplace=True)

    
    y_train = df_train['price_doc'].values * mult + 10
    id_test = df_test['id']

    df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
    df_test.drop(['id'], axis=1, inplace=True)

    num_train = len(df_train)
    df_all = pd.concat([df_train, df_test])
    df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')

    # Add month-year
    month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

    # Add week-year count
    week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

    # Add month and day-of-week
    df_all['month'] = df_all.timestamp.dt.month
    df_all['dow'] = df_all.timestamp.dt.dayofweek

    # Other feature engineering
    df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
    df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)


    # Remove timestamp column (may overfit the model in train)
    df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)


    factorize = lambda t: pd.factorize(t[1])[0]

    df_obj = df_all.select_dtypes(include=['object'])

    X_all = np.c_[
        df_all.select_dtypes(exclude=['object']).values,
        np.array(list(map(factorize, df_obj.iteritems()))).T
    ]

    X_train = X_all[:num_train]
    X_test = X_all[num_train:]


    # Deal with categorical values
    df_numeric = df_all.select_dtypes(exclude=['object'])
    df_obj = df_all.select_dtypes(include=['object']).copy()

    for c in df_obj:
        df_obj[c] = pd.factorize(df_obj[c])[0]

    df_values = pd.concat([df_numeric, df_obj], axis=1)


    # Convert to numpy values
    X_all = df_values.values

    X_train = X_all[:num_train]
    X_test = X_all[num_train:]

    df_columns = df_values.columns


    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }

    dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
    dtest = xgb.DMatrix(X_test, feature_names=df_columns)

    if istrain == True:
        num_boost_rounds = 420  # From Bruno's original CV, I think
        model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
        model.save_model('0003.model')
        
    model = xgb.Booster(dict(xgb_params, silent=0))
    model.load_model('0003.model')

    y_pred = model.predict(dtest)
    return pd.DataFrame({'id': id_test, 'price_doc': y_pred})



def main():

    path_train  = 'train.csv'
    path_test   = 'test.csv'
    path_macro  = 'macro.csv'
    path_xgb    = 'xgb.csv'
    path_mean    = 'mean.csv'
    path_ans    = 'ans.csv'

    first_output = first_model(path_train, path_test)
    print ('First OK')

    second_output = second_model(path_train, path_test)
    print ('Second OK')

    third_sub = third_model(path_train, path_test, path_macro)
    print ('Third OK')


    result = second_output.merge(third_sub, on="id", suffixes=['_louis','_bruno'])
    result["price_doc"] = np.exp( .714*np.log(result.price_doc_louis) + .286*np.log(result.price_doc_bruno) )  # multiplies out to .5 & .2
    result = result.merge(first_output, on="id", suffixes=['_follow','_gunja'])
    result["price_doc"] = np.exp( .78*np.log(result.price_doc_follow) + .22*np.log(result.price_doc_gunja) )
                                         
    result.drop(["price_doc_louis","price_doc_bruno","price_doc_follow","price_doc_gunja"],axis=1,inplace=True)
    result.to_csv(path_xgb, index=False)
    print ('Output merged')

    output.modify(path_test, path_xgb, path_mean)
    print ('Ans modified mean')

    # answer
    large.modify(path_mean, path_ans)
    print ('Ans modified large')

if __name__=='__main__':
    main()