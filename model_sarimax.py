
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from warnings import catch_warnings
from warnings import filterwarnings
import pickle
from collections import defaultdict
import datetime


def load_data(data_path='xrp_data_clean.csv'):
    '''
    Loads dataframe and outputs a single series of percent change of there
    close daily value
    '''
    #data_path = 'xrp_data_clean.csv'
    df = pd.read_csv(data_path)
    limit_date = datetime.datetime.strptime('2013-12-31', '%Y-%m-%d')
    df.sort_values('Date', inplace=True)
    df.Date = pd.to_datetime(df.Date)
    df = df[df.Date >= limit_date]
    df.set_index('Date', inplace=True)
    df = df.Close.pct_change()
    df.dropna(inplace=True)

    return df


def load_sent_data(data_path='df_sent.pk'):
    '''
    Loads dataframe and outputs a single series of percent change of there
    close daily value
    '''
    #data_path = 'df_sent.pk'
    df_sent = pd.read_pickle(data_path)
    df_sent.index.name = 'Date'
    df_sent.reset_index(inplace=True)
    df_sent.sort_values('Date', inplace=True)
    limit_date = datetime.datetime.strptime('2013-12-31', '%Y-%m-%d')
    df_sent = df_sent[df_sent.Date >= limit_date]
    df_sent.set_index('Date', inplace=True)

    start = datetime.datetime.strptime("2013-12-31", "%Y-%m-%d")
    end = datetime.datetime.strptime("2019-01-15", "%Y-%m-%d")
    date_generated = [
        start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]

    df_temp = pd.Series(data=np.nan, index=date_generated)
    df_sent = pd.concat([df_sent, df_temp], axis=1)
    df_sent.fillna(method='ffill', inplace=True)
    df_sent.drop(0, axis=1, inplace=True)
    df = df_sent['neg']
    return df


def data_split(df, df_sent, test=16):
    '''
    Separates the training/validation data from test data
    '''
    df_train = df.iloc[:-test]
    df_test = df.iloc[-test:]
    df_sent_train = df_sent.iloc[:-test]
    df_sent_test = df_sent.iloc[-test:]
    return df_train, df_test, df_sent_train, df_sent_test


def data_train(df_train, df_sent_train, iter=0, window_len=0, valid=30):
    '''
    Grabs the correct window of data, takes in the iteration that the
    function is on. If window_len is specified as not zero, there will be
    valid points iterations and will use all points - valid as the data to
    run the ar on.

    However if window_len is not zero, it will separate out different
    times of windows of that specified length. It also ignores the valid
    input

    input:
    df_train is taken from output of data_split step
    iter is the iteration user is on, function will be looped
    window_len a int calling out a specific window length to perform the ar on

    output:
    train_data is an array to train data on
    valid_
    '''
    if window_len == 0:
        window_len = len(df) - valid
    df_train_data = df_train.iloc[iter:window_len + iter]
    valid_data = df_train.iloc[window_len + iter]

    if window_len == 0:
        window_len = len(df) - valid
    df_sent_train_data = df_sent_train.iloc[iter:window_len + iter]

    # Shift date back one day
    df_sent_train_data.index.name = 'Date'
    df_sent_train_data = df_sent_train_data.reset_index()
    df_sent_train_data['Date'] = df_sent_train_data['Date'] + \
        pd.Timedelta('1 day')
    df_sent_train_data.set_index('Date', inplace=True)

    valid_sent_data = np.array(
        [df_sent_train.iloc[window_len + iter].tolist()])
    valid_sent_data = valid_sent_data.reshape((len(valid_sent_data), 1))

    return df_train_data, df_sent_train_data, valid_data, valid_sent_data


def RMSE(validation_points, prediction_points):
    '''
    Calculate RMSE between two vectors
    '''
    x = np.array(validation_points)
    y = np.array(prediction_points)

    return np.sqrt(np.mean((x - y)**2))


def n_iterations(df_train, window_len):
    return np.arange(start=0, stop=len(df_train) - window_len, step=1)


def sarimax_forecast(train_data, sent_data, valid_sent_data, config):
    '''
    Returns a sarimax prediction, same as sarima with the addition of the
    exogenous reddit data
    '''
    order, sorder, trend = config
    # Pull out configuraton terms

    # fit model
    model = SARIMAX(endog=train_data,
                    exog=sent_data,
                    order=order,
                    seasonal_order=sorder,
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    # make one step prediction
    prediction = model.fit(disp=0).forecast(exog=valid_sent_data)[0]
    return prediction


def walk_forward_validation(df_train, df_sent_train, config, window_len):
    '''
    Calculates the RMSE for a single SARIMAX configuration
    '''
    # Number of validation steps
    n = n_iterations(df_train=df_train, window_len=window_len)

    validation_points = list()
    prediction_points = list()
    # Step through each validation step
    for i in n:
        #print('Running config {} step {}'.format(config, i))
        # Grab correct block of data and validation data point
        df_train_data, df_sent_train_data, valid_data, valid_sent_data = data_train(
            df_train=df_train, df_sent_train=df_sent_train, iter=i, window_len=window_len)
        # Calculate a model on a specific parameter and make a prediction
        try:
            with catch_warnings():
                filterwarnings("ignore")
                prediction = sarimax_forecast(
                    train_data=df_train_data.values,
                    sent_data=df_sent_train_data.values,
                    valid_sent_data=valid_sent_data,
                    config=config)
        except:
            #print('Skipped config {} step {}'.format(config, i))
            continue

        # Append prediction and validation data to be evaluated
        validation_points.append(valid_data)
        prediction_points.append(prediction)

    # Calculate RMSE
    rmse = RMSE(validation_points, prediction_points)
    return rmse


def sarimax_configs(seasonal=[12]):
    '''
    Builds a list of configuration to perform grid search
    '''
    model_configs = list()
    # define config lists
    # p_params = np.arange(start=0, stop=2 + 1, step=1)
    # d_params = np.arange(start=0, stop=1 + 1, step=1)
    # q_params = np.arange(start=0, stop=2 + 1, step=1)
    # t_params = ['n', 'c', 't', 'ct']
    # P_params = np.arange(start=0, stop=2 + 1, step=1)
    # D_params = np.arange(start=0, stop=1 + 1, step=1)
    # Q_params = np.arange(start=0, stop=2 + 1, step=1)

    p_params = np.arange(start=1, stop=1 + 1, step=1)
    d_params = np.arange(start=0, stop=0 + 1, step=1)
    q_params = np.arange(start=0, stop=1 + 1, step=1)
    #t_params = ['n', 'c', 't', 'ct']
    t_params = ['n']
    P_params = np.arange(start=1, stop=1 + 1, step=1)
    D_params = np.arange(start=0, stop=0 + 1, step=1)
    Q_params = np.arange(start=0, stop=1 + 1, step=1)
    m_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p, d, q), (P, D, Q, m), t]
                                    model_configs.append(cfg)

    return model_configs


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def grid_search(df_train, df_sent_train, configs, window_len):
    rmse_list = list()
    sarima_dict = defaultdict(list)
    for config in configs:
        print('Running config: {}'.format(config))
        rmse = walk_forward_validation(
            df_train=df_train, df_sent_train=df_sent_train, config=config, window_len=window_len)
        print('Config {} scored {}'.format(config, rmse))

        sarima_dict[tuple(config)].append(rmse)
        #save_obj(sarima_dict, 'sarimax_neg_dict')
        rmse_list.append(rmse)

    return tuple([configs, rmse_list])


if __name__ == '__main__':
    # Loads data
    df = load_data()
    df_sent = load_sent_data()

    # Splits training/validation and test
    df_train, df_test, df_sent_train, df_sent_test = data_split(df, df_sent)

    # Window length (80/20 split of data)
    window_len = 1600

    # Loads search space
    configs = sarimax_configs()

    # Run grid search
    scores = grid_search(df_train=df_train,
                         df_sent_train=df_sent_train,
                         configs=configs,
                         window_len=window_len)

# Best model found was (1,0,1)(1,0,1,12),'n'
