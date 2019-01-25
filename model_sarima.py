
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from warnings import catch_warnings
from warnings import filterwarnings
import pickle
from collections import defaultdict


def load_data(data_path='xrp_data_clean.csv'):
    '''
    Loads dataframe and outputs a single series of percent change of there
    close daily value
    '''
    df = pd.read_csv(data_path)
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    df = df.Close.pct_change()
    df.dropna(inplace=True)
    return df


def data_split(df, test=16):
    '''
    Separates the training/validation data from test data
    '''
    df_train = df.iloc[:-test]
    df_test = df.iloc[-test:]
    return df_train, df_test


def data_train(df_train, iter=0, window_len=0, valid=30):
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

    return df_train_data, valid_data


def RMSE(validation_points, prediction_points):
    '''
    Calculate RMSE between two vectors
    '''
    x = np.array(validation_points)
    y = np.array(prediction_points)

    return np.sqrt(np.mean((x - y)**2))


def n_iterations(df_train, window_len):
    return np.arange(start=0, stop=len(df_train) - window_len, step=1)


def sarimax_forecast(train_data, config):
    '''
    Returns a sarimax prediction
    '''
    order, sorder, trend = config
    # Pull out configuraton terms

    # fit model
    model = SARIMAX(endog=train_data,
                    order=order,
                    seasonal_order=sorder,
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    # make one step prediction
    prediction = model.fit(disp=0).forecast()[0]
    return prediction


def walk_forward_validation(df_train, config, window_len):
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
        df_train_data, valid_data = data_train(
            df_train=df_train, iter=i, window_len=window_len)
        # Calculate a model on a specific parameter and make a prediction
        try:
            with catch_warnings():
                filterwarnings("ignore")
                prediction = sarimax_forecast(
                    train_data=df_train_data.values, config=config)
        except:
            #print('Skipped config {} step {}'.format(config, i))
            continue

        # Append prediction and validation data to be evaluated
        validation_points.append(valid_data)
        prediction_points.append(prediction)

    # Calculate RMSE
    rmse = RMSE(validation_points, prediction_points)
    return rmse


def sarima_configs(seasonal=[7]):
    '''
    Builds a list of configuration to perform grid search
    change the ranges of parameters as desired for p,d,q,P,D,Q,S,trend
    variables
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
    t_params = ['n']
    P_params = np.arange(start=0, stop=1 + 1, step=1)
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


def grid_search(df_train, configs, window_len):
    rmse_list = list()
    sarima_dict = defaultdict(list)
    for config in configs:
        print('Running config: {}'.format(config))
        rmse = walk_forward_validation(
            df_train=df_train, config=config, window_len=window_len)
        print('Config {} scored {}'.format(config, rmse))

        sarima_dict[tuple(config)].append(rmse)
        save_obj(sarima_dict, 'sarima_dict7')
        rmse_list.append(rmse)

    return tuple([configs, rmse_list])


if __name__ == '__main__':
    # Loads data
    df = load_data()
    # Splits training/validation and test
    df_train, df_test = data_split(df)
    # Window length - from 80/20 split
    window_len = 1600
    configs = sarima_configs()
    scores = grid_search(df_train=df_train,
                         configs=configs,
                         window_len=window_len)

# Best model found was (1,0,1)(1,0,1,12),'n'
