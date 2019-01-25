
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
    end = datetime.datetime.strptime("2019-01-16", "%Y-%m-%d")
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


def data_train(df_train, df_sent_train, iter=0, window_len=0, test=30):
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
        window_len = len(df_train) - test
    df_train_data = df_train.iloc[iter:window_len + iter]
    valid_data = df_train.iloc[window_len + iter]

    df_sent_train_data = df_sent_train.iloc[iter:window_len + iter]
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
    Returns a sarimax prediction
    '''
    order, sorder, trend = tuple(config[0])
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
        print('Predicting:{}'.format(df_train.index[window_len + i]))
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
                prediction_points.append(prediction)
        except:
            #print('Skipped config {} step {}'.format(config, i))
            continue

        # Append prediction and validation data to be evaluated
        validation_points.append(valid_data)

    # Calculate RMSE
    rmse = RMSE(validation_points, prediction_points)
    return validation_points, prediction_points, rmse


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
    q_params = np.arange(start=1, stop=1 + 1, step=1)
    #t_params = ['n', 'c', 't', 'ct']
    t_params = ['n']
    P_params = np.arange(start=1, stop=1 + 1, step=1)
    D_params = np.arange(start=0, stop=0 + 1, step=1)
    Q_params = np.arange(start=1, stop=1 + 1, step=1)
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


def plot_predicts(prediction_dates, validation_points, prediction_points):
    plt.figure(figsize=(10, 8))
    plt.plot(prediction_dates, np.array(validation_points) * 100)
    plt.plot(prediction_dates, np.array(prediction_points) * 100)
    plt.title('SARIMAX (1,0,1)(1,0,1,12,n) RMSE:{}%'.format(
        np.round(rmse * 100, 2)), fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.xticks(rotation='45')
    plt.ylabel('Daily Percent change [%]', fontsize=12)
    plt.ylim([-11, 11])
    plt.legend(['Actual', 'Prediction'], fontsize=12)
    plt.grid(which='both')
    plt.savefig('SARIMAX_close.svg')


if __name__ == '__main__':
    # Loads data - price and sentiment
    df_train = load_data()
    df_sent_train = load_sent_data()

    # df_train, df_test, df_sent_train, df_sent_test = data_split(
    #     df_train, df_sent_train, test=16)
    # df_test.std()
    df_train = load_data()

    # Window length - pulls 16 last day for test
    window_len = 1826

    # Loads optimized model from validation set
    config = sarimax_configs()
    validation_points, prediction_points, rmse = walk_forward_validation(
        df_train=df_train, df_sent_train=df_sent_train, config=config, window_len=window_len)

    # plots results
    plot_predicts(prediction_dates, validation_points, prediction_points)


# Best model found was (1,0,1)(1,0,1,12),'n'


# Code used to make presentation plot
# plt.figure(figsize=(10, 8))
# plt.subplot(2, 1, 1)
# plt.title('Ripple Coin Price and Percent Change', fontsize=16)
# plt.plot(prediction_dates, np.array(validation_points) * 100, color='tab:blue')
# plt.plot(prediction_dates, np.array(
#     prediction_points) * 100, color='tab:orange')
# plt.legend(['Actual', 'Prediction'], fontsize=12)
# #plt.xlabel('Date', fontsize=12)
# locs, labels = plt.xticks()
# plt.grid(which='major', axis='both')
# plt.xticks(ticks=[], labels=labels)
# plt.ylabel('Percent Change [%]', color='black', fontsize=12)
# plt.tick_params('y', colors='black')
# fig.savefig('Percentage.svg', bbox_inches='tight')
#
# plt.subplot(2, 1, 2)
# plt.plot(df_price.Close.iloc[window_len + 1:].index,
#          df_price.Close.iloc[window_len + 1:].values, color='tab:green')
# plt.tick_params('y', colors='black')
# plt.ylabel('Ripple Price [$]', fontsize=12)
# plt.grid(which='both')
# plt.xticks(rotation='45')
# plt.savefig('Price.svg', bbox_inches='tight')
