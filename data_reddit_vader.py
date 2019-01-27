from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    # Load xrp price data to plot with other
    data_path = 'xrp_data_clean.csv'
    df = pd.read_csv(data_path)
    df.sort_values('Date', inplace=True)
    df['Date'] = df.Date.apply(pd.to_datetime)
    df.set_index('Date', inplace=True)
    return df


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def sentiment_analyzer_scores(sentence):
    # runs vader on a text string
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))


def sentiment_analzer(dict_reddit):
    analyser = SentimentIntensityAnalyzer()
    # run sentiment analysis on entire dictionarty
    neg = []
    pos = []
    neu = []
    compound = []
    date = []

    for key, item in dict_reddit.items():
        text = " ".join(item)
        score = analyser.polarity_scores(text)

        neg.append(score['neg'])
        pos.append(score['pos'])
        neu.append(score['neu'])
        compound.append(score['compound'])
        date.append(key)

    # Put into a dataframe
    df_sent = pd.DataFrame()
    df_sent['neg'] = pd.Series(data=neg, index=date)
    df_sent['pos'] = pd.Series(data=pos, index=date)
    df_sent['neu'] = pd.Series(data=neu, index=date)
    df_sent['Compound'] = pd.Series(data=compound, index=date)
    df_sent.index.name = 'Date'
    df_sent.reset_index(inplace=True)
    df_sent['Date'] = df_sent.Date.apply(pd.to_datetime)
    df_sent.set_index('Date', inplace=True)

    return df_sent


def plots(df_sent, df):
    fig, ax1 = plt.subplots()
    ax1.scatter(df_sent.index, df_sent.Compound,
                s=1, alpha=.5, color='tab:blue')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Sentiment Score', color='tab:blue', fontsize=12)
    ax1.tick_params('y', colors='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(df.index, df.Close, color='tab:green')
    ax2.set_ylabel('sin', color='tab:green')
    ax2.tick_params('y', colors='tab:green')
    ax2.set_ylabel('Ripple Percent Change [%]', fontsize=12)
    plt.title('Ripple Percent Change and Reddit Sentiment', fontsize=16)
    fig.savefig('Price_and_Sentiment.svg')


if __name__ == '__main__':
    df = load_data()
    dict_reddit = load_obj('ripple_reddit_posts.pk')
    df_sent = sentiment_analzer(dict_reddit)
    plots(df_sent, df)
