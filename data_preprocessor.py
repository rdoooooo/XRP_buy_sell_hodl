
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline


def load_data(filename='xrp_data2.csv'):
    return pd.read_csv(filename)


def clean_df(df):
    # Drop extra column
    df.drop('Unnamed: 0', axis=1, inplace=True)
    # Remove symbol from column names
    df.columns = df.columns.str.replace('*', '')
    df.columns = df.columns.str.replace(' ', '_')
    # Convert Date to a date time object
    df['Date'] = pd.to_datetime(df.Date)
    # Set the date as index
    df.set_index('Date', inplace=True)
    # Removed commas from volume and market cap
    df['Volume'] = df['Volume'].str.replace(",", "")
    df['Market_Cap'] = df['Market_Cap'].str.replace(",", "")

    # Convert all fields to numeric, if it cant be done than replace with NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    return df


def plot_data(df):
    for col in df.columns:
        plt.figure()
        df[col].plot(logy=True)
        plt.title(col)


df = load_data()
df = clean_df(df)
plot_data(df)


#save_path = '/Users/richarddo/Documents/github/Metis/Projects/Project_2_Luther/xrp_data_clean2.csv'
# df.to_csv(save_path)
