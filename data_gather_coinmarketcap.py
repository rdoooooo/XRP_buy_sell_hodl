from selenium import webdriver
import time
import os
import numpy as np
import pandas as pd

url = 'https://coinmarketcap.com/currencies/ripple/#charts'

# Build the chrome driver with incognito


def build_driver(url):
    # Create new instance of Chrome in Incognito mode

    # First we start by adding the incognito argument to our webdriver.
    option = webdriver.ChromeOptions()
    option.add_argument("incognito")

    # Create a new instance of chrome
    driver = webdriver.Chrome(
        executable_path='/usr/local/bin/chromedriver/', options=option)
    driver.get(url)

    return driver


def click_historical_tab(driver):
    historical_tag = '//a[contains(text(), "Historical Data")]'
    # Boolean for while loop
    found_it = 0
    # Counter to have a limit times of action
    total_counts = 0
    while found_it == 0 and total_counts < 10:
        # Try to click the button, if you cant scroll down
        try:
            driver.find_elements_by_xpath(historical_tag)[0].click()
            found_it = 1
            print('Historical tab selected')
        except:
            driver.execute_script("window.scrollBy(0, 200)")
            wait()
            total_counts += 1
    return


def click_date_range(driver):
    report_range_tag = '//div[contains(@id, "reportrange")]'
    all_time_tag = '//li[contains(@data-range-key,"All Time")]'
    # Boolean for while loop
    found_it = 0
    # Counter to have a limit times of action
    total_counts = 0
    while found_it == 0 and total_counts < 10:
        # Try to click the button, if you cant scroll down
        try:
            driver.find_elements_by_xpath(report_range_tag)[0].click()
            found_it = 1
            print('Date range selected')
            driver.find_elements_by_xpath(all_time_tag)[0].click()
            print('All time selected')
        except:
            driver.execute_script("window.scrollBy(0, 200)")
            wait()
            total_counts += 1
    return


def wait(t=2):
    time.sleep(t)
    return


def get_table_data(driver):
    # Get table
    table_tag = '//table[@class="table"]'
    table_id = driver.find_elements_by_xpath(table_tag)
    table_rows = table_id[0].find_elements_by_tag_name('tr')
    table_columns = table_id[0].find_elements_by_tag_name('th')
    print('Found table data')
    return table_rows


def build_df(table_rows):
    # Compile all the rows
    table = []
    for row in table_rows:
        date = " ".join(row.text.split(' ')[:3])
        clean_row = [date] + row.text.split(' ')[3:]
        table.append(clean_row)

    # Get the columns
    columns_temp = np.array(str(" ".join(table[0])).split()[0:6])
    market_cap = str(" ".join(" ".join(table[0]).split()[6::]))
    columns = np.append(columns_temp, market_cap)

    # Build the DataFrame
    df = pd.DataFrame(data=table)
    df = df.iloc[1:]
    df.columns = columns

    return df


# Builds a driver at the web link
driver = build_driver(url=url)
wait()
click_historical_tab(driver)
click_date_range(driver)
table_rows = get_table_data(driver)
df = build_df(table_rows)
df.head()
#save_path = '/Users/richarddo/Documents/github/Metis/Projects/Project_2_Luther/xrp_data2.csv'
df.to_csv(save_path)
