#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[12]:


def read_file(filename):

    """
    Reads in the raw data.

    INPUTS:
    ----------------
        filename (string) : The name of the file including the file extension, i.e.
        'filename.csv'. File extension must be a csv.

    RETURNS:
    ----------------
        (pandas.core.frame.DataFrame): A dataframe of the data.

    """

    file_path = "../Data/" + filename
    return pd.read_csv(file_path)


# In[13]:


def preprocess(data, series_name):

    """
    Trims the raw dataframe down to the relevant time series specified by series_name and removes
    any leading zeros in this time series.

    INPUTS:
    ----------------
        data (pandas.core.frame.DataFrame) : The raw data previously read in as a pandas dataframe

        series_name (string) : The name of the series in the dataframe

    RETURNS:
    ----------------
        (pandas.core.frame.DataFrame): A time series.

    """

    dummy_data = data.copy()
    date_col = data.columns.values.tolist()[
        0
    ]  # assumes the datacol is the first column
    dummy_data[date_col] = pd.to_datetime(dummy_data[date_col], format="%d/%m/%Y")
    dummy_data = dummy_data[
        [date_col, series_name]
    ]  # trim to relevant date and series column
    first_nonzero = np.nonzero(dummy_data[series_name].to_list())  # get the first non-zero row
    if first_nonzero[0][0] > 0:
        dummy_data = (
            dummy_data.tail(-first_nonzero[0][0]).reset_index().drop("index", axis=1)
        )  # trim the zero rows
    else:
        pass
    return dummy_data


# In[7]:


def plot_series(time_series):

    """
    Returns a plot of the time series.

    INPUTS:
    ----------------
        time_series (pandas.core.frame.DataFrame): The time series to plot.

    RETURNS:
    ----------------
        (matplotlib.figure.Figure): A matplotlib figure.

    """

    ts_dummy = time_series.copy()
    date_col = ts_dummy.columns.values.tolist()[0]
    series_col = ts_dummy.columns.values.tolist()[1]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set(title=series_col, xlabel="Date", ylabel=series_col)

    ts_dummy = time_series.copy()
    date_col = ts_dummy.columns.values.tolist()[0]
    series_col = ts_dummy.columns.values.tolist()[1]

    plt.plot(ts_dummy[date_col], ts_dummy[series_col])

    # plt.show()

    return fig


# In[15]:


def resample(time_series, frequency):

    """
    Returns a time series aggregated up to a different time frequency. 
    i.e. daily data can be resampled to monthly.

    INPUTS:
    ----------------
        time_series (pandas.core.frame.DataFrame): The time series to aggregate

        frequency (string): Specify the frequency to aggregate up to. 
        E.g. weekly = 'W', monthly = 'M'. For a complete list of frequencies see:
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html

    RETURNS:
    ----------------
        (pandas.core.frame.DataFrame): A time series.

    """

    ts_dummy = time_series.copy()
    date_col = ts_dummy.columns.values.tolist()[0]
    series_col = ts_dummy.columns.values.tolist()[1]

    ts_dummy = ts_dummy.set_index(date_col).resample(frequency).sum()[[series_col]]
    ts_dummy = ts_dummy[ts_dummy[series_col] != 0]  # get rid of zero rows
    ts_dummy = ts_dummy.reset_index()

    return ts_dummy


# In[16]:


def difference(time_series, length):

    """
    Returns a time series with differencing applied. 
    This functions is used to make a non-stationary time series stationary.

    INPUTS:
    ----------------
        time_series (pandas.core.frame.DataFrame) : The time series to difference.

        length (int) : The length of the differencing.

    RETURNS:
    ----------------
        (pandas.core.frame.DataFrame): A time series.

    """

    ts_dummy = time_series.copy()
    series_col = ts_dummy.columns.values.tolist()[1]
    differenced_col = series_col + "_diff_" + str(length)

    ts_dummy[differenced_col] = np.nan

    for i in range(len(ts_dummy)):
        if i - length >= 0:
            ts_dummy[differenced_col].iloc[i] = (
                ts_dummy[series_col].iloc[i] - ts_dummy[series_col].iloc[i - length]
            )
        else:
            pass

    ts_dummy = ts_dummy.drop(series_col, axis=1)

    return ts_dummy


# In[17]:


def timeseries_type(time_series):

    """
    Deduces the frequency of the time series and returns:
    The lag required to capture yearly seasonality.
    An 'offset' value which can be added to the latest date in the time series,
    to get the date of next datapoint.
    A frequency parameter (Daily data = 'D', Weekly data = 'W', Monthly data = 'M')

    INPUTS:
    ----------------
        time_series (pandas.core.frame.DataFrame) : The time series.

    RETURNS:
    ----------------
        (tuple) : A tuple of the form (a,b,c)

        a (int) : The yearly lag parameters

        b (pandas._libs.tslibs.offsets.DateOffset) : The 'offset' value

        c (string) : The frequency parameters

    """

    ts_dummy = time_series.copy()
    date_col = ts_dummy.columns.values.tolist()[0]

    delta = ts_dummy[date_col].iloc[-1] - ts_dummy[date_col].iloc[-2]
    if delta.days == 1:
        seasonal_lag = 364
        offset = pd.DateOffset(days=1)
        freq = "D"
    elif delta.days == 7:
        seasonal_lag = 52
        offset = pd.DateOffset(days=7)
        freq = "W"
    else:
        seasonal_lag = 12
        offset = pd.DateOffset(months=1)
        freq = "M"

    return (seasonal_lag, offset, freq)


# In[4]:


# write all the above code to a py file but not this particular cell of code.

