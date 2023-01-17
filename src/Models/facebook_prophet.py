#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os, sys, importlib


# In[8]:


sys.path.append("../Functions")


# In[13]:


from prophet import Prophet
from preprocessing_functions import timeseries_type
from holiday_functions import holiday_df


# In[10]:


import numpy as np
import math
import pandas as pd
from IPython.display import clear_output


# In[11]:


def rolling_prophet(time_series, window_end_date, nsteps, window_length=None):

    """
    Takes a time series and computes forecasts on a rolling basis.

    Forecasts are based on Facebook's prophet model. See the mkdocs for an overview of Facebook prophet.

    INPUTS:
    ----------------
        time_series (pandas.core.frame.DataFrame) : A time series

        window_end_date (string) : A date string of the form 'yyyy-mm-dd'. This gives the model the latets piece of data it may
        use to make a forecast.

        nsteps (int) : The number of steps out to forecast

        window_length (int) : If specified, will make sure the rolling window the forecasts are based on is of this length. If
        not specified will just start from the first date in the time series

    RETURNS:
    ----------------
        (tuple): A tuple of the form (a, b, c)

        a (pandas.core.frame.DataFrame) : The original time series now with an additional column of fitted values

        b (pandas.core.frame.DataFrame) : A timeseries containing the final set of forecasts made

        c (string) : A string which contains the title of the model. This includes some details of the set parameters

    """

    ts_dummy = time_series.copy()

    if (
        window_length == None
    ):  # for no set window_length the series just takes all the historic data
        start_index = 0  # i.e it has a start index for the window at time 0.
        length_text = ""
        start_index_increment = (
            0  # and we aren't going to roll the start point forward from 0.
        )
    else:
        start_index = final_index - (
            window_length - 1
        )  # otherwise start the window, so that the length from start to end = window_length
        length_text = "_wl_" + str(window_length)
        start_index_increment = (
            nsteps  # the amount the window rolls forward by in each loop.
        )

    prediction_column = "prophet" + length_text + "_nsteps_" + str(nsteps)
    ts_dummy[prediction_column] = np.nan

    column_number = ts_dummy.columns.get_loc(prediction_column)
    old_date_col = ts_dummy.columns.values.tolist()[0]
    old_series_col = ts_dummy.columns.values.tolist()[1]

    # put the series in the form recogniseable to prophet
    fb_ts_dummy = pd.DataFrame(
        {"ds": ts_dummy[old_date_col].tolist(), "y": ts_dummy[old_series_col].tolist()}
    )
    new_date_col = fb_ts_dummy.columns.values.tolist()[0]
    new_series_col = fb_ts_dummy.columns.values.tolist()[1]

    final_index = fb_ts_dummy[fb_ts_dummy[new_date_col] == window_end_date].index[
        0
    ]  # get the final index the rolling window goes up to
    final_posn = final_index + 1

    # deduce series type
    seasonal_lag, offset, freq = timeseries_type(time_series)

    fb_hols = holiday_df(freq)

    # how many loops we need
    loops_required = math.floor(1 + (len(ts_dummy) - final_posn) / nsteps)
    loop = 0

    while final_posn <= len(fb_ts_dummy):

        rolling_window = fb_ts_dummy.iloc[start_index : final_index + 1]

        latest_training_datapoint = fb_ts_dummy[new_date_col].iloc[final_index]

        model = Prophet(yearly_seasonality=True, holidays=fb_hols)

        # model.fit(rolling_window, iter = 500)
        model.fit(rolling_window)

        # create a future data frame
        future = model.make_future_dataframe(periods=nsteps, freq=freq)
        prediction = model.predict(future)

        prediction = prediction[["yhat", "yhat_lower", "yhat_upper"]].tail(nsteps)

        # fill in predictions as far as the time series goes up to
        for i in range(0, min(nsteps, len(ts_dummy) - final_posn)):
            ts_dummy.iloc[final_index + i + 1, column_number] = prediction.iloc[i, 0]

        # roll the window forward by updating the start and end points of the window
        start_index = start_index + start_index_increment
        final_index = final_index + nsteps
        final_posn = final_index + 1

        # print out completion %
        loop = loop + 1
        clear_output()
        print(str(round(100 * loop / loops_required)) + " % done")

    forecast_column = prediction_column + "_forecasts"
    first_forecast_date = ts_dummy[old_date_col].iloc[-1] + offset
    forecasts = pd.DataFrame(
        {
            "End date": pd.date_range(
                start=first_forecast_date, periods=nsteps, freq=freq
            ),
            forecast_column: np.nan,
        }
    )
    for i in range(nsteps):
        forecasts[forecast_column].iloc[i] = prediction.iloc[i, 0]

    # put the lower and upper CI's for each prediction into a dataframe
    lower_CI_column = prediction_column + "_lower_CI"
    upper_CI_column = prediction_column + "_upper_CI"
    CI = pd.DataFrame(
        {
            "End date": pd.date_range(
                start=first_forecast_date, periods=nsteps, freq=freq
            ),
            lower_CI_column: np.nan,
            upper_CI_column: np.nan,
        }
    )
    for i in range(nsteps):
        CI[lower_CI_column].iloc[i] = prediction.iloc[i, 1]
        CI[upper_CI_column].iloc[i] = prediction.iloc[i, 2]

    return (ts_dummy, forecasts, prediction_column, CI)


# In[7]:


# write all the above code to a py file but not this particular cell of code.
