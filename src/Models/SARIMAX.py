#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os, sys, importlib


# In[3]:


sys.path.append("../Functions")


# In[4]:


from preprocessing_functions import timeseries_type


# In[14]:


from holiday_functions import bank_holiday_df


# In[23]:


importlib.reload(sys.modules["holiday_functions"])


# In[11]:


import numpy as np
import math
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from IPython.display import clear_output


# In[24]:


def rolling_forecast_SARIMA(
    time_series,
    window_end_date,
    model_params,
    bh_adj,
    bh_scale,
    nsteps,
    window_length=None,
):

    """
    Takes a time series and computes forecasts on a rolling basis.

    Forecasts are based on a SARIMAX model. See the mkdocs for an overview of SARIMA models.

    INPUTS:
    ----------------
        time_series (pandas.core.frame.DataFrame) : A time series

        window_end_date (string) : A date string of the form 'yyyy-mm-dd'. This gives the model the latets piece of data it may
        use to make a forecast.

        model_params (list) : A list of the SARIMA parameters [p,d,q,P,D,Q,S].

        bh_adj (bool) : If true, will add bankholidays to the SARIMA model

        bh_scale (bool) : If true, will scale the bank holiday binary indicators according to the level of the series, to
        account for the fact that in the dip on bank holidays is naturally bigger when more users are using the app.
        Note, bh_adj must be True for this to be implemented.

        nsteps (int) : The number of steps out to forecast

        window_length (int) : If specified, will make sure the rolling window the forecasts are based on is of this length. If
        not specified will just start from the first date in the time series

    RETURNS:
    ----------------
        (tuple): A tuple of the form (a, b, c, d)

        a (pandas.core.frame.DataFrame) : The original time series now with an additional column of fitted values

        b (pandas.core.frame.DataFrame) : A timeseries containing the final set of forecasts made

        c (string) : A string which contains the title of the model. This includes some details of the set parameters

        d (pandas.core.frame.DataFrame) : A dataframe of upper and lower 95% confidence interval bounds for the forecasts
    """

    ts_dummy = time_series.copy()
    date_col = ts_dummy.columns.values.tolist()[0]
    series_col = ts_dummy.columns.values.tolist()[1]
    final_index = ts_dummy[ts_dummy[date_col] == window_end_date].index[
        0
    ]  # get the final index the rolling window goes up to
    final_posn = final_index + 1

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

    # deduce series type
    seasonal_lag, offset, freq = timeseries_type(time_series)

    p = model_params[0]
    d = model_params[1]
    q = model_params[2]
    P = model_params[3]
    D = model_params[4]
    Q = model_params[5]
    S = model_params[6]

    params = (
        "("
        + str(p)
        + ","
        + str(d)
        + ","
        + str(q)
        + ")("
        + str(P)
        + ","
        + str(D)
        + ","
        + str(Q)
        + ")("
        + str(S)
        + ")"
    )

    if bh_adj == True:
        # resample bank holidays to align to series frequency and trim away old BH data
        bank_holidays = bank_holiday_df()
        bank_holidays = (
            bank_holidays.set_index("Date")
            .resample(freq)
            .sum()[
                [
                    "mon_bh",
                    "tues_bh",
                    "weds_bh",
                    "thurs_bh",
                    "fri_bh",
                    "sat_bh",
                    "sun_bh",
                    "christmas_bh",
                    "boxing_bh",
                    "new_years_bh",
                ]
            ]
        )
        bank_holidays = bank_holidays.reset_index()
        bank_holidays = (
            bank_holidays[bank_holidays["Date"] >= str(ts_dummy.iloc[0, 0])]
            .reset_index()
            .drop("index", axis=1)
        )
        # set the name of the column the rolling predictions will go in
        if bh_scale == True:
            prediction_column = (
                params + "_bh_scale" + length_text + "_nsteps_" + str(nsteps)
            )
        else:
            prediction_column = params + "_bh" + length_text + "_nsteps_" + str(nsteps)
    else:
        prediction_column = params + length_text + "_nsteps_" + str(nsteps)

    # initialise prediction column to nan and get model parameters
    ts_dummy[prediction_column] = np.nan
    column_number = ts_dummy.columns.get_loc(prediction_column)

    # how many loops we need
    loops_required = math.floor(1 + (len(ts_dummy) - final_posn) / nsteps)
    loop = 0

    while final_posn <= len(ts_dummy):

        rolling_window = ts_dummy.iloc[start_index : final_index + 1]
        latest_training_datapoint = ts_dummy[date_col].iloc[final_index]

        # get rolling window for BH's and get the BH df needed for an nstep ahead forecast
        if bh_adj == True:
            xreg_rolling_window = bank_holidays.iloc[
                start_index : final_index + 1
            ].copy()
            xreg_forecast = bank_holidays.iloc[
                final_index + 1 : final_index + nsteps + 1
            ].copy()

            if bh_scale == True:
                for j in range(len(xreg_rolling_window)):
                    for k in range(1, len(xreg_rolling_window.columns)):
                        xreg_rolling_window.iloc[j, k] = (
                            xreg_rolling_window.iloc[j, k]
                            * rolling_window.iloc[j][series_col]
                        )
                for k in range(1, len(xreg_forecast.columns)):
                    xreg_forecast.iloc[:, k] = (
                        xreg_forecast.iloc[:, k]
                        * rolling_window.iloc[final_index][series_col]
                    )

            xreg_rolling_window.drop("Date", axis=1, inplace=True)
            xreg_forecast.drop("Date", axis=1, inplace=True)
        else:
            xreg_rolling_window = None
            xreg_forecast = None

        # fit model
        model = SARIMAX(
            rolling_window[series_col],
            order=(p, d, q),
            seasonal_order=(P, D, Q, S),
            enforce_stationarity=False,
            exog=xreg_rolling_window,
        )
        fitted_model = model.fit(disp=0)
        prediction = fitted_model.get_forecast(
            steps=nsteps, exog=xreg_forecast
        ).summary_frame(alpha=0.05)

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

    # the final preditions are put into a forecast dataframe
    forecast_column = prediction_column + "_forecasts"
    first_forecast_date = ts_dummy[date_col].iloc[-1] + offset
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
        CI[lower_CI_column].iloc[i] = prediction.iloc[i, 2]
        CI[upper_CI_column].iloc[i] = prediction.iloc[i, 3]

    return (ts_dummy, forecasts, prediction_column, CI)


# In[25]:


# write all the above code to a py file but not this particular cell of code.

