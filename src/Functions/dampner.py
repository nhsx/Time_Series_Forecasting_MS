#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt
import scipy


# In[2]:


from preprocessing_functions import preprocess


# In[18]:


def recip_log(x, p, q):

    """
    A reciprocal log function.

    INPUTS:
    ----------------
        x (float) : The input to the function.

        p (float) : A parameter of the model

        q (float) : A parameter of the model

    RETURNS:
    ----------------
        (float): The output when the input x is passed through the function.

    """

    return 1 - p * np.reciprocal(np.log(q * x))


# In[19]:


def fit(user_data, n, plot=False):

    """
    Returns fitted parameters p and q of the recipricol log function, based on which parameters
    best approximate the curve of TotalRegisteredUsers as a % of total UK pop over time.
    Only the last n data points are considered for this fit as the curves behaviour more
    recently is significantly different compared to say in the early adoption phase of the app.

    INPUTS:
    ----------------
        data (float) : The raw user data to be read in, which contains the
        'TotalRegisteredUsers' time series.

        n (int) : Only the last n data points are taken into account.

        plot (bool) : If true, will produce a plot of the actual values of TotalRegisteredUsers
        (as a % of total UK pop) vs the values fitted by the reciprocal log function.

    RETURNS:
    ----------------
        (tuple): The fitted parameters p and q.

    """

    user_data = preprocess(user_data, "TotalRegisteredUsers")

    user_data["% of UK pop"] = (
        user_data["TotalRegisteredUsers"] / 44456850
    )  # 44456850 - current estimate of UK population size as of 01/01/2023.

    user_data_df = pd.DataFrame(
        {
            "% registered": user_data["% of UK pop"].tail(n),
            "% registered fitted": np.zeros(n),
        }
    ).reset_index()
    user_data_df = user_data_df.rename(columns={"index": "timestamp"})

    params, covariance = scipy.optimize.curve_fit(
        recip_log, user_data_df["timestamp"], user_data_df["% registered"]
    )  # learn the optimal parameters

    for i in range(len(user_data_df)):
        user_data_df["% registered fitted"].iloc[i] = recip_log(
            user_data_df["timestamp"].iloc[i], params[0], params[1]
        )  # for these optimal parameters get the fitted value

    if plot is True:  # show a plot of the forecasted values if plot parameter is True.
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set(
            title="% of UK Pop registered", xlabel="timestamp", ylabel="% registered"
        )

        ax = plt.plot(
            user_data_df["timestamp"], user_data_df["% registered"], color="blue"
        )
        ax = plt.plot(
            user_data_df["timestamp"], user_data_df["% registered fitted"], color="red"
        )
        
        plt.show()

    return params


# In[20]:


def dampen(user_data, n, plot=False):

    """
    Forecasts out TotalRegisteredUsers as a % of total UK pop.

    INPUTS:
    ----------------
        user_data (float) : The raw user data to be read in, which contains the
        'TotalRegisteredUsers' time series

        n (int) : Only the last n data points are taken into account when fitting the
        recipricol log function which will be used to make the forecasts.

        plot (bool) : If true, will produce a plot of the actual values of TotalRegisteredUsers
        (as a % of total UK pop) vs the values fitted by the reciprocal log function.

    RETURNS:
    ----------------
        (pandas.core.frame.DataFrame) : A dataframe containing the historic values of the
        TotalRegisteredUsers (as a % of total UK pop) as well as the forecasted values.

    """

    user_data = preprocess(user_data, "TotalRegisteredUsers")
    user_data["% of UK pop"] = user_data["TotalRegisteredUsers"] / 44456850

    params = fit(
        user_data, n, plot=plot
    )  # get the optimal parameters for the forecast function

    offset = pd.DateOffset(months=1)
    latest_date = user_data["Date"].iloc[-1]
    latest_index = len(user_data) - 1
    forecast_index = latest_index + 1

    data_list = []  # make a list of the forecasted values
    for i in range(50):
        latest_date = latest_date + offset + MonthEnd(0)
        latest_index = latest_index + 1
        data_dict = {
            "Date": latest_date,
            "% of UK pop": recip_log(latest_index, params[0], params[1]),
        }
        data_list.append(data_dict)

    user_data = pd.concat(
        [user_data, pd.DataFrame.from_records(data_list)], ignore_index=True
    )

    if plot is True:  # show a plot of the forecasted values if plot parameter is True.
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set(title="% UK pop registered", xlabel="Date", ylabel="% Registered")

        ax = plt.plot(
            user_data.iloc[:forecast_index, 0],
            user_data.iloc[:forecast_index, 2],
            color="blue",
        )
        ax = plt.plot(
            user_data.iloc[forecast_index:, 0],
            user_data.iloc[forecast_index:, 2],
            color="green",
        )
        
        plt.show()

    return user_data


# In[22]:


def trend_adjuster(actual_df, forecast_df, uptake_df):

    """
    Returns a dampened forecast of the timeseries accounting for the slowing trend in
    TotalRegisteredUsers (as a % of total UK pop)

    INPUTS:
    ----------------

        actual_df (pandas.core.frame.DataFrame) : A dataframe containing the historic time series data.
        These actual_df's are provided by each model when run over a given time series.

        forecast_df (pandas.core.frame.DataFrame) : A dataframe containing forecasts from a
        given model over a given time series. These forecast_df's are provided by each model
        when run over a given time series.

        uptake_df (pandas.core.frame.DataFrame) : The forecasted out TotalRegisteredUsers as
        a % of total UK pop as returned by the dampen function.

    RETURNS:
    ----------------
        (pandas.core.frame.DataFrame) : A dataframe containing the forecasted values of the
        time series, now dampened.

    """

    actual_date_col = actual_df.columns.values.tolist()[0]
    actual_series_col = actual_df.columns.values.tolist()[1]

    forecast_date_col = forecast_df.columns.values.tolist()[0]
    forecast_series_col = forecast_df.columns.values.tolist()[1]
    adj_forecast_col = forecast_series_col + "_adj"

    uptake_date_col = uptake_df.columns.values.tolist()[0]
    uptake_series_col = uptake_df.columns.values.tolist()[1]

    forecast_df_dummy = forecast_df.copy()
    uptake_df_dummy = uptake_df.copy()
    damped_df = pd.merge(
        forecast_df_dummy,
        uptake_df_dummy,
        how="left",
        left_on=forecast_date_col,
        right_on=uptake_date_col,
    ).drop(
        columns=[uptake_date_col]
    )  # merge the current forecasted values and uptake forecasts into one df.

    # add a few columns we'll need to make the dampened forecast.
    damped_df["val_ly"] = np.nan
    damped_df["monthly_scaler"] = np.nan
    damped_df[adj_forecast_col] = np.nan

    baseline_uptake = min(
        damped_df[uptake_series_col]
    )  # get the baseline uptake value.

    for i in range(len(damped_df)):
        current_date = damped_df[forecast_date_col].iloc[i]
        lagged_date = current_date - pd.DateOffset(months=12)
        if (
            i < 12
        ):  # if we're less than 12 months in then we have to look back at historic data to get LY's values
            damped_df["val_ly"].iloc[i] = actual_df[
                actual_df[actual_date_col] == lagged_date
            ][actual_series_col]
            damped_df["monthly_scaler"].iloc[i] = i + 1
        else:  # if we're atleast 12 months in then we look back at the forecast made 12 months prior
            damped_df["val_ly"].iloc[i] = forecast_df[
                forecast_df[forecast_date_col] == lagged_date
            ][forecast_series_col]
            damped_df["monthly_scaler"].iloc[i] = 12

        current_forecast = damped_df[forecast_series_col].iloc[i]

        val_ly = damped_df["val_ly"].iloc[i]
        current_uptake = damped_df[uptake_series_col].iloc[i]
        monthly_scaler = damped_df["monthly_scaler"].iloc[i]

        inc_on_ly = (
            current_forecast - val_ly
        )  # what's the current jump from last years forecast to our current forecast

        inc_on_ly_adj = inc_on_ly * (
            1 - ((current_uptake - baseline_uptake) / (1 - baseline_uptake))
        )  # adjust this jump by how much the uptake has grown relative to the baseline uptake value.

        delta = inc_on_ly - inc_on_ly_adj  # get the change in jump

        inc_on_ly_new = (
            inc_on_ly - (monthly_scaler / 12) * delta
        )  # scale down the change in jump for early forecasts, which won't need as much dampening 
           #as they're <12 months out from the latets known value.

        if i < 12:
            val_ly_adj = val_ly
        else:
            val_ly_adj = damped_df[adj_forecast_col].iloc[i - 12]

        adjusted_forecast = (
            val_ly_adj + inc_on_ly_new
        )  # get the value LY and add on the adjusted jump.

        damped_df[adj_forecast_col].iloc[i] = adjusted_forecast

    damped_df = damped_df.drop(
        columns=[
            forecast_series_col,
            uptake_series_col,
            "val_ly",
            "monthly_scaler",
        ]
    )

    return damped_df


# In[2]:


# write all the above code to a py file but not this particular cell of code.

