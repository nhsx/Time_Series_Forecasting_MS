#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


def weekly_distribution(data, series_name):

    """
    Groups a given time series by day of week in order to compute what % of the total aggregated
    values occur on a given day.

    INPUTS:
    ----------------
        data (pandas.core.frame.DataFrame): The raw data

        series_name (int) : The series within the raw data to group over.

    RETURNS:
    ----------------
        (pandas.core.frame.DataFrame): A dataframe with the time series grouped by dow and
        averaged by the series total.

    """

    dummy_data = data.copy()
    date_col = dummy_data.columns.values.tolist()[0]
    dummy_data[date_col] = pd.to_datetime(dummy_data[date_col], format="%d/%m/%Y")
    dummy_data = dummy_data.groupby(
        dummy_data[date_col].dt.day_name()
    ).sum()  # groupby dow

    for (
        col
    ) in (
        dummy_data.columns.values.tolist()
    ):  # divide by the column total to get the distribution as a %.
        col_sum = dummy_data[col].sum()
        dummy_data[col] = dummy_data[col] / col_sum

    dummy_data = dummy_data.reset_index()

    cats = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    dummy_data[date_col] = pd.Categorical(
        dummy_data[date_col], categories=cats, ordered=True
    )  # order the df by dow starting from monday
    dummy_data = dummy_data.sort_values(date_col).reset_index().drop("index", axis=1)

    dummy_data = dummy_data[[date_col, series_name]]

    return dummy_data


# In[3]:


def weekly_to_monthly_summary(time_series, weekly_distribution):

    """
    Takes a weekly time series and aggregates it up to monthly. This function adjusts for weeks
    which overlap across two months.

    INPUTS:
    ----------------
        time_series (pandas.core.frame.DataFrame) : A given time series.

        weekly_distribution (pandas.core.frame.DataFrame) : The output of the weekly_distribution
        function for the given time series.

    RETURNS:
    ----------------
        (pandas.core.frame.DataFrame): An (approximate) monthly aggregate of the weekly time series.

    """

    ts_dummy = time_series.copy().reset_index().drop("index", axis=1)

    date_col = ts_dummy.columns.values.tolist()[0]
    # series_col = ts_dummy.columns.values.tolist()[1]
    prediction_col = ts_dummy.columns.values.tolist()[1]

    series_col = weekly_distribution.columns.values.tolist()[1]

    ts_monthly = ts_dummy.set_index(date_col).resample("M").sum()[[prediction_col]]
    ts_monthly = ts_monthly.reset_index()
    ts_monthly[prediction_col] = 0

    for i in range(len(ts_dummy)):
        end_of_week_date = ts_dummy[date_col].iloc[i]
        end_of_week_month = end_of_week_date.month
        start_of_week_month = (end_of_week_date - pd.DateOffset(days=7)).month

        add_to_this_month = 0
        add_to_last_month = 0

        end_of_month_date = str(pd.Period(end_of_week_date, freq="M").end_time.date())
        month_index = ts_monthly[ts_monthly[date_col] == end_of_month_date].index[0]

        if (
            end_of_week_month == start_of_week_month
        ):  # if the week doesn't overlap multiple months
            add_to_this_month = ts_dummy[prediction_col].iloc[i]
            ts_monthly.iloc[month_index, 1] += add_to_this_month

        else:  # otherwise only add on the portion applicable to this month.
            days_in_last_month = 7 - end_of_week_date.day
            distribution_df = weekly_distribution.copy()
            distribution_df[series_col] = (
                distribution_df[series_col] * ts_dummy[prediction_col].iloc[i]
            )  # apply the distribution to the current week
            for j in range(days_in_last_month):
                add_to_last_month += distribution_df[series_col].iloc[j]
            add_to_this_month = distribution_df[series_col].sum() - add_to_last_month

            ts_monthly.iloc[month_index - 1, 1] += add_to_last_month
            ts_monthly.iloc[month_index, 1] += add_to_this_month

    return ts_monthly


# In[1]:


# write all the above code to a py file but not this particular cell of code.

