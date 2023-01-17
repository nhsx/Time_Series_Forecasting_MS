#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
from govuk_bank_holidays.bank_holidays import BankHolidays
import holidays


# In[13]:


# create a list of bank holidays from 2021 through to 2025
# remember bank holiday only accounts for things happening midweek!
def bank_holiday_df():

    """
    Creates a bankholiday dataframe. Each row is a date, and each column is a day of the week.
    The values in the dataframe are binary indicators which indicate which day of the week the
    bank holiday falls on. If they are all 0, the date is not a bankholiday.

    RETURNS:
    ----------------
        (pandas.core.frame.DataFrame): A dataframe.

    """

    bank_holidays = BankHolidays()

    start_date = "2021-01-01"
    holidays_df = pd.DataFrame({"Date": pd.date_range(start_date, periods=365 * 5)})
    holidays_df["dow"] = holidays_df["Date"].dt.day_name()
    holidays_df["holiday_name"] = np.nan

    # Create columns which will be binary indicators of which day a particular BH occurs on.
    holidays_df["mon_bh"] = 0
    holidays_df["tues_bh"] = 0
    holidays_df["weds_bh"] = 0
    holidays_df["thurs_bh"] = 0
    holidays_df["fri_bh"] = 0
    holidays_df["sat_bh"] = 0
    holidays_df["sun_bh"] = 0
    holidays_df["christmas_bh"] = 0
    holidays_df["boxing_bh"] = 0
    holidays_df["new_years_bh"] = 0

    # depending which dow the BH occurs on get the location of the correct binary indicator column
    dow_map = {
        "Monday": holidays_df.columns.get_loc("mon_bh"),
        "Tuesday": holidays_df.columns.get_loc("tues_bh"),
        "Wednesday": holidays_df.columns.get_loc("weds_bh"),
        "Thursday": holidays_df.columns.get_loc("thurs_bh"),
        "Friday": holidays_df.columns.get_loc("fri_bh"),
        "Saturday": holidays_df.columns.get_loc("sat_bh"),
        "Sunday": holidays_df.columns.get_loc("sun_bh"),
    }

    # Note division must be specified otherwise BH returns incomplete list for England and Wales
    for bank_holiday in bank_holidays.get_holidays(
        division=BankHolidays.ENGLAND_AND_WALES
    ):
        if 2021 <= bank_holiday["date"].year <= 2025:
            holiday_index = holidays_df[
                holidays_df["Date"] == str(bank_holiday["date"])
            ].index[0]
            holidays_df.iloc[holiday_index, 2] = bank_holiday["title"]

    # code to set the binary indicator to 1 in the relevant column
    for i in range(len(holidays_df)):
        holiday_name = str(holidays_df.iloc[i]["holiday_name"])
        dow = holidays_df.iloc[i]["dow"]
        if holiday_name != "nan":
            if "Christmas" in holiday_name:
                holidays_df.iloc[i, holidays_df.columns.get_loc("christmas_bh")] = 1
            elif "New" in holiday_name:
                holidays_df.iloc[i, holidays_df.columns.get_loc("new_years_bh")] = 1
            elif "Boxing" in holiday_name:
                holidays_df.iloc[i, holidays_df.columns.get_loc("boxing_bh")] = 1
            else:
                holidays_df.iloc[i, dow_map[dow]] = 1

    return holidays_df


# In[14]:


def holiday_df(freq):

    """
    Creates a holidays dataframe compatible with the facebook prophet model. This
    dataframe has a column of dates and a column of the holidays which occur on that
    particular date.

    INPUTS:
    ----------------
        frequency (string) : The frequency of the holiday dataframe. E.g. freq = 'W',
        will make the date column weekly.

    RETURNS:
    ----------------
        (pandas.core.frame.DataFrame): A dataframe.

    """

    fb_hols = pd.DataFrame(columns=["ds", "holiday"])
    years = [2021, 2022, 2023, 2024, 2025]
    hol_data = []
    for year in years:
        for i, j in holidays.UnitedKingdom(years=year, subdiv="England").items():
            row = {"ds": str(pd.Period(i, freq=freq).end_time.date()), "holiday": j}
            hol_data.append(row)

    fb_hols = pd.DataFrame(hol_data)
    return fb_hols


# In[3]:


# write all the above code to a py file but not this particular cell of code.

