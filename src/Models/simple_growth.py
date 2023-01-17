#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os, sys, importlib


# In[ ]:


sys.path.append("../Functions")


# In[ ]:


from preprocessing_functions import timeseries_type


# In[ ]:


import numpy as np
import math
import pandas as pd
from IPython.display import clear_output


# In[5]:


def rolling_forecast_LY_perc_inc(time_series, window_end_date, nsteps):

    # This mode just takes the value from the same time last year and uplifts it to account for trend.
    # Note the notion of a window to train on isn't applicable here. Window_end_date still specifies the
    # point we'll start making predictions from.

    ts_dummy = time_series.copy()
    date_col = ts_dummy.columns.values.tolist()[0]
    series_col = ts_dummy.columns.values.tolist()[1]
    final_index = ts_dummy[ts_dummy[date_col] == window_end_date].index[0]
    final_posn = final_index + 1

    # deduce series type
    seasonal_lag, offset, freq = timeseries_type(time_series)

    prediction_column = "LY_growth" + "_nsteps_" + str(nsteps)
    ts_dummy[prediction_column] = np.nan
    column_number = ts_dummy.columns.get_loc(prediction_column)

    latest_training_datapoint = ts_dummy[date_col].iloc[final_index]
    loops_required = math.floor(1 + (len(ts_dummy) - final_posn) / nsteps)
    loop = 0

    prediction = pd.DataFrame(columns=["preds"], index=range(0, nsteps))

    while final_posn <= len(ts_dummy):

        latest_training_datapoint = ts_dummy[date_col].iloc[final_index]

        forecasts_made = 0
        for j in range(0, nsteps):
            if j == 0:
                last_value = ts_dummy[series_col].iloc[final_index]
            else:
                last_value = prediction["preds"].iloc[j - 1]
            for k in [0, 1]:
                if k == 0 and ((forecasts_made - seasonal_lag - k) < 0):
                    growth_numerator = ts_dummy.iloc[
                        (final_index + forecasts_made + 1) - seasonal_lag, 1
                    ]
                elif k == 0 and ((forecasts_made - seasonal_lag - k) >= 0):
                    growth_numerator = prediction["preds"].iloc[
                        forecasts_made - seasonal_lag
                    ]
                elif k == 1 and ((forecasts_made - seasonal_lag - k) < 0):
                    growth_denominator = ts_dummy.iloc[
                        (final_index + forecasts_made + 1) - seasonal_lag - 1, 1
                    ]
                elif k == 1 and ((forecasts_made - seasonal_lag - k) >= 0):
                    growth_denominator = prediction["preds"].iloc[
                        forecasts_made - seasonal_lag - 1
                    ]

            growth = growth_numerator / growth_denominator
            prediction["preds"].iloc[j] = growth * last_value
            forecasts_made = forecasts_made + 1

        for i in range(0, min(nsteps, len(ts_dummy) - final_posn)):
            ts_dummy.iloc[final_index + i + 1, column_number] = prediction[
                "preds"
            ].iloc[i]

        final_index = final_index + nsteps
        final_posn = final_index + 1

        loop = loop + 1
        clear_output()
        print(str(round(100 * loop / loops_required)) + " % done")

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
        forecasts[forecast_column].iloc[i] = prediction["preds"].iloc[i]

    return (ts_dummy, forecasts, prediction_column)


# In[6]:
