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


def rolling_forecast_lagged_average(
    time_series, window_end_date, lag, num_lags, nsteps
):

    # This mode just take a historic value as a prediction. Note the notion of a window to train on
    # isn't applicable here. Window_end_date still specifies the point we'll start making predictions from.

    ts_dummy = time_series.copy()
    date_col = ts_dummy.columns.values.tolist()[0]
    series_col = ts_dummy.columns.values.tolist()[1]
    final_index = ts_dummy[ts_dummy[date_col] == window_end_date].index[0]
    final_posn = final_index + 1

    # deduce series type
    seasonal_lag, offset, freq = timeseries_type(time_series)

    prediction_column = "num_lags_" + str(num_lags) + "_nsteps_" + str(nsteps)
    ts_dummy[prediction_column] = np.nan
    column_number = ts_dummy.columns.get_loc(prediction_column)

    latest_training_datapoint = ts_dummy[date_col].iloc[final_index]
    loops_required = math.floor(1 + (len(ts_dummy) - final_posn) / nsteps)
    loop = 0

    prediction = pd.DataFrame(columns=["preds"], index=range(0, nsteps))

    while final_posn <= len(ts_dummy):

        latest_training_datapoint = ts_dummy[date_col].iloc[final_index]

        # make the forecasts
        sum = 0
        forecasts_made = 0
        for j in range(0, nsteps):
            for k in range(num_lags):
                if (forecasts_made - (k + 1) * lag) < 0:
                    sum = (
                        sum
                        + ts_dummy.iloc[
                            (final_index + forecasts_made) - ((k + 1) * lag - 1), 1
                        ]
                    )
                else:
                    sum = (
                        sum + prediction["preds"].iloc[(forecasts_made) - (k + 1) * lag]
                    )
            average = sum / num_lags
            prediction["preds"].iloc[j] = average
            sum = 0
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
