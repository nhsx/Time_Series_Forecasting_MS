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


def rolling_forecast_holt_winters(
    time_series, window_end_date, model_params, nsteps, window_length=None
):

    # works in pretty much the same way as SARIMA model except now a holt_winters model is called.

    ts_dummy = time_series.copy()
    date_col = ts_dummy.columns.values.tolist()[0]
    series_col = ts_dummy.columns.values.tolist()[1]
    final_index = ts_dummy[ts_dummy[date_col] == window_end_date].index[0]
    final_posn = final_index + 1

    # deduce series type
    seasonal_lag, offset, freq = timeseries_type(time_series)

    if window_length == None:
        start_index = 0
        length_text = ""
        start_index_increment = 0
    else:
        start_index = final_index - (window_length - 1)
        length_text = "_wl_" + str(window_length)
        start_index_increment = nsteps

    prediction_column = "holt_winters" + length_text + "_nsteps_" + str(nsteps)

    ts_dummy[prediction_column] = np.nan
    column_number = ts_dummy.columns.get_loc(prediction_column)
    trend = model_params[0]
    seasonal = model_params[1]
    seasonal_periods = model_params[2]

    latest_training_datapoint = ts_dummy[date_col].iloc[final_index]
    loops_required = math.floor(1 + (len(ts_dummy) - final_posn) / nsteps)
    loop = 0

    while final_posn <= len(ts_dummy):

        rolling_window = ts_dummy.iloc[start_index : final_index + 1]
        latest_training_datapoint = ts_dummy[date_col].iloc[final_index]

        model = ExponentialSmoothing(
            rolling_window[series_col],
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
        )
        fitted_model = model.fit()
        prediction = fitted_model.forecast(nsteps)

        for i in range(0, min(nsteps, len(ts_dummy) - final_posn)):
            ts_dummy.iloc[final_index + i + 1, column_number] = prediction[
                final_index + i + 1
            ]

        start_index = start_index + start_index_increment
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
        forecasts[forecast_column].iloc[i] = prediction.iloc[i]

    return (ts_dummy, forecasts, prediction_column)


# In[6]:
