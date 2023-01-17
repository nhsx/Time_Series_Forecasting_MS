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


def rolling_forecast_ensemble(model_outputs):

    # This model combines n model outputs and takes their average.

    # get a date and series column
    ts_dummy = model_outputs[0][0].iloc[:, 0:2].copy()

    # make a prediction column where the 'ensemble' predictions will go.
    date_col = ts_dummy.columns.values.tolist()[0]
    prediction_column = str(len(model_outputs)) + "_Model_Ensemble"
    ts_dummy[prediction_column] = 0

    # deduce series type
    seasonal_lag, offset, freq = timeseries_type(ts_dummy)

    # deduce nsteps based on how far out the models are forecasting
    nsteps = len(model_outputs[0][1])

    forecast_column = prediction_column + "_forecasts"
    first_forecast_date = ts_dummy[date_col].iloc[-1] + offset

    forecasts = pd.DataFrame(
        {
            "End date": pd.date_range(
                start=first_forecast_date, periods=nsteps, freq=freq
            ),
            forecast_column: np.zeros(nsteps),
        }
    )
    print(forecasts)
    loop = 0
    loops_required = len(model_outputs)

    # add up all the outputs/forecasts
    for model in model_outputs:
        ts_dummy[prediction_column] = ts_dummy[prediction_column] + model[0].iloc[:, 2]
        forecasts[forecast_column] = forecasts[forecast_column] + +model[1].iloc[:, 1]

        loop = loop + 1
        clear_output()
        print(str(round(100 * loop / loops_required)) + " % done")

    # average all the outputs/forecasts
    ts_dummy[prediction_column] = ts_dummy[prediction_column] / len(model_outputs)
    forecasts[forecast_column] = forecasts[forecast_column] / len(model_outputs)

    return (ts_dummy, forecasts, prediction_column)


# In[6]:
