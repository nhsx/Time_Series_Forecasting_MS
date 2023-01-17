#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[5]:


def n_weeks_forecast(df, nsteps, n_week_sum):

    """
    Uses historic data to compare how an aggregated n week out forecast compares to
    the actual historic aggregates, given we have m < n pieces of data already.
    For example, by setting nsteps = 1, and n_week_sum = 4, we are comparing 4
    week aggregates against an aggregated forecast over same 4 weeks, whereby only the
    last 1 week is actually forecast out. The previous 3 weeks are assumed known.
    This function allows us to see how an aggregated 4 week out forecast will improve
    over time with each new week of data that comes in.

    INPUTS:
    ----------------
        df (pandas.core.frame.DataFrame) : A dataframe of the actual time series values
        aswell as a column of values which have been forecast out nsteps ahead by one of
        our models. Such a dataframe is returned by each of our models.

        nsteps (int) : The number of steps ahead we are forecasting out.

        n_week_sum (int) : The number of weeks we aggregate over.

    RETURNS:
    ----------------
        (pandas.core.frame.DataFrame): A dataframe with a column of historic aggregated
        values and a column of forecasts.

    """

    ts_dummy = df.copy()
    date_col = ts_dummy.columns.values.tolist()[0]
    series_col = ts_dummy.columns.values.tolist()[1]
    pred_col = ts_dummy.columns.values.tolist()[2]

    cols = ["forecast_end_date", "actual_" + str(n_week_sum) + "_week_sum", pred_col]
    forecast_df = pd.DataFrame(columns=cols)

    # get the first index where the predicted data starts at
    final_training_index = ts_dummy[pred_col].first_valid_index() - 1
    first_forecast_index = final_training_index + 1

    while (final_training_index + nsteps) <= (len(ts_dummy) - 1):

        predicted_value = 0
        actual_value = 0
        look_back = n_week_sum - nsteps

        # if we make a 2 step ahead forcast on on a date D, then the actual 4 week total 
        #is D-1, D, D+1 and D+2.
        for i in range(look_back):
            predicted_value = (
                predicted_value + ts_dummy[series_col].iloc[final_training_index - i]
            )
            actual_value = (
                actual_value + ts_dummy[series_col].iloc[final_training_index - i]
            )
        for j in range(nsteps):
            predicted_value = (
                predicted_value + ts_dummy[pred_col].iloc[first_forecast_index + j]
            )
            actual_value = (
                actual_value + ts_dummy[series_col].iloc[first_forecast_index + j]
            )

        # Create a new dataframe of the forecasts
        actual_end_date = ts_dummy[date_col].iloc[final_training_index + nsteps]
        vals = [actual_end_date, actual_value, predicted_value]
        # forecast_df = forecast_df.append(dict(zip(cols,vals)), ignore_index = True)
        forecast_df = pd.concat(
            [forecast_df, pd.DataFrame.from_records([dict(zip(cols, vals))])],
            ignore_index=True,
        )

        final_training_index = final_training_index + nsteps
        first_forecast_index = final_training_index + 1

    return forecast_df


# In[6]:


def performance_metrics(df, performance_lag, summarise=True):

    """
    Computes performance metrics (mae, mape, rmse) for the historic forecasts of a time series.

    INPUTS:
    ----------------
        df (pandas.core.frame.DataFrame) : A dataframe of the actual time series values
        aswell as a column of values which have been forecast by one of our models.
        Such a dataframe is returned by each of our models.

        performance_lag (int) : Only use this many pieces of the latest data to compute
        the performance metrics.

        summarise (bool) : Whether to leave the performance metrics in their raw form,
        i.e broken down by date (bool = False). Or whether to average all these
        performance metrics to get a single set of of performance metrics (bool = True).

    RETURNS:
    ----------------
        (pandas.core.frame.DataFrame): A dataframe of the performance metrics of the model.

    """

    ts_dummy = df.copy()

    series_col = ts_dummy.columns.values.tolist()[1]
    preds_col = ts_dummy.columns.values.tolist()[2]

    ts_dummy[series_col] = pd.to_numeric(ts_dummy[series_col])
    ts_dummy[preds_col] = pd.to_numeric(ts_dummy[preds_col])

    # add in the columns required for the first step of the calc
    ts_dummy["perc error"] = np.nan
    ts_dummy["abs error"] = np.nan
    ts_dummy["sq error"] = np.nan

    ts_dummy["perc error"] = (
        abs(ts_dummy[series_col] - ts_dummy[preds_col]) / ts_dummy[series_col]
    )
    ts_dummy["abs error"] = abs(ts_dummy[series_col] - ts_dummy[preds_col])
    ts_dummy["sq error"] = abs(ts_dummy[series_col] - ts_dummy[preds_col]) ** 2

    ts_dummy = ts_dummy.tail(performance_lag)

    if summarise is False:  # return raw performance metrics unless specifief otherwise
        return ts_dummy
    else:
        # add in the columns required for the second step of the calc if summaraising is required.
        mae = ts_dummy["abs error"].mean()
        rmse = (ts_dummy["sq error"].mean()) ** 0.5
        mape = ts_dummy["perc error"].mean()

        df_data = {"model": preds_col, "mae": mae, "rmse": rmse, "mape": mape}
        metrics_df = pd.DataFrame(data=df_data, index=[0])

        return metrics_df


# In[7]:


def performance_merger(performance_dfs, metric):

    """
    Merges together a list of performance dataframes, so models can be easily compared at a glance.

    INPUTS:
    ----------------
        performance_dfs (list) : A list of performance dataframes. These dataframes are
        the outputs of the performance_metrics function. See the performance_metrics
        documentation for details.

        metric (string) : The metric we wants to compare the models over. This can
        either be 'mae', mape' or 'rmse'.

    RETURNS:
    ----------------
        (pandas.core.frame.DataFrame): A dataframe of the performance metrics for each
        of the different models.

    """
    
    date_col = performance_dfs[0].columns.values.tolist()[0]
    merged_df = performance_dfs[1][[date_col]]

    for i in range(len(performance_dfs)):
        model_name = performance_dfs[i].columns.values.tolist()[2]
        trimmed_df = performance_dfs[i][[date_col, metric]].rename(
            columns={metric: model_name}
        )
        merged_df = merged_df.merge(trimmed_df, how="inner", on=[date_col])

    return merged_df


# In[8]:


def error_df(model_performance_dfs, number_of_known_weeks, metric):

    """
    Takes a list of dataframes, each of which summarises performance across a number of models.
    These dataframes are assumed to have been constructed by first passing outputs of the
    n_weeks_forecast function into the performance_metrics function. Therefore each
    dataframe is based on knowing a specified number of weeks. This function returns a
    dataframe whose rows are the number of known weeks and each column is a model. Each
    value is the specified performance metric.

    INPUTS:
    ----------------
        model_performance_dfs (list) : A list of dataframes, each of which is summarises
        performance across a number of models.

        number_of_known_weeks (int) : The number of known weeks for each dataframe in the list.
        Note, the order of the number_of_known_weeks list should correspond to the ordering of
        the dataframes. i.e. if the first dataframe is based on knowing 0 weeks of data, then
        the first entry of number_of_known_weeks list should be 0.

        metric (string) : The metric we wants to compare the models over.
        This can either be 'mae', mape' or 'rmse'

    RETURNS:
    ----------------
        (pandas.core.frame.DataFrame): A dataframe whose rows are the number of known weeks
        and each column is a model. Each value is the specified performance metric.

    """

    metric_posn = model_performance_dfs[0].columns.get_loc(metric)
    error_reduction_df = pd.DataFrame({"number of known weeks": number_of_known_weeks})
    for i in range(len(model_performance_dfs[0])):
        error_reduction_df[str(model_performance_dfs[0]["model"].iloc[i])] = np.nan

    for col in range(len(error_reduction_df.columns.values.tolist()) - 1):
        for row in range(4):
            error_reduction_df.iloc[row, col + 1] = model_performance_dfs[row].iloc[
                col, metric_posn
            ]

    return error_reduction_df


# In[5]:


# write all the above code to a py file but not this particular cell of code.

