#!/usr/bin/env python
# coding: utf-8

# # Pipeline

# In[85]:


import os, sys, importlib


# In[86]:


# need this so we can import from different folders
sys.path.append("../Functions")


# In[87]:


sys.path.append("../Models")


# In[88]:


from preprocessing_functions import *
from performance_functions import *
from aggregate_functions import *
from dampner import dampen, trend_adjuster


# In[89]:


from SARIMAX import rolling_forecast_SARIMA
from simple_growth import rolling_forecast_LY_perc_inc
from lagged_average import rolling_forecast_lagged_average
from holt_winters import rolling_forecast_holt_winters
from ensemble import rolling_forecast_ensemble


# In[90]:


import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
# supress SettingWithCopyWarning. In future use loc instead of iloc when editing df's


# In[91]:


# the series we want to run the pipeline over
series_name = "dummy_series"

# the data files
data = "dummy_data.csv"
user_data = "TotalRegisteredUsers_dummy.csv"


# In[92]:


app_data = read_file(data)


# In[135]:


app_data.head(5) #time series data must have a date format of dd/mm/yyyy.


# In[93]:


# preprocess and resample into weekly data
ts_data = preprocess(app_data, series_name)
ts_data_weekly = resample(ts_data, "W")
ts_data_weekly = ts_data_weekly.iloc[:-1]


# In[94]:


# where we want to start assessing model performance from.
weekly_measure_from = "2022-03-27"
monthly_measure_from = "2022-03-31"


# ## Which is the best 1 week out model?

# In[95]:


# run the models for 1,2,3 and 4 week out forecasts and add the outputs to a list.

weekly_model_list_nstep_4 = []
weekly_model_list_nstep_3 = []
weekly_model_list_nstep_2 = []
weekly_model_list_nstep_1 = []

weekly_model_list_nstep_i = [
    weekly_model_list_nstep_1,
    weekly_model_list_nstep_2,
    weekly_model_list_nstep_3,
    weekly_model_list_nstep_4,
]


# In[96]:


for i in range(4):

    nsteps = i + 1

    SARIMA = rolling_forecast_SARIMA(
        ts_data_weekly,
        window_end_date=weekly_measure_from,
        model_params=[0, 1, 1, 0, 1, 0, 52],
        bh_adj=False,
        bh_scale=False,
        nsteps=nsteps,
    )
    weekly_model_list_nstep_i[i].append(SARIMA)

    SARIMA_bh = rolling_forecast_SARIMA(
        ts_data_weekly,
        window_end_date=weekly_measure_from,
        model_params=[0, 1, 1, 0, 1, 0, 52],
        bh_adj=True,
        bh_scale=False,
        nsteps=nsteps,
    )
    weekly_model_list_nstep_i[i].append(SARIMA_bh)

    SARIMA_bh_adj = rolling_forecast_SARIMA(
        ts_data_weekly,
        window_end_date=weekly_measure_from,
        model_params=[0, 1, 1, 0, 1, 0, 52],
        bh_adj=True,
        bh_scale=True,
        nsteps=nsteps,
    )
    weekly_model_list_nstep_i[i].append(SARIMA_bh_adj)

    SARIMA_basic = rolling_forecast_SARIMA(
        ts_data_weekly,
        window_end_date=weekly_measure_from,
        model_params=[0, 1, 0, 0, 1, 0, 52],
        bh_adj=False,
        bh_scale=False,
        nsteps=nsteps,
    )
    weekly_model_list_nstep_i[i].append(SARIMA_basic)

    lagged_av_1 = rolling_forecast_lagged_average(
        time_series=ts_data_weekly,
        window_end_date=weekly_measure_from,
        lag=1,
        num_lags=1,
        nsteps=nsteps,
    )
    weekly_model_list_nstep_i[i].append(lagged_av_1)

    lagged_av_3 = rolling_forecast_lagged_average(
        time_series=ts_data_weekly,
        window_end_date=weekly_measure_from,
        lag=1,
        num_lags=3,
        nsteps=nsteps,
    )
    weekly_model_list_nstep_i[i].append(lagged_av_3)

    simple_growth = rolling_forecast_LY_perc_inc(
        time_series=ts_data_weekly, window_end_date=weekly_measure_from, nsteps=nsteps
    )
    weekly_model_list_nstep_i[i].append(simple_growth)

    ensemble = rolling_forecast_ensemble([SARIMA, lagged_av_1])

    weekly_model_list_nstep_i[i].append(ensemble)


# In[97]:


number_of_weekly_models = len(weekly_model_list_nstep_i[0])


# In[98]:


# read in the equivalent prophet outputs. Note the intialisation params, e.g. series names, weekly_measure_from = '2022-03-27' etc...
# also needs to be set in the prophet file, before running it to produce the outputs.

file_path = (
    os.path.split(os.getcwd())[0]
    + "/Prophet Scripts/"
    + series_name
    + "_weekly_prophet_nsteps_1"
)
infile = open(file_path, "rb")
weekly_prophet = pickle.load(infile)
infile.close()
weekly_model_list_nstep_i[0].append(weekly_prophet)


# In[99]:


# see which is the best 1 week out model
performance_df = pd.DataFrame()
for model in weekly_model_list_nstep_i[0]:
    performance_df = pd.concat(
        [performance_df, performance_metrics(model[0], performance_lag=12)], axis=0
    )
performance_df = performance_df.reset_index().drop("index", axis=1)
performance_df


# ###  Historic performance for some of the weekly models

# In[100]:


# get all the historic forecasts into a big dataframe
validation_df = ts_data_weekly.copy()
for model in weekly_model_list_nstep_i[0]:
    validation_df[model[2]] = model[0][model[2]]


# In[101]:


fig, ax = plt.subplots(figsize=(12, 5))

ax.set(
    title=series_name + " Weekly Model Performance", xlabel="Date", ylabel=series_name
)

ax1 = plt.plot(validation_df.iloc[:,0], validation_df[series_name], color="blue")
ax4 = plt.plot(
    validation_df.iloc[:,0], validation_df["num_lags_3_nsteps_1"], color="orange"
)
ax5 = plt.plot(
    validation_df.iloc[:,0], validation_df["(0,1,1)(0,1,0)(52)_nsteps_1"], color="green"
)

plt.show()


# ## 4 week out forecast performance change with each week gained

# In[102]:


# see how a 4 week out forecast improves with 0,1,2 and 3 weeks of data gained.
# weekly_model_list_out_4_nstep_4 means we're forecasting nsteps = 4 ahead so this is where the 0 weeks of gained data goes.
# weekly_model_list_out_4_nstep_3 means we're forecasting nsteps = 3 ahead so this is where the 1 week of gained data goes.
# etc...


weekly_model_list_out_4_nstep_4 = []
weekly_model_list_out_4_nstep_3 = []
weekly_model_list_out_4_nstep_2 = []
weekly_model_list_out_4_nstep_1 = []

weekly_model_list_out_4_nstep_i = [
    weekly_model_list_out_4_nstep_1,
    weekly_model_list_out_4_nstep_2,
    weekly_model_list_out_4_nstep_3,
    weekly_model_list_out_4_nstep_4,
]


# In[103]:


for i in range(4):

    for j in range(number_of_weekly_models):

        nsteps = i + 1

        weekly_model_list_out_4_nstep_i[i].append(
            n_weeks_forecast(
                weekly_model_list_nstep_i[i][j][0], nsteps=nsteps, n_week_sum=4
            )
        )


# In[104]:


# read in the equivalent prophet outputs. Note the intialisation params, e.g. series names, weekly_measure_from = '2022-03-27' etc...
# also needs to be set in the prophet file, before running it to produce the outputs.

file_path = (
    os.path.split(os.getcwd())[0]
    + "/Prophet Scripts/"
    + series_name
    + "_weekly_prophet_out_4_nsteps_i"
)
infile = open(file_path, "rb")
weekly_prophet_out_4_nsteps_i = pickle.load(infile)
infile.close()


# In[105]:


for i in range(len(weekly_model_list_out_4_nstep_i)):
    weekly_model_list_out_4_nstep_i[i].append(weekly_prophet_out_4_nsteps_i[i])


# In[106]:


# for each of the weekly_model_list_out_4_nstep_i, create a model performance dataframe

performance_df_out_4_nsteps_1 = pd.DataFrame()
performance_df_out_4_nsteps_2 = pd.DataFrame()
performance_df_out_4_nsteps_3 = pd.DataFrame()
performance_df_out_4_nsteps_4 = pd.DataFrame()

performance_df_out_4_nsteps_i = [
    performance_df_out_4_nsteps_1,
    performance_df_out_4_nsteps_2,
    performance_df_out_4_nsteps_3,
    performance_df_out_4_nsteps_4,
]


# In[107]:


for i in range(len(performance_df_out_4_nsteps_i)):
    for model in weekly_model_list_out_4_nstep_i[i]:
        performance_df_out_4_nsteps_i[i] = pd.concat(
            [
                performance_df_out_4_nsteps_i[i],
                performance_metrics(model, performance_lag=12),
            ],
            axis=0,
        )
        performance_df_out_4_nsteps_i[i] = (
            performance_df_out_4_nsteps_i[i].reset_index().drop("index", axis=1)
        )


# In[108]:


# as an example here is the model summary with 0 weeks of data gained (same formt as before)
performance_df_out_4_nsteps_i[3]


# In[109]:


# create a df of the mape values for each additional week gained
mape_df = error_df(
    model_performance_dfs=list(reversed(performance_df_out_4_nsteps_i)),
    number_of_known_weeks=[0, 1, 2, 3],
    metric="mape",
)


# In[110]:


# create a df of the rmse values for each additional week gained
rmse_df = error_df(
    model_performance_dfs=list(reversed(performance_df_out_4_nsteps_i)),
    number_of_known_weeks=[0, 1, 2, 3],
    metric="rmse",
)


# In[111]:


fig, ax = plt.subplots(figsize=(12, 5))

ax.set(
    title="Weekly Models Mape Reduction", xlabel="number of known weeks", ylabel="MAPE"
)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.xaxis.set_major_locator(mtick.MultipleLocator(1))

col_names = mape_df.columns.values.tolist()
col_names.remove("number of known weeks")
for col in col_names:
    plt.plot(mape_df["number of known weeks"], mape_df[col], label=col)
plt.legend()
plt.show()


# In[112]:


fig, ax = plt.subplots(figsize=(12, 5))

ax.set(
    title="Weekly Models RMSE Reduction", xlabel="number of known weeks", ylabel="RMSE"
)
ax.xaxis.set_major_locator(mtick.MultipleLocator(1))

col_names = rmse_df.columns.values.tolist()
col_names.remove("number of known weeks")
for col in col_names:
    plt.plot(rmse_df["number of known weeks"], rmse_df[col], label=col)
plt.legend()
plt.show()


# ## A look at the Christmas period by week

# In[113]:


# take the weekly SARIMA model out 13 steps. Note this may not be the best weelky model for every series!
weekly_sarima_nsteps_13 = rolling_forecast_SARIMA(
    ts_data_weekly,
    window_end_date="2022-10-30",
    model_params=[0, 1, 1, 0, 1, 0, 52],
    bh_adj=False,
    bh_scale=False,
    nsteps=13,
)


# In[114]:


fig, ax = plt.subplots(figsize=(12, 5))

ax.set(title=series_name + " Forecasts by Week", xlabel="Date", ylabel=series_name)

ax = plt.plot(
    weekly_sarima_nsteps_13[0].iloc[:, 0].tail(20),
    weekly_sarima_nsteps_13[0].iloc[:, 1].tail(20),
)
ax = plt.plot(
    weekly_sarima_nsteps_13[1].iloc[:, 0], weekly_sarima_nsteps_13[1].iloc[:, 1]
)
plt.fill_between(
    weekly_sarima_nsteps_13[3].iloc[:, 0],
    weekly_sarima_nsteps_13[3].iloc[:, 1],
    weekly_sarima_nsteps_13[3].iloc[:, 2],
    color="r",
    alpha=0.1,
)

plt.show()


# ## Which is the best 1 month out model?

# In[115]:


# use the weekly SARIMA model to make monthly forecasts (4 weeks ~ 1 month)
weekly_SARIMA_nsteps_4 = rolling_forecast_SARIMA(
    ts_data_weekly,
    window_end_date=weekly_measure_from,
    model_params=[0, 1, 1, 0, 1, 0, 52],
    bh_adj=False,
    bh_scale=False,
    nsteps=4,
)


# In[116]:


# to aggregate weekly forecasts into monthly ones we need to know how to distribute the weeks which overlap two months
# into each of the months correctly
distr = weekly_distribution(app_data, series_name)


# In[117]:


# aggregate the weekly forecasts into monthly ones
start = weekly_SARIMA_nsteps_4[0].iloc[:, 2].first_valid_index()
end = len(weekly_SARIMA_nsteps_4[0])
weekly_SARIMA_nsteps_4_trimmed = (
    weekly_SARIMA_nsteps_4[0].iloc[start:end].drop(columns=series_name, axis=1)
)
monthly_forecasts = weekly_to_monthly_summary(weekly_SARIMA_nsteps_4_trimmed, distr)


# In[118]:


# see the performance by month
monthly_forecasts_df = resample(preprocess(app_data, series_name), "M").merge(
    monthly_forecasts, how="inner"
)
pm1 = performance_metrics(monthly_forecasts_df, performance_lag=40, summarise=False)


# In[119]:


# resample the data as monthly
ts_data_monthly = resample(preprocess(app_data, series_name), "M")


# In[120]:


# run some other monthly models as a point of comparison and get their respective performances by month

monthly_SARIMA = rolling_forecast_SARIMA(
    ts_data_monthly,
    window_end_date=monthly_measure_from,
    model_params=[0, 1, 0, 0, 1, 0, 12],
    bh_adj=False,
    bh_scale=False,
    nsteps=1,
)
pm2 = performance_metrics(monthly_SARIMA[0], performance_lag=7, summarise=False)

monthly_lagged_av_1 = rolling_forecast_lagged_average(
    time_series=ts_data_monthly,
    window_end_date=monthly_measure_from,
    lag=1,
    num_lags=1,
    nsteps=1,
)
pm3 = performance_metrics(monthly_lagged_av_1[0], performance_lag=7, summarise=False)

monthly_lagged_av_3 = rolling_forecast_lagged_average(
    time_series=ts_data_monthly,
    window_end_date=monthly_measure_from,
    lag=1,
    num_lags=3,
    nsteps=1,
)
pm4 = performance_metrics(monthly_lagged_av_3[0], performance_lag=7, summarise=False)

monthly_simple_growth = rolling_forecast_LY_perc_inc(
    time_series=ts_data_monthly, window_end_date=monthly_measure_from, nsteps=1
)
pm5 = performance_metrics(monthly_simple_growth[0], performance_lag=7, summarise=False)

monthly_ensemble = rolling_forecast_ensemble([monthly_SARIMA, monthly_lagged_av_1])
pm6 = performance_metrics(monthly_ensemble[0], performance_lag=7, summarise=False)


# In[121]:


# read in the equivalent prophet outputs. Note the intialisation params, e.g. series names, weekly_measure_from = '2022-03-27' etc...
# also needs to be set in the prophet file, before running it to produce the outputs.

file_path = (
    os.path.split(os.getcwd())[0]
    + "/Prophet Scripts/"
    + series_name
    + "_monthly_prophet_forecasts"
)
infile = open(file_path, "rb")
pm_prophet = pickle.load(infile)
infile.close()


# In[122]:


performance_merger([pm1, pm2, pm3, pm4, pm5, pm_prophet, pm6], "perc error")


# ## Which is the best 3 month out model?

# In[123]:


# do exactly the same as before but now forecasting 3 months ahead instead of 1

weekly_SARIMA_nsteps_12 = rolling_forecast_SARIMA(
    ts_data_weekly,
    window_end_date=weekly_measure_from,
    model_params=[0, 1, 1, 0, 1, 0, 52],
    bh_adj=False,
    bh_scale=False,
    nsteps=12,
)
start = weekly_SARIMA_nsteps_12[0].iloc[:, 2].first_valid_index()
end = len(weekly_SARIMA_nsteps_12[0])
weekly_SARIMA_nsteps_12_trimmed = (
    weekly_SARIMA_nsteps_12[0].iloc[start:end].drop(columns=series_name, axis=1)
)
monthly_forecasts = weekly_to_monthly_summary(weekly_SARIMA_nsteps_12_trimmed, distr)
monthly_forecasts_df = resample(preprocess(app_data, series_name), "M").merge(
    monthly_forecasts, how="inner"
)

pm1 = performance_metrics(monthly_forecasts_df, performance_lag=40, summarise=False)

_3_monthly_SARIMA = rolling_forecast_SARIMA(
    ts_data_monthly,
    window_end_date=monthly_measure_from,
    model_params=[0, 1, 0, 0, 1, 0, 12],
    bh_adj=False,
    bh_scale=False,
    nsteps=3,
)
pm2 = performance_metrics(_3_monthly_SARIMA[0], performance_lag=7, summarise=False)

_3_monthly_lagged_av_1 = rolling_forecast_lagged_average(
    time_series=ts_data_monthly,
    window_end_date=monthly_measure_from,
    lag=1,
    num_lags=1,
    nsteps=3,
)
pm3 = performance_metrics(_3_monthly_lagged_av_1[0], performance_lag=7, summarise=False)

_3_monthly_lagged_av_3 = rolling_forecast_lagged_average(
    time_series=ts_data_monthly,
    window_end_date=monthly_measure_from,
    lag=1,
    num_lags=3,
    nsteps=3,
)
pm4 = performance_metrics(_3_monthly_lagged_av_3[0], performance_lag=7, summarise=False)


_3_monthly_simple_growth = rolling_forecast_LY_perc_inc(
    time_series=ts_data_monthly, window_end_date=monthly_measure_from, nsteps=3
)
pm5 = performance_metrics(
    _3_monthly_simple_growth[0], performance_lag=7, summarise=False
)

_3_monthly_ensemble = rolling_forecast_ensemble(
    [_3_monthly_SARIMA, _3_monthly_lagged_av_1]
)
pm6 = performance_metrics(_3_monthly_ensemble[0], performance_lag=7, summarise=False)


# In[124]:


file_path = (
    os.path.split(os.getcwd())[0]
    + "/Prophet Scripts/"
    + series_name
    + "_3_monthly_prophet_forecasts"
)
infile = open(file_path, "rb")
pm_prophet_3 = pickle.load(infile)
infile.close()


# In[125]:


performance_merger([pm1, pm2, pm3, pm4, pm5, pm_prophet_3, pm6], "perc error")


# ### A 3 month out forecast

# In[126]:


# take the monthly SARIMA model out 3 steps. Note this may not be the best weelky model for every series!

_3_monthly_SARIMA = rolling_forecast_SARIMA(
    ts_data_monthly,
    window_end_date="2022-10-31",
    model_params=[0, 1, 0, 0, 1, 0, 12],
    bh_adj=False,
    bh_scale=False,
    nsteps=3,
)


# In[127]:


fig, ax = plt.subplots(figsize=(12, 5))

ax.set(title=series_name + " Forecasts by Month", xlabel="Date", ylabel=series_name)

ax = plt.plot(
    _3_monthly_SARIMA[0].iloc[:, 0].tail(20), _3_monthly_SARIMA[0].iloc[:, 1].tail(20)
)
ax = plt.plot(_3_monthly_SARIMA[1].iloc[:, 0], _3_monthly_SARIMA[1].iloc[:, 1])
plt.fill_between(
    _3_monthly_SARIMA[3].iloc[:, 0],
    _3_monthly_SARIMA[3].iloc[:, 1],
    _3_monthly_SARIMA[3].iloc[:, 2],
    color="r",
    alpha=0.1,
)

plt.show()


# ## Longterm Forecasts

# In[128]:


registered_users = read_file(user_data)


# In[137]:


registered_users.head(5) #registered users data must have a date format of dd/mm/yyyy. The headers must also be the same as below.


# In[129]:


# get the dampened uptake data
uptake_data = dampen(registered_users, n=10)


# In[130]:


uptake_data = uptake_data.drop(columns=["TotalRegisteredUsers"])


# In[131]:


# take the monthly SARIMA model out 25 steps. Note this may not be the best weelky model for every series!

_24_monthly_SARIMA = rolling_forecast_SARIMA(
    ts_data_monthly,
    window_end_date="2022-10-31",
    model_params=[0, 1, 0, 0, 1, 0, 12],
    bh_adj=False,
    bh_scale=False,
    nsteps=25,
)


# In[132]:


# dampen the forecasts out to account for the fact app uptake will slow over time.

trend_adj_forecast = trend_adjuster(
    _24_monthly_SARIMA[0], _24_monthly_SARIMA[1], uptake_data
)


# In[133]:


plt.figure(figsize=(12, 5))
fig, ax = plt.subplots(figsize=(12, 5))

ax.set(title=series_name + " Forecasts by Month", xlabel="Date", ylabel=series_name)

ax = plt.plot(
    _24_monthly_SARIMA[0].iloc[:, 0].tail(20), _24_monthly_SARIMA[0].iloc[:, 1].tail(20)
)
ax = plt.plot(_24_monthly_SARIMA[1].iloc[:, 0], _24_monthly_SARIMA[1].iloc[:, 1])
ax = plt.plot(trend_adj_forecast.iloc[:, 0], trend_adj_forecast.iloc[:, 1])

plt.fill_between(
    _24_monthly_SARIMA[3].iloc[:, 0],
    _24_monthly_SARIMA[3].iloc[:, 1],
    _24_monthly_SARIMA[3].iloc[:, 2],
    color="r",
    alpha=0.1,
)

plt.show()


# In[139]:


# write all the above code to a py file but not this particular cell of code.

