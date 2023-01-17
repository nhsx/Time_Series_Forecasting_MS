#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os, sys, importlib


# In[ ]:





# In[7]:


sys.path.append("../Functions")


# In[8]:


sys.path.append("../Models")


# In[9]:


from preprocessing_functions import *
from performance_functions import *
from aggregate_functions import *


# In[10]:


from facebook_prophet import rolling_prophet


# In[11]:


import pickle
import matplotlib.pyplot as plt


# In[12]:


series_name = "dummy_series"
data = "dummy_data.csv"


# In[13]:


app_data = read_file(data)


# In[14]:


ts_data = preprocess(app_data, series_name)
ts_data_weekly = resample(ts_data, "W")
ts_data_weekly = ts_data_weekly.iloc[:-1]
ts_data_weekly.tail(2)


# In[15]:


weekly_measure_from = "2022-03-27"  #'2022-05-29'
monthly_measure_from = "2022-03-31"


# #  Weekly Model

# In[16]:


weekly_prophet_nsteps_1 = rolling_prophet(
    ts_data_weekly, window_end_date=weekly_measure_from, nsteps=1, window_length=None
)


# In[17]:


prophet_weekly_performance = performance_metrics(
    weekly_prophet_nsteps_1[0], performance_lag=12
)
prophet_weekly_performance


# In[18]:


file_name = series_name + "_weekly_prophet_nsteps_1"
outfile = open(file_name, "wb")
pickle.dump(weekly_prophet_nsteps_1, outfile)
outfile.close()


# In[19]:


# Have a look how these models would have performed historically

plt.figure(figsize=(12, 5))

ax = weekly_prophet_nsteps_1[0][series_name].plot(color="blue")
ax = weekly_prophet_nsteps_1[0]["prophet_nsteps_1"].plot(color="red")

plt.show()


# # 4 week out performance gain with each new week

# In[20]:


#%%capture
weekly_prophet_nsteps_1 = rolling_prophet(
    ts_data_weekly, window_end_date=weekly_measure_from, nsteps=1, window_length=None
)
weekly_prophet_nsteps_2 = rolling_prophet(
    ts_data_weekly, window_end_date=weekly_measure_from, nsteps=2, window_length=None
)
weekly_prophet_nsteps_3 = rolling_prophet(
    ts_data_weekly, window_end_date=weekly_measure_from, nsteps=3, window_length=None
)
weekly_prophet_nsteps_4 = rolling_prophet(
    ts_data_weekly, window_end_date=weekly_measure_from, nsteps=4, window_length=None
)


# In[21]:


#%%capture
weekly_prophet_out_4_nsteps_i = []

weekly_prophet_out_4_nsteps_1 = n_weeks_forecast(
    weekly_prophet_nsteps_1[0], nsteps=1, n_week_sum=4
)
weekly_prophet_out_4_nsteps_i.append(weekly_prophet_out_4_nsteps_1)

weekly_prophet_out_4_nsteps_2 = n_weeks_forecast(
    weekly_prophet_nsteps_2[0], nsteps=2, n_week_sum=4
)
weekly_prophet_out_4_nsteps_i.append(weekly_prophet_out_4_nsteps_2)

weekly_prophet_out_4_nsteps_3 = n_weeks_forecast(
    weekly_prophet_nsteps_3[0], nsteps=3, n_week_sum=4
)
weekly_prophet_out_4_nsteps_i.append(weekly_prophet_out_4_nsteps_3)

weekly_prophet_out_4_nsteps_4 = n_weeks_forecast(
    weekly_prophet_nsteps_4[0], nsteps=4, n_week_sum=4
)
weekly_prophet_out_4_nsteps_i.append(weekly_prophet_out_4_nsteps_4)


# In[22]:


file_name = series_name + "_weekly_prophet_out_4_nsteps_i"
outfile = open(file_name, "wb")
pickle.dump(weekly_prophet_out_4_nsteps_i, outfile)
outfile.close()


# In[23]:


performance_dfs = []

performance_df_out_4_nsteps_1 = performance_metrics(
    weekly_prophet_out_4_nsteps_1, performance_lag=12
)
performance_dfs.append(performance_df_out_4_nsteps_1)

performance_df_out_4_nsteps_2 = performance_metrics(
    weekly_prophet_out_4_nsteps_2, performance_lag=12
)
performance_dfs.append(performance_df_out_4_nsteps_2)

performance_df_out_4_nsteps_3 = performance_metrics(
    weekly_prophet_out_4_nsteps_3, performance_lag=12
)
performance_dfs.append(performance_df_out_4_nsteps_3)

performance_df_out_4_nsteps_4 = performance_metrics(
    weekly_prophet_out_4_nsteps_4, performance_lag=12
)
performance_dfs.append(performance_df_out_4_nsteps_4)


# In[24]:


prophet_performance_df = pd.DataFrame()
for df in performance_dfs:
    prophet_performance_df = pd.concat([prophet_performance_df, df], axis=0)
prophet_performance_df = prophet_performance_df.reset_index().drop("index", axis=1)
prophet_performance_df


# # What's the 1 month out forecast error?

# In[25]:


weekly_prophet_nsteps_4 = rolling_prophet(
    ts_data_weekly, window_end_date=weekly_measure_from, nsteps=4, window_length=None
)


# In[58]:


distr = weekly_distribution(app_data, series_name)


# In[59]:


start = weekly_prophet_nsteps_4[0].iloc[:, 2].first_valid_index()
end = len(weekly_prophet_nsteps_4[0])
weekly_prophet_nsteps_4_trimmed = (
    weekly_prophet_nsteps_4[0].iloc[start:end].drop(columns=series_name, axis=1)
)
monthly_forecasts = weekly_to_monthly_summary(weekly_prophet_nsteps_4_trimmed, distr)


# In[60]:


monthly_prophet_forecasts_df = resample(preprocess(app_data, series_name), "M").merge(
    monthly_forecasts, how="inner"
)
pm_prophet = performance_metrics(
    monthly_prophet_forecasts_df, performance_lag=40, summarise=False
)
pm_prophet


# In[61]:


file_name = series_name + "_monthly_prophet_forecasts"
outfile = open(file_name, "wb")
pickle.dump(pm_prophet, outfile)
outfile.close()


# # What's the 3 month out forecast error?

# In[62]:


weekly_prophet_nsteps_12 = rolling_prophet(
    ts_data_weekly, window_end_date=weekly_measure_from, nsteps=12, window_length=None
)


# In[63]:


distr = weekly_distribution(app_data, series_name)


# In[64]:


start = weekly_prophet_nsteps_12[0].iloc[:, 2].first_valid_index()
end = len(weekly_prophet_nsteps_12[0])
weekly_prophet_nsteps_12_trimmed = (
    weekly_prophet_nsteps_12[0].iloc[start:end].drop(columns=series_name, axis=1)
)
tri_monthly_forecasts = weekly_to_monthly_summary(
    weekly_prophet_nsteps_12_trimmed, distr
)


# In[65]:


monthly_prophet_forecasts_df = resample(preprocess(app_data, series_name), "M").merge(
    tri_monthly_forecasts, how="inner"
)
pm_prophet_3 = performance_metrics(
    monthly_prophet_forecasts_df, performance_lag=40, summarise=False
)
pm_prophet_3


# In[66]:


file_name = series_name + "_3_monthly_prophet_forecasts"
outfile = open(file_name, "wb")
pickle.dump(pm_prophet_3, outfile)
outfile.close()


# In[68]:


