# Databricks notebook source
# MAGIC %md
# MAGIC ## Factor analysis & Weekly volume

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run ntb

# COMMAND ----------

# MAGIC %run "./01_data_preprocessing"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup

# COMMAND ----------

!pip install factor_analyzer  

# COMMAND ----------

from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# COMMAND ----------

pd_model = spark.table('kv_df_features').toPandas()

# COMMAND ----------

categorical_cols = (spark
                    .table('kv_likely_to_register_cols')
                    .select('categorical_cols')
                    .collect()[0][0]
                   )

pd_model_transformed = pd.get_dummies(pd_model, columns=categorical_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weekly volumes

# COMMAND ----------

# # Show weekly volume 
# df_ga_weekly_volume = (df_ga
#         .withColumn('date', f.to_date('event_timestamp'))
#         .dropDuplicates(['idm_id', 'user_pseudo_id', 'date'])
#         .withColumn('week', f.weekofyear('event_timestamp'))
#                       .filter(f'event_timestamp>="{start_date}"'))
# 
# df_ga_weekly_volume.cache().count()

# COMMAND ----------

# # since start_date
# display(df_ga_weekly_volume
#         .filter('week>27 AND week<45')
#         .groupBy('week').count())

# COMMAND ----------

# display(df_ga_weekly_volume)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Factor Analysis

# COMMAND ----------

pd_model_transformed['desktop_retail'] = pd_model_transformed['device_category_desktop']*pd_model_transformed['VisitedRetailFlag']
pd_model_transformed['fixtures_uk'] = pd_model_transformed['VisitedFixturesFlag']*pd_model_transformed['geo_country_grouped_United Kingdom']

# COMMAND ----------

columns_rules = ['number_of_events',
                 'change_from_running_avg',
                 'previous_website_medium_cleaned_grouped_social',
                 'VisitedCityPlusFlag',
                 'VisitedMembershipFlag',
                 'previous_website_name_cleaned_grouped_google',
                 'previous_website_name_cleaned_grouped_direct',
                 'isFixture',
                 'previous_website_medium_cleaned_grouped_cpc',
                 'device_category_mobile',
                 'desktop_retail',
                 #'previous_website_name_cleaned_grouped_premier_league',
                 #'fixtures_uk'
                ]

pd_model_transformed_subset = pd_model_transformed[columns_rules]

# COMMAND ----------

x_all = pd_model_transformed_subset
fa = FactorAnalyzer()
fa.fit(x_all)
#Get Eigen values and plot them
ev, v = fa.get_eigenvalues()
ev
plt.plot(range(1,x_all.shape[1]+1),ev)

# COMMAND ----------

# varimax rotation, which maximizes the sum of the variance of squared loadings while ensuring that the factors created are not correlated (orthogonality)

fa_varimax_minres = FactorAnalyzer(2, 
                                   rotation='varimax',
                                  )
fa_varimax_minres.fit(x_all)

fa_varimax_ml = FactorAnalyzer(2, 
                               rotation='promax'
                              )
fa_varimax_ml.fit(x_all)

loads_all = fa_varimax_ml.loadings_

# COMMAND ----------

pd_factors = pd.DataFrame(zip(columns_rules, 
                              *np.round(loads_all,3).T), 
                          columns = ['feature', 'factor1', 'factor2'],
                         ).set_index('feature')

# COMMAND ----------

components = []
for i in pd_factors.index:
  components.append(pd_factors.loc[i,:].idxmax())

# COMMAND ----------

pd_factors['final_component'] = components

pd_factors
