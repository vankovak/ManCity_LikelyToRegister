# Databricks notebook source
# MAGIC %md
# MAGIC ## Data pre-processing

# COMMAND ----------

dbutils.widgets.text("StartDate", "2021-07-06")
start_date = dbutils.widgets.get("StartDate")
print(f"Data since date: {start_date}")

dbutils.widgets.text("DropSessionsUnregistered", "7")
unregistered_drop_sessions_lastdays = int(dbutils.widgets.get("DropSessionsUnregistered"))
print(f"Drop sessions of unregistered users in the last {unregistered_drop_sessions_lastdays} days.")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Import libraries

# COMMAND ----------

import os

import pyspark.sql.functions as f
import pyspark.sql.types as T
from pyspark.sql import Window

from pyspark.mllib.stat import Statistics

from scipy.spatial.distance import jaccard

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from adap_spark.utils import removeAllWhitespace, GA_filter_event

# COMMAND ----------

# MAGIC %md
# MAGIC #### Import tables

# COMMAND ----------

## Define Hive tables
tables = ['prod_solutions_sdm.out_google_analytics_actions', 
          'prod_solutions_sdm.out_nonpersonal',
          'prod_solutions_fsfan.out_fan360_profile_current',
          'prod_dataregular_fixtures.cleansed_fixtures'
         ]

## Define DataFrame names
names = ['df_ga',
         'df_nonpersonal',
         'df_profile',
         'df_fixtures'
        ]

## Loop through tables and create DataFrames
for tab, name in zip(tables, names):
  spark.catalog.refreshTable(tab)
  exec(f"{name} = spark.table('{tab}')")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create variables

# COMMAND ----------

run_date = datetime.now().date()

# COMMAND ----------

# udf for max
udf_max = udf(lambda c: max(c))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Tables setup

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Profile table setup

# COMMAND ----------

df_profile_clean = (df_profile
                    .withColumn('id_clean', removeAllWhitespace(f.trim(f.lower(f.col('Profile_ID_UniqueID')))))
                    .withColumn('id_clean', f.regexp_replace('id_clean', '\\-', ''))
                    .withColumnRenamed('id_clean', 'Id')
                    .select('Id', 'Profile_Acquisition_Source', 'Profile_Acquisition_Subsource')
                    )

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Nonpersonal table setup

# COMMAND ----------

df_nonpersonal_clean = (df_nonpersonal
                        .withColumn('id_clean', removeAllWhitespace(f.trim(f.lower(f.col('Id')))))
                        .withColumn('id_clean', f.regexp_replace('id_clean', '\\-', ''))
                        .drop('Id')
                        .withColumnRenamed('id_clean', 'Id')
                        .select('Id', 'GdprConsentDate', 'GdprConsentSource', 'ProfileCreated')
                        .withColumn('RegistrationDate', 
                                    f.when(
                                      f.lower(f.trim(f.col('GdprConsentSource')))=="idm", 
                                      f.col('GdprConsentDate'))
                                    .otherwise(
                                      f.col('ProfileCreated')))
                        )

# COMMAND ----------

# MAGIC %md
# MAGIC ##### GA tables setup

# COMMAND ----------

# clean IDs 
df_ga_clean = (df_ga
               .withColumn('GA_ID', removeAllWhitespace(f.trim(f.lower(f.col('IDM_Id')))))
               .withColumn('GA_ID', f.regexp_replace('GA_ID', '\\-', ''))
               )

# COMMAND ----------

# dataframe with idm_id - user_pseudo_id pairing, 
# where user_pseudo_id is linked to more idm_ids then
#    -> when it is linked to <=10 idm_ids -> user_pseudo_id will be linked to the most active idm_id 
#    -> when it is linked to > 10 idm_ids -> records are dropped

df_id_cookie_pairing_keep = (df_ga_clean
                             .selectExpr('GA_ID AS idm_id_paired', 'user_pseudo_id')
                             .filter('idm_id_paired IS NOT NULL')
                             .dropDuplicates()
                             .groupBy('user_pseudo_id').count()
                             .filter('count<=10')
                             .drop('count')
                             )

# COMMAND ----------

# get most active fan count
df_most_active_fan_count = (df_ga_clean
                           .join(df_id_cookie_pairing_keep, ['user_pseudo_id'], 'inner')
                           .filter('GA_ID IS NOT NULL')
                           .groupBy('user_pseudo_id', 'GA_ID')
                           .count()
                           .groupBy('user_pseudo_id')
                           .agg(f.max('count').alias('max_count'))
                           )

# select fans with the highest count and randomly when there is a tie
df_most_active_fan_pairing = (df_ga_clean
                             .filter('GA_ID IS NOT NULL')
                             .groupBy('user_pseudo_id', 'GA_ID')
                             .agg(f.count('GA_ID').alias('max_count'))
                             .join(df_most_active_fan_count, ['user_pseudo_id', 'max_count'], 'inner')
                             .dropDuplicates(['user_pseudo_id'])
                             .drop('max_count')
                             )

# COMMAND ----------

# cookie - id pairing, with duplicate user_pseudo_ids
df_pre_id_cookie_pairing = (df_ga_clean
                           .selectExpr('GA_ID AS idm_id_paired', 'user_pseudo_id')
                           .filter('idm_id_paired IS NOT NULL')
                           .dropDuplicates()
                           .join(df_id_cookie_pairing_keep, ['user_pseudo_id'], 'inner')
                           )

# keep only most active fans
df_id_cookie_pairing = (df_pre_id_cookie_pairing
                        .join(df_most_active_fan_pairing, ['user_pseudo_id'], 'inner')
                        .filter('GA_ID = idm_id_paired')
                        .drop('GA_ID'))

# COMMAND ----------

# filter history
df_ga_clean_filtered = (df_ga_clean
                        .filter(f.col('event_timestamp') >= start_date)
                       )

# COMMAND ----------

# get video seconds
df_ga_clean_filtered_video = (df_ga_clean_filtered
                              .withColumn('VideoSecondsString', 
                                          f.when(f.col('event_params_video_seconds_viewed').getItem('video') == 'Null', None)
                                          .otherwise(f.col('event_params_video_seconds_viewed').getItem('video')))
                              .withColumn('VideoSecondsArray', 
                                          f.array_remove(f.split(f.col('VideoSecondsString'), '\\|'), 'Null'))
                              .withColumn('VideoSecondsArrayNull',
                                          f.when(f.col('VideoSecondsArray').isNull(), f.array(f.lit(0)))
                                          .otherwise(f.col('VideoSecondsArray')))
                              .withColumn('VideoSeconds', udf_max(f.col('VideoSecondsArrayNull').cast("array<int>")))
                              .drop('VideoSecondsString', 'VideoSecondsArray', 'VideoSecondsArrayNull')
                              
                             )

# COMMAND ----------

# select columns
df_ga_clean_filtered_sub = (df_ga_clean_filtered_video
                           .select('GA_ID', 'user_pseudo_id',
                                   'event_timestamp', 
                                   'event_params_page_location',  
                                   'events_names',   
                                   'VideoSeconds', 
                                   'traffic_source_source', # previous webpage
                                   'traffic_source_medium', # previous medium (cpc, shop, affiliates,...messy)
                                   'geo_country', 'geo_city',
                                   'device_category', 'device_operating_system', 'device_mobile_brand_name'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Additional data analyses

# COMMAND ----------

# display(df_ga_clean.selectExpr('GA_ID AS idm_id_paired', 'user_pseudo_id')
#         .dropDuplicates()
#         .filter('idm_id_paired IS NOT NULL')
#        .groupBy('user_pseudo_id').count()
#        )

# COMMAND ----------

# import plotly.express as px
# import pandas as pd
# pd_cookie_fan_counts = pd.DataFrame([['1 cookie : >10 fans', 50718],
#                                      ['1 cookie : 2-10 fans', 493847],
#                                      ['1 cookie : 1 fan', 1199445],
#                                      ['1 cookie : 1 not a fan', 8053674],
#                                      ], columns = ['relation', 'count'])
# fig = px.bar(
#   y = pd_cookie_fan_counts['relation'], 
#   x = pd_cookie_fan_counts['count'], 
#   template = 'plotly_white',
#   labels = {'x': 'Count of cookies',
#             'y': 'Category'},
#   text = pd_cookie_fan_counts['count']
# )
# fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
# fig.show()

# COMMAND ----------

# display(df_ga_clean.selectExpr('GA_ID AS idm_id_paired', 'user_pseudo_id')
#         .dropDuplicates()
#         .filter('idm_id_paired IS NOT NULL')
#         .groupBy('idm_id_paired').count())
