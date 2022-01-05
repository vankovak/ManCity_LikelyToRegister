# Databricks notebook source
# MAGIC %md
# MAGIC ## Evaluate the effectiveness of the change

# COMMAND ----------

import os

import pyspark.sql.functions as f
import pyspark.sql.types as T

# COMMAND ----------

df_ga_actions = spark.table('prod_solutions_sdm.out_google_analytics_actions')
df_nonpersonal = spark.table('prod_solutions_sdm.out_nonpersonal')

# COMMAND ----------

# DBTITLE 1,Number of unique cookies that interacted with the campaign - control group
(df_ga_actions
        .filter('event_timestamp>="2021-11-19"')
        .filter('event_params_campaign LIKE "%competition_fave_player_shirt_2111_A_control%"')
        .select('user_pseudo_id').distinct()
        .count()
       )

# COMMAND ----------

# DBTITLE 1,Number of unique cookies that interacted with the campaign - GA audience
(df_ga_actions
        .filter('event_timestamp>="2021-11-19"')
        .filter('event_params_campaign LIKE "%competition_fave_player_shirt_2111_B_prop%"')
        .select('user_pseudo_id').distinct()
        .count()
       )

# COMMAND ----------

# DBTITLE 1,Number of registered users - GA audience
# The number of idm_ids paired to the user_pseudo_id that saw the campaign
# 1561 suggests that already registered users have seen the campaign
(df_ga_actions
 .join(
   df_ga_actions
   .filter('event_timestamp>="2021-11-19"')
   .filter('event_params_campaign LIKE "%competition_fave_player_shirt_2111_B_prop%"')
   .select('user_pseudo_id')
   .distinct(),
   ['user_pseudo_id'], 
   'inner'
 )
 .filter('event_timestamp>="2021-11-19"')
 .filter('idm_id IS NOT NULL')
 .select('idm_id')
 .distinct()
 .count())

# COMMAND ----------

# The number of user_pseudo_ids from GA audience that had GA event register
display(df_ga_actions
        .join(
          df_ga_actions
          .filter('event_timestamp>="2021-11-19"')
          .filter('event_params_campaign LIKE "%competition_fave_player_shirt_2111_B_prop%"')
          .select('user_pseudo_id')
          .distinct(),
          ['user_pseudo_id'], 
          'inner')
        .filter('ARRAY_CONTAINS(events_names, "register")')
        .dropDuplicates(['user_pseudo_id'])
       )

# COMMAND ----------

# DBTITLE 1,Number of registered users - control group
(df_ga_actions.join(
df_ga_actions
        .filter('event_timestamp>="2021-11-19"')
        .filter('event_params_campaign LIKE "%competition_fave_player_shirt_2111_A_control%"')
        .select('user_pseudo_id').distinct(),
['user_pseudo_id'], 'inner')
       .filter('idm_id IS NOT NULL')
       .select('idm_id').distinct()
       .count())

# COMMAND ----------

# The number of user_pseudo_ids from control group that had GA event register
display(df_ga_actions.join(
df_ga_actions
        .filter('event_timestamp>="2021-11-19"')
        .filter('event_params_campaign LIKE "%competition_fave_player_shirt_2111_A_control%"')
        .select('user_pseudo_id').distinct(),
['user_pseudo_id'], 'inner')
       .filter('ARRAY_CONTAINS(events_names, "register")')
       .dropDuplicates(['user_pseudo_id'])
       )

# COMMAND ----------


