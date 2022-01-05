# Databricks notebook source
# MAGIC %md
# MAGIC ## Create data model

# COMMAND ----------

dbutils.widgets.text("eventsApartTimeLimit", "30")
eventsApartTimeLimit = int(dbutils.widgets.get("eventsApartTimeLimit"))
print(f"Events can be apart no more than {eventsApartTimeLimit} mins ({eventsApartTimeLimit/60} hrs) to fall into the same timeframe.")

dbutils.widgets.text("writeTables", "False")
writing_tables = dbutils.widgets.get("writeTables")
print(f"Writing tables: {writing_tables}")

dbutils.widgets.text("droppingLastSessionsUnregistered", "False")
dropping_last_sessions_of_unregistered = dbutils.widgets.get("droppingLastSessionsUnregistered")
print(f"Reading and writing of tables will be from tables where last sessions of unregistered fans is (True) or is not (False) dropped: {dropping_last_sessions_of_unregistered}")
# # if True -> drops last sessions of unregistered fans (the amount is given by parameter unregistered_drop_sessions_lastdays specified in 01_data_preprocessing notebook)

dbutils.widgets.text("unregisteredSampleFraction", "0.05")
unregistered_users_sample_fraction = float(dbutils.widgets.get("unregisteredSampleFraction"))
print(f"Sample of unregistered fans is chosen with fraction: {unregistered_users_sample_fraction}")

dbutils.widgets.text("outliersTooManyCookiesQuantile", "0.01")
outliers_too_many_cookies_quantile = float(dbutils.widgets.get("outliersTooManyCookiesQuantile"))
print(f"Dropping top {outliers_too_many_cookies_quantile}% of users with too many cookies.")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Variables

# COMMAND ----------

if dropping_last_sessions_of_unregistered:
  events_table_name = 'kv_likelytoregister_events_dropped'
  timeframes_table_name = 'kv_likelytoregister_timeframes_dropped'
else:
  events_table_name = 'kv_likelytoregister_events'
  timeframes_table_name = 'kv_likelytoregister_timeframes'


# COMMAND ----------

# MAGIC %md
# MAGIC #### Run ntbs

# COMMAND ----------

# MAGIC %run "./01_data_preprocessing"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Transformations

# COMMAND ----------

# add paired idm_id_paired to the records
# drop rows with not null idm_id and null idm_id_paired - this record belongs to the case when there is more than 1 user per 1 user_pseudo_id
# and in the case of 2-10 users per cookie - it is not the most active one, otherwise for >10 drop all
df_ga_paired = (df_ga_clean_filtered_sub
                .join(df_id_cookie_pairing, ['user_pseudo_id'], 'left') 
                .filter("""
                (GA_ID IS NULL) OR 
                (GA_ID IS NOT NULL AND idm_id_paired IS NOT NULL)
                """))

# COMMAND ----------

(df_ga_paired.filter('event_timestamp>"2021-11-13"').filter('idm_id_paired IS NULL').select('user_pseudo_id').distinct().count())

# COMMAND ----------

# drop fans with too many cookies (1% outliers)
df_fans_too_many_cookies_quantile = (df_ga_paired
                                    .filter('idm_id_paired IS NOT NULL')
                                    .select('idm_id_paired', 'user_pseudo_id')
                                    .distinct()
                                    .groupBy('idm_id_paired').count()
                                    .approxQuantile('count', [1-outliers_too_many_cookies_quantile], 0))[0]


df_fans_too_many_cookies_ids = (df_ga_paired
                               .filter('idm_id_paired IS NOT NULL')
                               .select('idm_id_paired', 'user_pseudo_id')
                               .distinct()
                               .groupBy('idm_id_paired').count()
                               .filter(f'count>{df_fans_too_many_cookies_quantile}'))

df_ga_paired = (df_ga_paired
                .join(df_fans_too_many_cookies_ids.select('idm_id_paired'), ['idm_id_paired'], 'left_anti')
               )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create sessions and timeframes

# COMMAND ----------

# calculate difference in minutes between events
w1 = Window.partitionBy('idm_id_paired', 'user_pseudo_id').orderBy('event_timestamp')

df_ga_lagged = (df_ga_paired
                .withColumn('previous_event_timestamp', f.lag('event_timestamp').over(w1).cast(T.TimestampType()))
                .withColumn('event_timestamp', f.col('event_timestamp').cast(T.TimestampType()))
                .withColumn('MinutesDifference', 
                            (f.unix_timestamp(f.col('event_timestamp')) - f.unix_timestamp(f.col('previous_event_timestamp')))/60))

# COMMAND ----------

# handle when a user watched a video

df_ga_video = (df_ga_lagged
               .withColumn('VideoMinutesLag', (f.lag(f.col('VideoSeconds')).over(w1)/60).cast(T.DoubleType())))

# COMMAND ----------

# subtract minutes when a fan watched a video from the time between events
# calculate session
df_ga_session = (df_ga_video
                 .withColumn('time_between_activities', f.greatest(f.lit(0),f.col('MinutesDifference')-f.col('VideoMinutesLag')))
                 .withColumn('isGreaterThanLimit', (f.col('time_between_activities')>eventsApartTimeLimit).cast(T.IntegerType()))
                 .withColumn('isGreaterThanLimit', f.when(f.col('isGreaterThanLimit').isNull(), 0).otherwise(f.col('isGreaterThanLimit')))
                 .withColumn('timeframe', f.sum(f.col('isGreaterThanLimit')).over(w1).cast(T.LongType()))
                 
                 .withColumn('isGreaterThan30', (f.col('time_between_activities')>30).cast(T.IntegerType()))
                 .withColumn('isGreaterThan30', f.when(f.col('isGreaterThan30').isNull(), 0).otherwise(f.col('isGreaterThan30')))
                 .withColumn('session', f.sum(f.col('isGreaterThan30')).over(w1).cast(T.LongType()))
                )



df_ga_session = (df_ga_session
                 .drop('previous_event_timestamp', 
                       'MinutesDifference', 
                       'VideoMinutesLag', 
                       'time_between_activities',
                       'isGreaterThanLimit',
                       'isGreaterThan30'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Drop sessions in the last week by unregistered users

# COMMAND ----------

df_sessions_to_drop = (df_ga_session
                       .filter('idm_id_paired IS NULL')
                       .groupBy('user_pseudo_id', 'session')
                       .agg(f.max('event_timestamp').alias('session_end'))
                       .filter(f.col('session_end')>f.lit(run_date - timedelta(unregistered_drop_sessions_lastdays)))
                       .drop('session_end'))

df_ga_session = (df_ga_session
                 .join(df_sessions_to_drop, ['user_pseudo_id', 'session'], 'left_anti'))

# COMMAND ----------

if writing_tables=='True':
  df_ga_session.cache().count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Random sample of never registered users

# COMMAND ----------

# Sample 5% of never registered users
df_never_registered_users = (df_ga_session
                             .filter('idm_id_paired IS NULL')
                             .select('user_pseudo_id')
                             .distinct())

df_never_registered_users_sample_ids = (df_never_registered_users
                                        .sample(fraction = unregistered_users_sample_fraction, withReplacement = False, seed = 42))

df_never_registered_users = (df_ga_session
                             .join(df_never_registered_users_sample_ids, ['user_pseudo_id'], 'inner'))

# COMMAND ----------

# Select registered users
df_registered_users = (df_ga_session
                       .filter('idm_id_paired IS NOT NULL'))

# COMMAND ----------

# Union registered and never registered users
df_subset = (df_never_registered_users
            .union(
            df_registered_users))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Drop events after registration

# COMMAND ----------

# Extract registration date
df_registration_date = (df_nonpersonal_clean
                        .selectExpr('Id AS idm_id_paired', 'RegistrationDate'))

# COMMAND ----------

# keep events before registration
df_before_registration = (df_subset
                          .join(df_registration_date, ['idm_id_paired'], 'left')
                          .filter('RegistrationDate IS NULL OR RegistrationDate>=event_timestamp')
                         )

# COMMAND ----------

# Impute registration date for fans whose RegistrationDate is null 
df_wo_missing_registration = (df_before_registration
                              .filter('RegistrationDate IS NULL AND idm_id_paired IS NOT NULL')
                              .select('idm_id_paired')
                              .distinct())

# COMMAND ----------

# impute registration date
df_registration_imputed = (df_wo_missing_registration
                           .join(
                             
                             GA_filter_event(spark = spark, df = df_ga, event = "register")
                             .filter('idm_id IS NOT NULL')
                             .groupBy('idm_id')
                             .agg(f.min('event_timestamp').alias('RegistrationDate'))
                             .withColumn('idm_id_paired', removeAllWhitespace(f.trim(f.lower(f.col('idm_id')))))
                             .withColumn('idm_id_paired', f.regexp_replace('idm_id_paired', '\\-', '')),
                             
                             ['idm_id_paired'], 
                             'left')
                           
                           .filter('RegistrationDate IS NOT NULL')
                           .selectExpr('idm_id_paired', 'RegistrationDate AS RegistrationDate_imputed')
                          )

# COMMAND ----------

# keep events before imputed registration 
df_with_registration = (df_before_registration
                       .join(df_registration_imputed, ['idm_id_paired'], 'left')
                       .filter('RegistrationDate_imputed IS NULL OR RegistrationDate_imputed>=event_timestamp')
                       )

# COMMAND ----------

# concat columns and drop fans with unknown registration date
df_with_registration = (df_with_registration
                       .withColumn('RegistrationDate_combined', 
                                   f.when(f.col('RegistrationDate').isNotNull(), f.col('RegistrationDate'))
                                  .otherwise(f.col('RegistrationDate_imputed')))
                       .drop('RegistrationDate', 'RegistrationDate_imputed'))

df_with_registration = (df_with_registration.filter("""
                          (RegistrationDate_combined IS NOT NULL AND idm_id_paired IS NOT NULL) OR 
                          (RegistrationDate_combined IS NULL AND idm_id_paired IS NULL)""")
                       )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create target

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Not yet registered fans

# COMMAND ----------

df_target_unregistered = (df_with_registration
                         .withColumn('Target', 
                                     f.when(f.col('idm_id_paired').isNull(), 0))
                         .filter('idm_id_paired IS NULL'))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Registered fans

# COMMAND ----------

df_last_event_before_registration = (
  df_with_registration
        .filter('idm_id_paired IS NOT NULL')
        .filter('RegistrationDate_combined>event_timestamp')
        .groupBy('idm_id_paired', 'user_pseudo_id')
        .agg(f.max('event_timestamp').alias('last_event_before_registration')) 
)

# COMMAND ----------

df_last_timeframe_before_registration = (
  df_with_registration
  .filter('idm_id_paired IS NOT NULL')
  .join(df_last_event_before_registration, ['idm_id_paired', 'user_pseudo_id'], 'inner')
  .withColumn('LastTimeframeNumber', 
              f.when(
                f.col('last_event_before_registration')==f.col('event_timestamp'), 
                f.col('timeframe'))
              .otherwise(None))
  .groupBy('idm_id_paired', 'user_pseudo_id')
  .agg(f.max('timeframe').alias('last_timeframe_before_registration'))
)

# COMMAND ----------

df_target_registered = (
  df_with_registration
  .join(df_last_timeframe_before_registration, ['idm_id_paired', 'user_pseudo_id'], 'inner')
  .withColumn('Target', 
              f.when(
                f.col('timeframe')==f.col('last_timeframe_before_registration'), 
                1)
              .otherwise(0))
  .drop('last_timeframe_before_registration')
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Union registered and unregistered users

# COMMAND ----------

df_model_events = (df_target_registered
                   .union(
                     df_target_unregistered
                   )
                  .drop('GA_ID'))

# df_model_events.cache().count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Write table and load it back

# COMMAND ----------

if writing_tables=='True':
  df_model_events.write.mode('overwrite').saveAsTable(events_table_name)

# COMMAND ----------

if writing_tables=='True':
  df_model_events = spark.table(events_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Relabeling of session number and timeframe number

# COMMAND ----------

df_model_events = (df_model_events
                   .withColumn('idm_id_paired', 
                               f.when(
                                 f.col('idm_id_paired').isNull(), 'unknown')
                               .otherwise(f.col('idm_id_paired'))))

# COMMAND ----------

# relabel sessions with max 0
df_max_session = (
  df_model_events
  .groupBy('idm_id_paired', 'user_pseudo_id')
  .agg(f.max('session').alias('max_session'))
)

df_model_events_session_relabeled = (
  df_model_events
  .join(df_max_session, ['idm_id_paired', 'user_pseudo_id'], 'inner')
  .withColumn('session_relabeled', (f.col('session') - f.col('max_session')))
  .drop('session', 'max_session')
  .withColumnRenamed('session_relabeled', 'session')
)


# COMMAND ----------

# relabel timeframes with max 0
df_max_timeframe = (
  df_model_events
  .groupBy('idm_id_paired', 'user_pseudo_id')
  .agg(f.max('timeframe').alias('max_timeframe'))
)

df_model_events_timeframe_relabeled = (
  df_model_events_session_relabeled
  .join(df_max_timeframe, ['idm_id_paired', 'user_pseudo_id'], 'inner')
  .withColumn('timeframe_relabeled', (f.col('timeframe') - f.col('max_timeframe')))
  .drop('timeframe', 'max_timeframe')
  .withColumnRenamed('timeframe_relabeled', 'timeframe')
)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Write tables

# COMMAND ----------

if writing_tables=='True':
  df_model_events_timeframe_relabeled.write.mode('overwrite').saveAsTable(events_table_name)

# COMMAND ----------

df_model_events = spark.table(events_table_name)

# COMMAND ----------

df_model_timeframes = (df_model_events
                       .select('idm_id_paired', 'user_pseudo_id', 'timeframe', 'Target')
                       .distinct())

# COMMAND ----------

if writing_tables=='True':
  df_model_timeframes.write.mode('overwrite').saveAsTable(timeframes_table_name)

# COMMAND ----------

df_model_timeframes = spark.table(timeframes_table_name)

# COMMAND ----------

# sessions
print('Number of rows in the final data model: ', 
      df_model_events.select('idm_id_paired', 'user_pseudo_id', 'session', 'Target').distinct().count())
print('Number of targets in the final data model: ', 
      df_model_events.select('idm_id_paired', 'user_pseudo_id', 'session', 'Target').distinct().filter('Target=1').count())


# COMMAND ----------

# timeframes
print('Number of rows in the final data model: ', 
      df_model_events.select('idm_id_paired', 'user_pseudo_id', 'timeframe', 'Target').distinct().count())
print('Number of targets in the final data model: ', 
      df_model_events.select('idm_id_paired', 'user_pseudo_id', 'timeframe', 'Target').distinct().filter('Target=1').count())


# COMMAND ----------

df_model_events.count()

# COMMAND ----------

# timeframe length in days
display(df_model_events
        .groupBy('idm_id_paired', 'user_pseudo_id', 'timeframe')
        .agg(f.min('event_timestamp'), f.max('event_timestamp'))
        .withColumn('timeframe_length', 
                   (f.col('max(event_timestamp)').cast('double')-f.col('min(event_timestamp)').cast('double'))/(60*60*24))
        .summary()
       )

# COMMAND ----------

# B: Time between consecutive sessions
# On average, sessions are apart 78 hrs = 3 days, with minimum of 31 mins and maximum of 2288 hours = 95 days (three months)
# 1st quantile is 3 hrs --> suggested timeframe parameter = 3 hrs between events
# Note: dropping records with very first session
(df_model_events
       .groupBy('idm_id_paired', 'user_pseudo_id', 'session')
       .agg(f.min('event_timestamp').alias('session_start'),
            f.max('event_timestamp').alias('session_end'))
        .withColumn('previous_session_end', f.lag(f.col('session_end')).over(Window().partitionBy('idm_id_paired', 'user_pseudo_id').orderBy('session_start')))
       .withColumn('time_between_sessions', 
                   (f.col('session_start').cast('double') - f.col('previous_session_end').cast('double'))/(60*60))
        .filter('time_between_sessions IS NOT NULL')
       .approxQuantile('time_between_sessions', [0.25, 0.3, 0.35, 0.4, 0.5], 0))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Additional analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ##### CityPlus x video interaction

# COMMAND ----------

# df_never_registered_users = (df_ga_session
#         .filter('idm_id_paired IS NULL')
#         .select('user_pseudo_id')
#         .distinct()
#        )
# 
# df_never_registered_users.cache().count()

# COMMAND ----------

# df_cityplus_users = (df_ga.join(df_never_registered_users, ['user_pseudo_id'], 'inner')
#                      .withColumn('cityplus', f.when(f.col('event_params_page_location').rlike("/city-plus"), 1).otherwise(0))
#                      .filter('cityplus = 1')
#                      .select('user_pseudo_id')
#                      .distinct())
# 
# df_cityplus_video = (df_ga.join(df_never_registered_users, ['user_pseudo_id'], 'inner')
#                      .join(df_cityplus_users, ['user_pseudo_id'], 'inner')
#                      .select('user_pseudo_id', 'event_timestamp', 'event_params_page_location')
#                      .distinct()
#                      .withColumn('cityplus', f.when(f.col('event_params_page_location').rlike("/city-plus"), 1).otherwise(0))
#                      .withColumn('video', f.when(f.col('event_params_page_location').rlike('citytv'), 1).otherwise(0))
#                      .withColumn('video', f.when(f.col('cityplus')==1, 0).otherwise(f.col('video')))
#                      .filter('cityplus = 1 OR video = 1')
#                     )
# 

# COMMAND ----------

# # cityplus and then video
# df_cityplus_video_ts = (df_cityplus_video
#         .select('user_pseudo_id', 'event_timestamp','cityplus', 'video')
#         .dropDuplicates()
#         .withColumn('next', f.lead('video').over(Window().partitionBy('user_pseudo_id').orderBy('event_timestamp')))
#         .filter('cityplus = 1 AND (next IS NULL OR next = 1)')
#         .withColumn('last_cityplus_before_video_or_overall', f.col('event_timestamp'))
#                              )
# 
# df_cityplus_and_then_video = (df_cityplus_video_ts
#                               .join(df_cityplus_video_ts
#                                    .groupBy('user_pseudo_id').count().filter('count=1').drop('count'), 
#                                     ['user_pseudo_id'],
#                                    'inner'))

# COMMAND ----------

# display(df_cityplus_and_then_video
#         .selectExpr('count(*)')
#        )

# COMMAND ----------

# # video -> cityplus
# df_video_cityplus_ts = (df_cityplus_video
#                         .select('user_pseudo_id', 'event_timestamp','cityplus', 'video')
#                         .dropDuplicates()
#                         .withColumn('next', f.lead('cityplus').over(Window().partitionBy('user_pseudo_id').orderBy('event_timestamp')))
#                         .filter('video = 1 AND (next IS NULL OR next = 1)')
#                         .withColumn('last_video_before_cityplus_or_overall', f.col('event_timestamp'))
#                              )
# 
# df_video_and_then_cityplus = (df_video_cityplus_ts
#                               .join(df_video_cityplus_ts
#                                    .groupBy('user_pseudo_id').count().filter('count=1').drop('count'), 
#                                     ['user_pseudo_id'],
#                                    'inner'))

# COMMAND ----------

# display(df_video_and_then_cityplus
#        .selectExpr('count(*)')
#       )

# COMMAND ----------

# display(df_video_cityplus_ts
#                                    .groupBy('user_pseudo_id').count().filter('count>1').drop('count').selectExpr('count(*)'))

# COMMAND ----------

# display(df_cityplus_video_ts
#                                    .groupBy('user_pseudo_id').count().filter('count>1').drop('count').selectExpr('count(*)'))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Registration date

# COMMAND ----------

# pd_dayofweek = (spark.table('kv_likelytoregister_events_dropped')
#         .filter('idm_id_paired != "unknown"')
#        .filter('Target = 1')
#        .groupBy('idm_id_paired', 'user_pseudo_id')
#        .agg(f.max('event_timestamp').alias('registration_ts'))
#        .withColumn('day_of_week', f.dayofweek('registration_ts'))
#        .groupBy('day_of_week').count()
#                .orderBy('day_of_week')).toPandas()
# 
# pd_dayofweek['days'] = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

# COMMAND ----------

# import plotly.express as px
# fig = px.bar(x = pd_dayofweek['days'], y = pd_dayofweek['count'],
#              labels = {'x':'Days', 'y': ''},
#              title = 'Registration count by day of week')
# fig.update_layout(template = 'plotly_white')
