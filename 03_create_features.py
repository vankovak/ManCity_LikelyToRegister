# Databricks notebook source
# MAGIC %md
# MAGIC ## Create features

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run ntbs

# COMMAND ----------

# MAGIC %run "./02_create_data_model"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set parameters

# COMMAND ----------

unique_key_session = ['idm_id_paired', 'user_pseudo_id', 'session']
unique_key_timeframe = ['idm_id_paired', 'user_pseudo_id', 'timeframe']

# visited content flag feature
content_groups_dict = {'News': '/news',
                       'Retail': 'shop.mancity',
                       'Video': 'citytv',
                       'CityPlus': '/city-plus', 
                       'Ticketing': 'tickets.mancity|/ticketing-and-hospitality|/tickets|hospitality.mancity',
                       'Membership': 'membership',
                       'Fixtures': '/fixtures|/results|/tables'}

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define functions

# COMMAND ----------

def return_top_n_groups(df: f.DataFrame, col_name: str, n: int):
  df_out = (df
           .groupBy(col_name)
           .count()
           .orderBy(f.desc('count'))
           .limit(n)
           .drop('count')
           .withColumn(col_name+'_grouped', f.col(col_name)))
  
  return df_out

# COMMAND ----------

def return_top_group_per_unique_key(df: f.DataFrame, unique_key, col_name: str):
  
  df_max = (df
           .groupBy(*unique_key, col_name)
           .count()
           .groupBy(*unique_key)
           .agg(f.max('count').alias('count'))
           )
  
  df_out = (df
           .groupBy(*unique_key, col_name)
           .count()
           .join(df_max, [*unique_key, 'count'], 'inner')
           .dropDuplicates(unique_key)
           .drop('count')
           )
  
  return df_out

# COMMAND ----------

def join_groups(df_join_left: f.DataFrame, 
                df_join_right: f.DataFrame, 
                join_key: str, 
                group_other_name: str):
  
  col_name = join_key + '_grouped'
  
  df_out = (df_join_left
            .join(df_join_right, [join_key], 'left')
            .withColumn(col_name, 
                        f.when(f.col(col_name).isNull(), group_other_name)
                        .otherwise(f.col(col_name)))
           .drop(col_name[:-8]))
  
  return df_out

# COMMAND ----------

# MAGIC %md
# MAGIC #### Number of actions

# COMMAND ----------

df_number_of_events = (df_model_events
                       .groupBy(*unique_key_timeframe)
                       .agg(f.count('*').alias('number_of_events'))
                      )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Number of sessions

# COMMAND ----------

df_number_of_sessions = (df_model_events
                         .select(*unique_key_timeframe, 'session')
                         .distinct()
                         .groupBy(unique_key_timeframe)
                         .agg(f.count('session').alias('number_of_sessions'))
                        )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Number of articles/news

# COMMAND ----------

df_number_of_news = (df_model_events
                     .filter('ARRAY_CONTAINS(events_names, "page_view")')
                     .filter('event_params_page_location LIKE "%mancity.com/news/%"')
                     .dropDuplicates([*unique_key_timeframe, 'event_params_page_location'])
                     .groupBy(*unique_key_timeframe)
                     .agg(f.count('*').alias('number_of_news')))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Number of videos

# COMMAND ----------

df_number_of_videos = (df_model_events
                       .filter('ARRAY_CONTAINS(events_names, "video")')
                       .dropDuplicates([*unique_key_timeframe, 'event_params_page_location'])
                       .groupBy(*unique_key_timeframe)
                       .agg(f.count('*').alias('number_of_videos')))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Device and location characteristics

# COMMAND ----------

# Most common
df_device_location = (
  return_top_group_per_unique_key(df = df_model_events, unique_key = unique_key_timeframe, col_name = 'device_category')
  .join(
    return_top_group_per_unique_key(df = df_model_events, unique_key = unique_key_timeframe, col_name = 'device_operating_system'),
    unique_key_timeframe,
    'inner'
  )
  .join(
    return_top_group_per_unique_key(df = df_model_events, unique_key = unique_key_timeframe, col_name = 'device_mobile_brand_name'),
    unique_key_timeframe,
    'inner'
  )
  .join(
    return_top_group_per_unique_key(df = df_model_events, unique_key = unique_key_timeframe, col_name = 'geo_country'),
    unique_key_timeframe,
    'inner'
  )
  .join(
    return_top_group_per_unique_key(df = df_model_events, unique_key = unique_key_timeframe, col_name = 'geo_city'),
    unique_key_timeframe,
    'inner'
  )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create grouped device OS

# COMMAND ----------

# device OS: select top 4 + others
df_device_operating_system_groups = return_top_n_groups(df = df_device_location,
                                                        col_name = 'device_operating_system',
                                                        n = 4)

# COMMAND ----------

# create device_operating_system_grouped and drop device_operating_system
df_device_location = join_groups(df_join_left = df_device_location,
                                 df_join_right = df_device_operating_system_groups,
                                 join_key = 'device_operating_system',
                                 group_other_name = 'other'
                                )

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create grouped device brand

# COMMAND ----------

# device mobile brand: select top 6 + others
df_device_brand_groups = return_top_n_groups(df = df_device_location,
                                             col_name = 'device_mobile_brand_name',
                                             n = 6)

# COMMAND ----------

# create device_operating_system_grouped and drop device_operating_system
df_device_location = join_groups(df_join_left = df_device_location,
                                 df_join_right = df_device_brand_groups,
                                 join_key = 'device_mobile_brand_name',
                                 group_other_name = 'other'
                                )

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create grouped geo country

# COMMAND ----------

# geo country: select top 5 + others
df_geo_country_groups = return_top_n_groups(df = df_device_location,
                                            col_name = 'geo_country',
                                            n = 5)

# COMMAND ----------

# create device_operating_system_grouped and drop device_operating_system
df_device_location = join_groups(df_join_left = df_device_location,
                                 df_join_right = df_geo_country_groups,
                                 join_key = 'geo_country',
                                 group_other_name = 'other'
                                )

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create isManchester Flag

# COMMAND ----------

df_device_location = (df_device_location
                     .withColumn('isManchesterFlag', 
                                 f.when(
                                   (f.col('geo_city')=="Manchester") 
                                   &
                                   (f.col('geo_country_grouped')=="United Kingdom"),
                                   1)
                                .otherwise(0))
                      .drop('geo_city')
                     )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Previous website before timeframe start

# COMMAND ----------

df_first_event_ts = (df_model_events
                     .groupBy(*unique_key_timeframe)
                     .agg(f.min('event_timestamp'))
                     )

df_previous_website = (df_model_events
                      .join(df_first_event_ts, unique_key_timeframe, 'inner')
                      .filter(f.col('event_timestamp') == f.col('min(event_timestamp)'))
                      .dropDuplicates([*unique_key_timeframe, 'event_timestamp'])
                      .selectExpr(*unique_key_timeframe, 
                                  'traffic_source_source AS previous_website_name', 
                                  'traffic_source_medium AS previous_website_medium'))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Clean and group previous_website_medium

# COMMAND ----------

df_previous_website_medium_clean = (
  df_previous_website
  .withColumn('previous_website_medium_cleaned', 
              f.regexp_replace(f.trim(f.lower(f.col('previous_website_medium'))), '[-.,|\/-_()]', ''))
  .withColumn('previous_website_medium_cleaned',
              f.when(f.col('previous_website_medium_cleaned')=="(none)", "unknown")
              .otherwise(
              f.when(f.col('previous_website_medium_cleaned').rlike("refer"), "referral")
              .otherwise(
              f.when(f.col('previous_website_medium_cleaned').rlike('email'), "email")
              .otherwise(
              f.when(f.col('previous_website_medium_cleaned').rlike('social|facebook|linkedin|twitter|instagram'), "social")
              .otherwise(
              f.when(f.col('previous_website_medium_cleaned').rlike('link'), "link")
              .otherwise(f.col('previous_website_medium_cleaned')))))))
  .drop('previous_website_medium')
  )

# COMMAND ----------

df_previous_website_medium_groups = return_top_n_groups(df = df_previous_website_medium_clean,
                                                        col_name = 'previous_website_medium_cleaned',
                                                        n = 7)

# COMMAND ----------

df_previous_website = join_groups(df_join_left = df_previous_website_medium_clean,
                                  df_join_right = df_previous_website_medium_groups,
                                  join_key = 'previous_website_medium_cleaned',
                                  group_other_name = 'other'
                                 )

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Clean and group previous_website_name

# COMMAND ----------

df_previous_website_name_clean = (
  df_previous_website
  .withColumn('previous_website_name_cleaned',
              f.regexp_replace(f.trim(f.lower(f.col('previous_website_name'))), '[|,()]', ''))
  .withColumn('previous_website_name_cleaned',
              f.regexp_replace(
                f.col('previous_website_name_cleaned'), 
                '.com|.co|.uk|.org|.kr|.fr|.eu|.no|.se|.info|.io|.ua|.pl|.net|.br|.cz|.dk|.cc|.ae|.uz|.tr|.ru|.kz|.by|.bb|www.', 
                ''))
  .withColumn('previous_website_name_cleaned',
              f.when(f.col('previous_website_name_cleaned').rlike('facebook'), 'facebook')
              .otherwise(
              f.when(f.col('previous_website_name_cleaned').rlike('instagram'), 'instagram')
              .otherwise(
              f.when(f.col('previous_website_name_cleaned').rlike('premier'), 'premier_league')
              .otherwise(f.col('previous_website_name_cleaned')))))
  .drop('previous_website_name')
)

# COMMAND ----------

df_previous_website_name_groups = return_top_n_groups(df = df_previous_website_name_clean,
                                                      col_name = 'previous_website_name_cleaned',
                                                      n = 7)

# COMMAND ----------

df_previous_website = join_groups(df_join_left = df_previous_website_name_clean,
                                  df_join_right = df_previous_website_name_groups,
                                  join_key = 'previous_website_name_cleaned',
                                  group_other_name = 'other'
                                 )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Last website of the session

# COMMAND ----------

# df_last_event_ts = (df_model_events
#                    .groupBy(unique_key)
#                    .agg(f.max('event_timestamp'))
#                    )
# 
# df_last_website = (df_model_events
#                   .join(df_last_event_ts, unique_key, 'inner')
#                   .filter(f.col('event_timestamp')==f.col('max(event_timestamp)'))
#                   .dropDuplicates([*unique_key, 'event_timestamp'])
#                   .selectExpr(*unique_key, 'event_params_page_location AS last_website'))

# COMMAND ----------

# df_last_website = (
#   df_last_website
#   .withColumn('last_website_cleaned_grouped',
#              f.when(f.col('last_website').rlike('shop.mancity'), 'retail')
#              .otherwise(
#              f.when(f.col('last_website').rlike('/news'), 'news')
#              .otherwise(
#              f.when(f.col('last_website').rlike('/city-plus'), 'city_plus')
#              .otherwise(
#              f.when(f.col('last_website').rlike('login.mancity.com'), 'registration')
#              .otherwise(
#              f.when(f.col('last_website').rlike('citytv'), 'video')
#              .otherwise(
#              f.when(f.col('last_website').rlike('tickets.mancity|/ticketing-and-hospitality|/tickets|hospitality.mancity'), 'ticketing')
#              .otherwise(
#              f.when(f.col('last_website').rlike('membership'), 'membership')
#              .otherwise(
#              f.when(f.col('last_website').rlike('/fixtures|/results|/tables'), 'fixtures')
#              .otherwise(
#              f.when(f.col('last_website')=="www.mancity.com", 'homepage')
#              .otherwise('other'))))))))))
#   .drop('last_website')
# )

# COMMAND ----------

# possible issue: in the last session before reg "registration" is the most common - may not be very predictive
# --> do not use


# COMMAND ----------

# MAGIC %md
# MAGIC #### Visited content flag

# COMMAND ----------

# display(df_model_events
#         .groupBy(*unique_key, 'event_params_page_location').agg(f.min('event_timestamp'), f.max('event_timestamp'))
#         .withColumn('visited_content_time_spent', (f.col('max(event_timestamp)').cast('double')-f.col('min(event_timestamp)').cast('double'))/60)
#        )

# COMMAND ----------

# create flags for all the items in the dict

df_visited_content_flag = df_model_events

for key, value in content_groups_dict.items():
  df_visited_content_flag = (
    df_visited_content_flag
    .withColumn(f'Visited{key}Flag', f.col('event_params_page_location').rlike(value).cast(T.IntegerType()))
  )

# COMMAND ----------

# handle video, when cityplus = 1 then video -> 0
df_visited_content_flag = (
  df_visited_content_flag
  .withColumn('VisitedVideoFlag', 
              f.when(f.col('VisitedCityPlusFlag')==1, 0)
              .otherwise(f.col('VisitedVideoFlag')))
)

# COMMAND ----------

# group it to timeframes
df_visited_content_flag = (
  df_visited_content_flag
  .groupBy(unique_key_timeframe)
  .sum()
  .drop('sum(timeframe)', 'sum(session)', 'sum(Target)')
)

# COMMAND ----------

for key, value in content_groups_dict.items():
  df_visited_content_flag = (
    df_visited_content_flag
    .withColumn(f'Visited{key}Flag', (f.col(f'sum(Visited{key}Flag)')>0).cast(T.IntegerType()))
    .drop(f'sum(Visited{key}Flag)')
  )

# COMMAND ----------

# Content
# content_flag = 'VisitedFixturesFlag'
# df_unreg_grouped = (df_visited_content_flag
#                     .filter('idm_id_paired = "unknown"')
#                     .groupBy(content_flag)
#                     .count()
#                    )
# 
# df_unreg_rel = (df_unreg_grouped
#                 .withColumn('prop', f.round(f.col('count')/df_unreg_grouped.selectExpr('sum(count)').collect()[0][0], 2))
#                 .withColumn('user', f.lit('unreg')))
# 
# df_reg_t0_grouped = (df_visited_content_flag
#                       .filter('idm_id_paired != "unknown"')
#                      .filter('timeframe = 0')
#                       .groupBy(content_flag)
#                       .count()
#                      )
# 
# df_reg_t0_rel = (df_reg_t0_grouped
#                   .withColumn('prop', f.round(f.col('count')/df_reg_t0_grouped.selectExpr('sum(count)').collect()[0][0], 2))
#                   .withColumn('user', f.lit('reg_t0')))
# 
# df_reg_tx_grouped = (df_visited_content_flag
#                       .filter('idm_id_paired != "unknown"')
#                      .filter('timeframe != 0')
#                       .groupBy(content_flag)
#                       .count()
#                      )
# 
# df_reg_tx_rel = (df_reg_tx_grouped
#                   .withColumn('prop', f.round(f.col('count')/df_reg_tx_grouped.selectExpr('sum(count)').collect()[0][0], 2))
#                   .withColumn('user', f.lit('reg_tx')))
# 
# df_content_all = (df_unreg_rel
#                  .union(
#                    df_reg_tx_rel
#                  )
#                  .union(
#                    df_reg_t0_rel
#                   )
#                  )
# 
# pd_content_all = df_content_all.toPandas()
# pd_content_all[content_flag] = pd_content_all[content_flag].astype('str')
# pd_content_all

# COMMAND ----------

# content_flag_replaced = content_flag.replace('Visited', '').replace('Flag', '')
# fig = px.bar(x = pd_content_all['user'], 
#              y = pd_content_all['prop'], 
#              text = pd_content_all['prop'],
#              barmode = 'stack', 
#              color_discrete_sequence = px.colors.qualitative.Prism[1:3],
#              color = pd_content_all[content_flag],
#              labels = {'x':'Timeframe types', 'y': 'Proportion'},
#              title = f'Proportion of timeframes during which a user visited {content_flag_replaced}')
# 
# fig.update_layout(template = 'plotly_white')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Total time spent

# COMMAND ----------

df_total_time_mins = (df_model_events
                      .groupBy(*unique_key_session, 'timeframe')
                      .agg(
                        ((f.max('event_timestamp').cast('long')-
                          f.min('event_timestamp').cast('long'))
                         /60)
                        .alias('total_time_spent_min'))
                     .groupBy(unique_key_timeframe)
                     .agg(f.sum('total_time_spent_min').alias('total_time_spent_min'))
                     )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Total time spent on videos

# COMMAND ----------

df_total_time_video_mins = (df_model_events
                            .filter('ARRAY_CONTAINS(events_names, "video")')
                            .groupBy(unique_key_timeframe)
                            .agg((f.sum('VideoSeconds')/60).alias('time_spent_on_videos_mins')))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Running average of time spent

# COMMAND ----------

w = Window().partitionBy('idm_id_paired', 'user_pseudo_id').orderBy('timeframe')

df_running_avg_time = (df_total_time_mins
                       .withColumn('running_avg', 
                                   f.avg('total_time_spent_min').over(w))
                       .withColumn('time_minus_running_avg', (f.col('total_time_spent_min') - f.col('running_avg')))
                       .withColumn('change_from_running_avg', f.col('total_time_spent_min')/f.col('running_avg'))
                       .drop('total_time_spent_min', 'time_minus_running_avg', 'running_avg'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Day time of session end

# COMMAND ----------

# not used in the new approach
# df_day_parts = (
#   df_model_events
#   .groupBy(unique_key)
#   .agg(f.max('event_timestamp').alias('last_event_ts'))
#   .withColumn('event_hour', f.hour(f.col('last_event_ts')))
#   .withColumn('day_parts', 
#               f.when((f.col('event_hour')>=0) & (f.col('event_hour')<6), 'night')
#               .otherwise(
#               f.when((f.col('event_hour')>=6) & (f.col('event_hour')<11), 'morning')
#               .otherwise(
#               f.when((f.col('event_hour')>=11) & (f.col('event_hour')<14), 'noon')
#               .otherwise(
#               f.when((f.col('event_hour')>=14) & (f.col('event_hour')<18), 'afternoon')
#               .otherwise('evening')))))
# )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Timeframe happened during a fixture game

# COMMAND ----------

df_fixture_dates = (df_fixtures
                    .filter('Team = "First Team"')
                    .selectExpr('DATE(Date) AS Date')
                    .filter(f.col('Date')>=start_date)
                    .dropDuplicates())

# COMMAND ----------

df_timeframe = (df_model_events
                .groupBy(unique_key_timeframe)
                .agg(f.to_date(f.min('event_timestamp')).alias('timeframe_start'),
                     f.to_date(f.max('event_timestamp')).alias('timeframe_end')))

df_is_fixture = (df_timeframe
                 .join(df_fixture_dates, 
                       [df_timeframe.timeframe_start<=df_fixture_dates.Date,
                        df_timeframe.timeframe_end>=df_fixture_dates.Date], 
                       'left')
                 .dropDuplicates(unique_key_timeframe)
                 .withColumn('isFixture', f.col('Date').isNotNull().cast(T.IntegerType()))
                 .drop('timeframe_start', 'timeframe_end', 'Date')
                )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Final join

# COMMAND ----------

df_model = (df_model_timeframes
            .join(df_number_of_events, unique_key_timeframe, 'left')
            .join(df_number_of_news, unique_key_timeframe, 'left')
            .join(df_number_of_videos, unique_key_timeframe, 'left')
            .join(df_device_location, unique_key_timeframe, 'left')
            .join(df_previous_website, unique_key_timeframe, 'left')
            # .join(df_last_website, unique_key_timeframe, 'left') last session before reg has registration link, do not use
            .join(df_visited_content_flag, unique_key_timeframe, 'left')
            .join(df_total_time_mins, unique_key_timeframe, 'left')
            .join(df_total_time_video_mins, unique_key_timeframe, 'left')
            .join(df_running_avg_time, unique_key_timeframe, 'left')
            .join(df_is_fixture, unique_key_timeframe, 'left')
            .join(df_number_of_sessions, unique_key_timeframe, 'left')
            #.join(df_day_parts, unique_key_timeframe, 'left') do not use with the new timeframe approach
            
            .fillna(0, ['number_of_videos', 'number_of_news', 'total_time_spent_min', 'time_spent_on_videos_mins', 'change_from_running_avg',
                        'VisitedNewsFlag', 'VisitedRetailFlag', 'VisitedVideoFlag', 'VisitedCityPlusFlag', 'VisitedTicketingFlag', 
                        'VisitedMembershipFlag', 'VisitedFixturesFlag',
                        'isManchesterFlag',
                        'isFixture']
                   )
            .fillna('unknown', ['device_category', 'device_operating_system_grouped', 'device_mobile_brand_name_grouped', 
                                'geo_country_grouped', 
                                'previous_website_medium_cleaned_grouped', 'previous_website_name_cleaned_grouped',
                               # 'day_parts',
                                #'last_website_cleaned_grouped'
                               ]
                   )
           )

# df_model.count()-df_model_timeframes.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Additional analyses

# COMMAND ----------

# Quantile plot of number of timeframes of registered users (sample)
# display(df_model
#         .filter('idm_id_paired != "unknown"')
#        .groupBy('idm_id_paired', 'user_pseudo_id')
#        .count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Web visit frequency analysis

# COMMAND ----------

# A: Session length
# On average, one session lasts less than 4 mins, with minimum of 0 mins and maximum of 1724 mins = 28 hrs
# display(df_model_events
#        .groupBy(*unique_key)
#        .agg(f.min('event_timestamp').alias('session_start'),
#             f.max('event_timestamp').alias('session_end'))
#        .withColumn('session_length', 
#                    (f.col('session_end').cast('double') - f.col('session_start').cast('double'))/(60))
#        .summary())

# COMMAND ----------

# B: Time between consecutive sessions
# On average, sessions are apart 78 hrs = 3 days, with minimum of 31 mins and maximum of 2288 hours = 95 days (three months)
# 1st quantile is 3 hrs --> suggested timeframe parameter = 3 hrs between events
# Note: dropping records with very first session
# display(df_model_events
#        .groupBy(*unique_key)
#        .agg(f.min('event_timestamp').alias('session_start'),
#             f.max('event_timestamp').alias('session_end'))
#         .withColumn('previous_session_end', f.lag(f.col('session_end')).over(Window().partitionBy('idm_id_paired', 'user_pseudo_id').orderBy('session_start')))
#        .withColumn('time_between_sessions', 
#                    (f.col('session_start').cast('double') - f.col('previous_session_end').cast('double'))/(60*60))
#         .filter('time_between_sessions IS NOT NULL')
#        .summary())

# COMMAND ----------

# C: Number of sessions per week
# Approximately 1-74 sessions per week are made per user+cookie
# Vast majority of users make no more than 1 session per week using the same cookie
# display(df_model_events
#        .withColumn('week', f.weekofyear(f.col('event_timestamp')))
#         .select('idm_id_paired', 'user_pseudo_id', 'session', 'week')
#         .distinct()
#         .groupBy('idm_id_paired', 'user_pseudo_id', 'week')
#         .count()
#         .summary()
#        )

# COMMAND ----------

# D: Total number of sessions per week
# display(df_model_events
#        .withColumn('week', f.weekofyear(f.col('event_timestamp')))
#         .select('idm_id_paired', 'user_pseudo_id', 'session', 'week')
#         .distinct()
#         .groupBy('idm_id_paired', 'user_pseudo_id', 'week')
#         .count()
#         .groupBy('week')
#         .sum('count')
#         .orderBy('week')
#        )

# COMMAND ----------

# C: Number of sessions per week - Quantiles
# Vast majority of users make no more than 1 session per week using the same cookie
# 10% of users make more than 1 session per week on average
# (df_model_events
#        .withColumn('week', f.weekofyear(f.col('event_timestamp')))
#         .select('idm_id_paired', 'user_pseudo_id', 'session', 'week')
#         .distinct()
#         .groupBy('idm_id_paired', 'user_pseudo_id', 'week')
#         .count()
#         .approxQuantile('count', [0.8, 0.9, 0.95, 0.97], 0)
#        )

# COMMAND ----------

# suggested timeframe parameter: events apart no more than 6 hrs -> timeframe length not more than 4 days

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Time spent per timeframe / session

# COMMAND ----------

# col_of_interest = 'total_time_spent_min'
# 
# df_reg_t0 = (df_model
#              .filter('Target = 1')
#              .select(col_of_interest)
#              .withColumn('series', f.lit('reg_t0')))
# 
# df_reg_tx = (df_model
#              .filter('timeframe != 0 AND idm_id_paired != "unknown"')
#              .select(col_of_interest)
#              .withColumn('series', f.lit('reg_tx')))
# 
# df_notreg = (df_model
#              .filter('idm_id_paired == "unknown"')
#              .select(col_of_interest)
#              .withColumn('series', f.lit('notreg')))
# 
# pd_series = df_reg_t0.union(df_reg_tx).union(df_notreg).toPandas()

# COMMAND ----------

# pd_describe = pd_series[pd_series['series']=='reg_t0'].describe()
# pd_describe[f'{col_of_interest}_reg_t0'] = pd_describe[col_of_interest]
# pd_describe[f'{col_of_interest}_reg_tx'] = pd_series[pd_series['series']=='reg_tx'].describe()
# pd_describe[f'{col_of_interest}_notreg'] = pd_series[pd_series['series']=='notreg'].describe()
# np.round(pd_describe[[f'{col_of_interest}_reg_t0', f'{col_of_interest}_reg_tx', f'{col_of_interest}_notreg']],1)

# COMMAND ----------

# import plotly.express as px
# 
# col_of_interest_name = col_of_interest.replace('_',' ')
# fig = px.histogram(pd_series[pd_series[col_of_interest]<15], 
#                    x = col_of_interest, 
#                    color = "series", 
#                    barmode = "overlay",
#                    title = f'Histogram of {col_of_interest_name} for: <br>unregistered, registered - LTBR, registered - in the previous timeframes',
#                    histnorm='percent',
#                    nbins = 30)
# 
# fig.update_layout(template = 'plotly_white')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### News articles - number

# COMMAND ----------

# col_of_interest = 'number_of_news' # VisitedNewsFlag
# 
# df_reg_t0 = (df_model
#              .filter('Target = 1')
#              .select(col_of_interest)
#              .withColumn('series', f.lit('reg_t0')))
# 
# df_reg_tx = (df_model
#              .filter('timeframe != 0 AND idm_id_paired != "unknown"')
#              .select(col_of_interest)
#              .withColumn('series', f.lit('reg_tx')))
# 
# df_notreg = (df_model
#              .filter('idm_id_paired == "unknown"')
#              .select(col_of_interest)
#              .withColumn('series', f.lit('notreg')))
# 
# pd_series = df_reg_t0.union(df_reg_tx).union(df_notreg).toPandas()
# 
# if col_of_interest == "number_of_news":
#   pd_series = pd_series[pd_series[col_of_interest]>0]

# COMMAND ----------

# pd_describe = pd_series[pd_series['series']=='reg_t0'].describe()
# pd_describe[f'{col_of_interest}_reg_t0'] = pd_describe[col_of_interest]
# pd_describe[f'{col_of_interest}_reg_tx'] = pd_series[pd_series['series']=='reg_tx'].describe()
# pd_describe[f'{col_of_interest}_notreg'] = pd_series[pd_series['series']=='notreg'].describe()
# np.round(pd_describe[[f'{col_of_interest}_reg_t0', f'{col_of_interest}_reg_tx', f'{col_of_interest}_notreg']],1)

# COMMAND ----------

# col_of_interest_name = col_of_interest.replace('_',' ')
# fig = px.histogram(pd_series[pd_series[col_of_interest]<6], 
#                    x = col_of_interest, 
#                    color = "series", 
#                    barmode = "overlay",
#                    title = f'Histogram of {col_of_interest_name} for: <br>unregistered, registered - LTBR, registered - in the previous timeframes',
#                    histnorm='percent',
#                    nbins = 5,
#                   #marginal="rug"
#                   )
# 
# fig.update_layout(template = 'plotly_white')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Videos - number

# COMMAND ----------

# col_of_interest = 'number_of_videos' # VisitedVideoFlag, number_of_videos
# 
# df_reg_t0 = (df_model
#              .filter('Target = 1')
#              .select(col_of_interest)
#              .withColumn('series', f.lit('reg_t0')))
# 
# df_reg_tx = (df_model
#              .filter('timeframe != 0 AND idm_id_paired != "unknown"')
#              .select(col_of_interest)
#              .withColumn('series', f.lit('reg_tx')))
# 
# df_notreg = (df_model
#              .filter('idm_id_paired == "unknown"')
#              .select(col_of_interest)
#              .withColumn('series', f.lit('notreg')))
# 
# pd_series = df_reg_t0.union(df_reg_tx).union(df_notreg).toPandas()
# 
# if col_of_interest == "number_of_videos":
#   pd_series = pd_series[pd_series[col_of_interest]>0]

# COMMAND ----------

# pd_describe = pd_series[pd_series['series']=='reg_t0'].describe()
# pd_describe[f'{col_of_interest}_reg_t0'] = pd_describe[col_of_interest]
# pd_describe[f'{col_of_interest}_reg_tx'] = pd_series[pd_series['series']=='reg_tx'].describe()
# pd_describe[f'{col_of_interest}_notreg'] = pd_series[pd_series['series']=='notreg'].describe()
# np.round(pd_describe[[f'{col_of_interest}_reg_t0', f'{col_of_interest}_reg_tx', f'{col_of_interest}_notreg']],1)

# COMMAND ----------

# col_of_interest_name = col_of_interest.replace('_',' ')
# fig = px.histogram(pd_series[pd_series[col_of_interest]<6], 
#                    x = col_of_interest, 
#                    color = "series", 
#                    barmode = "overlay",
#                    title = f'Histogram of {col_of_interest_name} for: <br>unregistered, registered - LTBR, registered - in the previous timeframes',
#                    histnorm='percent',
#                    nbins = 10,
#                   #marginal="rug"
#                   )
# 
# fig.update_layout(template = 'plotly_white')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Events - number
# MAGIC Need to rerun ntbs 1 and 2 with work_...events table and edit code accordingly

# COMMAND ----------

# display(df_model)

# COMMAND ----------

# col_of_interest = 'number_of_events'  # 'change_from_running_avg', 'number_of_events'
# 
# # Registrovani, tesne pred registraci
# df_reg_t0 = (df_model
#              .filter('timeframe = 0 AND Target = 1')
#              .select(col_of_interest)
#              .withColumn('series', f.lit('reg_t0')))
# 
# # Registrovani, driv nez se registrovali
# df_reg_tx = (df_model
#              .filter('timeframe != 0 AND idm_id_paired != "unknown"')
#              .select(col_of_interest)
#              .withColumn('series', f.lit('reg_tx')))
# 
# # Neregistrovani
# df_notreg = (df_model
#              .filter('idm_id_paired == "unknown"')
#              .select(col_of_interest)
#              .withColumn('series', f.lit('notreg')))
# 
# pd_series = df_reg_t0.union(df_reg_tx).union(df_notreg).toPandas()

# COMMAND ----------

# pd_describe = pd_series[pd_series['series']=='reg_t0'].describe()
# pd_describe[f'{col_of_interest}_reg_t0'] = pd_describe[col_of_interest]
# pd_describe[f'{col_of_interest}_reg_tx'] = pd_series[pd_series['series']=='reg_tx'].describe()
# pd_describe[f'{col_of_interest}_notreg'] = pd_series[pd_series['series']=='notreg'].describe()
# np.round(pd_describe[[f'{col_of_interest}_reg_t0', f'{col_of_interest}_reg_tx', f'{col_of_interest}_notreg']],1)

# COMMAND ----------

# import plotly.express as px
# col_of_interest_name = col_of_interest.replace('_',' ')
# fig = px.histogram(pd_series[pd_series[f'{col_of_interest}']<100], 
#                    x = col_of_interest, 
#                    color = "series", 
#                    barmode = "overlay",
#                    title = f'Histogram of {col_of_interest_name} for: <br>unregistered, registered - LTBR, registered - in the previous timeframes',
#                    histnorm='percent')
# 
# fig.update_layout(template = 'plotly_white')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Fixtures - most common

# COMMAND ----------

# display(df_fixtures.withColumn('Date', f.to_date('Date')).dropDuplicates().filter('Team = "First Team"').groupBy('Result').count())

# COMMAND ----------

# # Fixtures - registrations
# # Registration counts
# df_reg_counts = (df_timeframe
#                  .join(df_fixture_dates, 
#                        [df_timeframe.timeframe_start<=df_fixture_dates.Date,
#                         df_timeframe.timeframe_end>=df_fixture_dates.Date], 
#                        'left')
#                  .dropDuplicates(unique_key_timeframe)
#                  .withColumn('isFixture', f.col('Date').isNotNull().cast(T.IntegerType()))
#         .filter('timeframe = 0 AND idm_id_paired != "unknown"')
#        .filter('isFixture = 1')
#        .groupBy('Date')
#        .count()
#        .filter('Date<"2021-10-27"')
#        .orderBy('Date')
#                 )
# ##Counts by fixture result / competition
# display(df_reg_counts
#       .join(df_fixtures.withColumn('Date', f.to_date('Date')).dropDuplicates().filter('Team = "First Team"'), ['Date'], 'inner')
#       .groupBy('Competition') #competition / result
#       .agg(f.count('*'),
#            f.sum('count'))
#       .orderBy(f.desc('sum(count)'))
#        )
# 

# COMMAND ----------

# # Fixtures - registrations
# # Registration counts - On MatchDay Analysis
# df_reg_counts = (df_timeframe
#                  .join(df_fixture_dates, 
#                        df_timeframe.timeframe_end==df_fixture_dates.Date, 
#                        'left')
#                  .dropDuplicates(unique_key_timeframe)
#                  .withColumn('isFixture', f.col('Date').isNotNull().cast(T.IntegerType()))
#         .filter('timeframe = 0 AND idm_id_paired != "unknown"')
#        .filter('isFixture = 1')
#        .groupBy('Date')
#        .count()
#        .filter('Date<"2021-10-27"')
#        .orderBy('Date')
#                 )
# ##Counts by fixture result / competition
# display(df_reg_counts
#       .join(df_fixtures.withColumn('Date', f.to_date('Date')).dropDuplicates().filter('Team = "First Team"'), ['Date'], 'inner')
#       .groupBy('Competition') #competition / result
#       .agg(f.count('*'),
#            f.sum('count'))
#       .orderBy(f.desc('sum(count)'))
#        )
# 

# COMMAND ----------

# Total number of all registrations
#display(df_timeframe
#                 .join(df_fixture_dates, 
#                       [df_timeframe.timeframe_start<=df_fixture_dates.Date,
#                        df_timeframe.timeframe_end>=df_fixture_dates.Date], 
#                       'left')
#                 .dropDuplicates(unique_key_timeframe)
#                 .withColumn('isFixture', f.col('Date').isNotNull().cast(T.IntegerType()))
#        .filter('timeframe = 0 AND idm_id_paired != "unknown"')
#       .filter('isFixture = 1')
#       .groupBy('Date')
#       .count()
#       .filter('Date<"2021-10-27"')
#        .selectExpr('sum(count)'))

# COMMAND ----------

# fixtures with registration counts
#display(df_timeframe
#        .join(df_fixture_dates, 
#              [df_timeframe.timeframe_start<=df_fixture_dates.Date,
#               df_timeframe.timeframe_end>=df_fixture_dates.Date], 
#              'left')
#        .dropDuplicates(unique_key_timeframe)
#        .withColumn('isFixture', f.col('Date').isNotNull().cast(T.IntegerType()))
#        .filter('timeframe = 0 AND idm_id_paired != "unknown"')
#        .filter('isFixture = 1')
#        .groupBy('Date')
#        .count()
#        .filter('Date<"2021-10-27"')
#        .orderBy('Date')
#        .join((df_fixtures.filter('Team = "First Team"').filter(f.col('Date')>=start_date)
#               .withColumn('Date', f.to_date('Date'))
#               .dropDuplicates()), ['Date'], 'inner')
#        .withColumn('Competition', f.when(f.col('Competition').contains('Friendly'), 'Friendly').otherwise(f.col('Competition')))
#       #.groupBy('Competition')
#       #.count()
#       #.orderBy(f.col('count').desc())
#       )

# COMMAND ----------

# On matchday : fixtures with registration counts
# display(df_timeframe
#         .join(df_fixture_dates, 
#               df_timeframe.timeframe_end==df_fixture_dates.Date, 
#               'left')
#         .dropDuplicates(unique_key_timeframe)
#         .withColumn('isFixture', f.col('Date').isNotNull().cast(T.IntegerType()))
#         .filter('timeframe = 0 AND idm_id_paired != "unknown"')
#         .filter('isFixture = 1')
#         .groupBy('Date')
#         .count()
#         .filter('Date<"2021-10-27"')
#         .orderBy('Date')
#         .join((df_fixtures.filter('Team = "First Team"').filter(f.col('Date')>=start_date)
#                .withColumn('Date', f.to_date('Date'))
#                .dropDuplicates()), ['Date'], 'inner')
#         .withColumn('Competition', f.when(f.col('Competition').contains('Friendly'), 'Friendly').otherwise(f.col('Competition')))
#         .orderBy('Date')
#        #.groupBy('Competition')
#        #.count()
#        #.orderBy(f.col('count').desc())
#        )

# COMMAND ----------

# display(df_fixtures
#                     .filter('Team = "First Team"')
#                     .filter(f.col('Date')>=start_date)
#                     .dropDuplicates()
#        .filter('DATE(Date) = "2021-08-21" OR DATE(Date)="2021-07-27" OR DATE(Date)="2021-07-31"'))

# COMMAND ----------

# List of fixtures for the rules
# display(df_fixtures
#         .filter('Team = "First Team"')
#         #.selectExpr('DATE(Date) AS Date')
#         .filter(f.col('Date')>=run_date)
#         .dropDuplicates()
#         .filter('LOWER(Competition) = "premier league"')
#        )

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Fixtures by day of week

# COMMAND ----------

# pd_fixtures_days = (df_fixture_dates
#         .withColumn('day_of_week', f.dayofweek('Date'))
#         .groupBy('day_of_week')
#         .count()
#                     .orderBy('day_of_week')
#         .toPandas())

# COMMAND ----------

# pd_fixtures_days.loc[4] = [2, 0]
# pd_fixtures_days.loc[5] = [5, 0]
# pd_fixtures_days.loc[6] = [6, 0]
# pd_fixtures_days = pd_fixtures_days.sort_values(by = 'day_of_week')
# pd_fixtures_days['days'] = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

# COMMAND ----------

# import plotly.express as px
# fig = px.bar(x = pd_fixtures_days['days'], y = pd_fixtures_days['count'],
#              labels = {'x':'Days', 'y': ''},
#              title = 'First Team Fixture count by day of week')
# fig.update_layout(template = 'plotly_white')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Registration development

# COMMAND ----------

# # Registration development
# display(df_model_events
#        .filter('timeframe = 0')
#          .filter('idm_id_paired != "unknown"')
#         .dropDuplicates(['idm_id_paired'])
#         .withColumn('date', f.to_date('event_timestamp'))
#         .withColumn('week', f.weekofyear('event_timestamp'))
#         .filter('date<"2021-11-05"')
#         .groupBy('date')
#         .count()
#         .withColumn('cum_sum', f.sum('count').over(Window.orderBy('date')))
#        )

# COMMAND ----------

# # Summary stats of weekly registration increments
# display(df_model_events
#        .filter('timeframe = 0')
#          .filter('idm_id_paired != "unknown"')
#         .dropDuplicates(['idm_id_paired'])
#         .withColumn('date', f.to_date('event_timestamp'))
#         .withColumn('week', f.weekofyear('event_timestamp'))
#         .filter('date<"2021-11-05"')
#         .groupBy('week')
#         .count()
#         .withColumn('cum_sum', f.sum('count').over(Window.orderBy('week')))
#         #.select('count')
#         .summary()
#        )
