# Databricks notebook source
# MAGIC %md
# MAGIC ## Feature selection

# COMMAND ----------

dbutils.widgets.text("allowedCorrelation", "0.6")
allowed_corr_limit = float(dbutils.widgets.get("allowedCorrelation"))
print(f"Allowed correlation between numerical and boolean columns is {allowed_corr_limit}.")

dbutils.widgets.text("categoricalMinIV", "0.01")
categorical_min_iv_value = float(dbutils.widgets.get("categoricalMinIV"))
print(f"Categorical columns with IV less than {categorical_min_iv_value} will be dropped from dataframe.")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run ntbs

# COMMAND ----------

# MAGIC %run "./03_create_features"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define functions

# COMMAND ----------

def calculate_woe_iv(dataset, feature, target):
    """Calculate WoE and IV.

    Args:
        dataset (df): Spark dataframe with columns consisting of a feature of interest and target.
        feature (str): Name of a column in the dataset which represents the feature which IV we want to calculate.
        target (str): Name of a column in the dataset which represents the target.
    Returns:
        Information value (int).
    """
    
    lst = []
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': dataset[dataset[feature] == val].count()[feature],
            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],
            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]
        })
        
    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    iv = dset['IV'].sum()
    
    return iv

# COMMAND ----------

def information_value_analysis(pd_model: pd.DataFrame,
                               feature_type: str,  #'numerical', 'binary'
                               cols: list,
                               target: str,
                               allowed_corr_limit: int,
                               pairs_for_inf_value_analysis: list
                              ):
  
  if feature_type == "numerical":
    # Prepare data
    target_numerical_cols = list(pd_model.columns)
    pd_quantiles = (pd_model
                    .drop(target, axis=1)
                    .quantile(list(np.linspace(0, 1, 9)))
                   )
    
    # Define bins for numerical features
    pd.options.mode.chained_assignment = None
    for num in cols:
      bins = list(pd_quantiles[num])
      pd_model[num+'_Bins'] = pd.cut(pd_model[num], bins = bins)
        
    pd_model = pd_model.drop(cols, axis = 1)
    
  elif feature_type == 'binary':
    pass
  
  # Calculate IV
  cols_iv_dict = {}
  cols_bins = list(pd_model.columns).copy()
  cols_bins.remove(target)
  
  for col in cols_bins:
    cols_iv_dict[col] = calculate_woe_iv(pd_model[[target, col]], col, target)
    cols_iv_dict[col.replace('_Bins','')] = cols_iv_dict[col]
    if '_Bins' in col:
      cols_iv_dict.pop(col)
  
  # Compare IVs of each pair and define list of columns to remove
  cols_to_remove = []
  
  for cols_tuple in pairs_for_inf_value_analysis:
    iv_left_col = cols_iv_dict[cols_tuple[0]]
    iv_right_col = cols_iv_dict[cols_tuple[1]]
   
    if iv_left_col > iv_right_col:
      if cols_tuple[0] not in cols_to_remove:  # if the col with higher IV is not already to be removed, then remove col with lower IV
        cols_to_remove.append(cols_tuple[1])
    else:
      if cols_tuple[1] not in cols_to_remove:  # if the col with higher IV is not already to be removed, then remove col with lower IV
        cols_to_remove.append(cols_tuple[0]) 
    
  print('Columns to remove:', list(set(cols_to_remove)))
  return list(set(cols_to_remove))
  

# COMMAND ----------

def df_correlated_features(df: f.DataFrame,
                           feature_type: str,  #'numerical', 'binary'
                           cols: list,
                           target: str,
                           allowed_corr_limit: int,
                          ):
  # Calculate correlation metric
  pd_model = df.select(*cols, target).toPandas()
    
  if feature_type == 'numerical':
    # Pearson correlation matrix
    num_features = df.select(cols).rdd.map(lambda row: row[0:])
    corr_mat = Statistics.corr(num_features, method = 'pearson')
    pd_corr_metric = pd.DataFrame(corr_mat)
    pd_corr_metric.index, pd_corr_metric.columns = numerical_cols, numerical_cols
    
  elif feature_type == 'binary':
    # Jaccard matrix
    pd_model_w_target = pd_model.copy()
    pd_model = pd_model.drop(target, axis = 1)
    pd_corr_metric = pd.DataFrame(columns = pd_model.columns)
    
    for row in pd_model.columns:
      jaccard_dict = {}
      for col in pd_model.columns: 
        jaccard_dict[col] = 1 - jaccard(pd_model[row], pd_model[col])
      pd_corr_metric = pd_corr_metric.append(jaccard_dict, ignore_index = True)
    pd_corr_metric.index = pd_corr_metric.columns
    
    pd_model = pd_model_w_target.copy()
    
  # Get highly correlated pairs
  pairs_for_inf_value_analysis = []

  for index, row in enumerate(cols):
    for column in cols[index+1:]:
      if row!=column:
        corr_value = pd_corr_metric.loc[row, column]
        if corr_value > allowed_corr_limit:
          print('High correlation betweeen:', row,'and', column, ' : ', corr_value)
          pairs_for_inf_value_analysis.append((row, column))
  
  print('Correlated pairs: ', pairs_for_inf_value_analysis)
  
  # Create list of unique columns with high correlation
  if len(pairs_for_inf_value_analysis)==0:
    print('No correlation issues')
    return []
  else:
    print('Information Value Analysis')
  
  cols_list = []
  for cols_tuple in pairs_for_inf_value_analysis:
    for pair in cols_tuple:
      if pair not in cols_list:
        cols_list.append(pair)
  
  cols_list_target = cols_list.copy()
  cols_list_target.append(target)
  
  cols_out = information_value_analysis(pd_model = pd_model[cols_list_target],
                                        feature_type = feature_type,
                                        cols = cols_list,
                                        target = target,
                                        allowed_corr_limit = allowed_corr_limit,
                                        pairs_for_inf_value_analysis = pairs_for_inf_value_analysis)
  
 
  return cols_out

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Setup

# COMMAND ----------

df_model.cache().count()

# COMMAND ----------

display(df_model)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Check nulls

# COMMAND ----------

for col in df_model.columns:
  null_value_count = df_model.filter(f'{col} IS NULL').count()
  if null_value_count>0:
    print(col, null_value_count)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define column types

# COMMAND ----------

unique_key = ['idm_id_paired', 'user_pseudo_id', 'timeframe']
target = 'Target'

categorical_cols = ['device_category',
                    'device_operating_system_grouped',
                    'device_mobile_brand_name_grouped',
                    'geo_country_grouped',
                    'previous_website_medium_cleaned_grouped',
                    'previous_website_name_cleaned_grouped']

numerical_cols = ['number_of_events',
                  'number_of_news',
                  'number_of_videos',
                  'total_time_spent_min',
                  'time_spent_on_videos_mins',
                  'change_from_running_avg',
                  'number_of_sessions']

binary_cols = ['isManchesterFlag',
               'VisitedNewsFlag',
               'VisitedRetailFlag',
               'VisitedVideoFlag',
               'VisitedCityPlusFlag',
               'VisitedTicketingFlag',
               'VisitedMembershipFlag',
               'VisitedFixturesFlag',
               'isFixture']

# COMMAND ----------

# check whether any columns are missing in lists
set(df_model.columns)^set(unique_key)^set(categorical_cols)^set(numerical_cols)^set(binary_cols)^set([target])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Handle numerical and binary features

# COMMAND ----------

num_cols_to_drop = df_correlated_features(df = df_model, 
                                          feature_type = 'numerical', 
                                          cols = numerical_cols, 
                                          target = target,
                                          allowed_corr_limit = allowed_corr_limit)

# COMMAND ----------

binary_cols_to_drop = df_correlated_features(df = df_model, 
                                             feature_type = 'binary', 
                                             cols = binary_cols, 
                                             target = target,
                                             allowed_corr_limit = allowed_corr_limit)

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Handle categorical cols

# COMMAND ----------

pd_categorical = df_model.select(*categorical_cols, target).toPandas()

# COMMAND ----------

# Check IV of categorical cols
cat_cols_to_drop = []

for col in categorical_cols:
  calc_iv = calculate_woe_iv(pd_categorical[[target, col]], col, target)
  print(col, ':', calc_iv)
  if calc_iv < categorical_min_iv_value:
    cat_cols_to_drop.append(col)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Select columns in df_model

# COMMAND ----------

df_model = df_model.drop(*num_cols_to_drop, *binary_cols_to_drop, *cat_cols_to_drop)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Write table

# COMMAND ----------

df_model.write.mode('overwrite').saveAsTable('kv_df_features')

# COMMAND ----------

for num_col in num_cols_to_drop:
  numerical_cols.remove(num_col)

# COMMAND ----------

for cat_col in cat_cols_to_drop:
  categorical_cols.remove(cat_col)

# COMMAND ----------

for bin_col in binary_cols_to_drop:
  binary_cols.remove(bin_col)

# COMMAND ----------

data = [{'categorical_cols' : categorical_cols, 
         'binary_cols' : binary_cols, 
         'numerical_cols': numerical_cols}]

# COMMAND ----------

display(spark.createDataFrame(data))

# COMMAND ----------

# Save col names to hive
spark.createDataFrame(data).write.mode('overwrite').saveAsTable('kv_likely_to_register_cols')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Additional analyses

# COMMAND ----------

# import pyspark.sql.functions as f
# import plotly.express as px
# import pandas as pd
# import numpy as np
# 
# df_features = spark.table('kv_df_features')

# COMMAND ----------

# display(df_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Number of events

# COMMAND ----------

# col_of_interest = 'number_of_events'  # 'change_from_running_avg', 'number_of_events'
# 
# # Registrovani, tesne pred registraci
# df_reg_t0 = (df_features
#              .filter('timeframe = 0 AND Target = 1')
#              .select(col_of_interest)
#              .withColumn('series', f.lit('reg_t0')))
# 
# # Registrovani, driv nez se registrovali
# df_reg_tx = (df_features
#              .filter('timeframe != 0 AND idm_id_paired != "unknown"')
#              .select(col_of_interest)
#              .withColumn('series', f.lit('reg_tx')))
# 
# # Neregistrovani
# df_notreg = (df_features
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

# col_of_interest_name = col_of_interest.replace('_',' ')
# fig = px.histogram(pd_series[pd_series[f'{col_of_interest}']<100], 
#                    x = col_of_interest, 
#                    color = "series", 
#                    barmode = "overlay",
#                    title = f'Histogram of {col_of_interest_name} for: <br>unregistered, registered - LTBR, registered - in the previous timeframes',
#                    histnorm='percent')
# 
# fig.update_layout(template = 'plotly_white')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Change from running avg

# COMMAND ----------

# col_of_interest = 'change_from_running_avg'  # 'change_from_running_avg', 'number_of_events'
# 
# # Registrovani, tesne pred registraci
# df_reg_t0 = (df_features
#              .filter('timeframe = 0 AND Target = 1')
#              .select(col_of_interest)
#              .withColumn('series', f.lit('reg_t0')))
# 
# pd_series = df_reg_t0.toPandas()

# COMMAND ----------

# display(df_reg_t0)

# COMMAND ----------

# col_of_interest_name = col_of_interest.replace('_',' ')
# fig = px.histogram(pd_series, 
#                    x = col_of_interest, 
#                    color = "series", 
#                    barmode = "overlay",
#                    title = f'Histogram of {col_of_interest_name} for registered - LTBR',
#                    histnorm='percent',
#                   nbins = 20)
# 
# fig.update_layout(template = 'plotly_white')
