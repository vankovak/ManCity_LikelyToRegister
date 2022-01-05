# Databricks notebook source
# MAGIC %md
# MAGIC ## Feature pipeline

# COMMAND ----------

dbutils.widgets.text("trainSplitRatio", "0.7")
train_ratio = float(dbutils.widgets.get("trainSplitRatio"))
print(f"Train set will contain {train_ratio} portion of the original dataset and test set will contain 1 - {train_ratio} = {round(1-train_ratio, 2)} of the dataset.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup

# COMMAND ----------

import os

import pyspark.sql.functions as f
import pyspark.sql.types as T

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler as sklearn_mms
from sklearn.compose import ColumnTransformer as sklearn_ct
from sklearn.preprocessing import OneHotEncoder as sklearn_ohe
from sklearn.model_selection import train_test_split

from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors

# MLflow
import mlflow
import mlflow.spark

# COMMAND ----------

df_features = spark.table('kv_df_features')
columns = spark.table('kv_likely_to_register_cols')

# COMMAND ----------

categorical_cols = columns.select('categorical_cols').collect()[0][0]
numerical_cols = columns.select('numerical_cols').collect()[0][0]
binary_cols = columns.select('binary_cols').collect()[0][0]

# COMMAND ----------

# MAGIC %md
# MAGIC ### SPARK - Feature pipeline 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train test split

# COMMAND ----------

df_train, df_test = df_features.randomSplit([train_ratio, 1-train_ratio], 42)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train test to Pandas

# COMMAND ----------

pd_X_train = df_train.select(*categorical_cols, *numerical_cols, *binary_cols).toPandas()
pd_y_train = df_train.selectExpr('Target AS label').toPandas()

pd_X_test = df_test.select(*categorical_cols, *numerical_cols, *binary_cols).toPandas()
pd_y_test = df_test.selectExpr('Target AS label').toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Categorical features

# COMMAND ----------

# MAGIC %md
# MAGIC ##### String Indexer

# COMMAND ----------

si_cols = [feature+'_index' for feature in categorical_cols]

StrIndexer = StringIndexer(inputCols = categorical_cols, 
                           outputCols = si_cols, 
                           handleInvalid='skip', 
                           stringOrderType='frequencyAsc')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### One Hot Encoder

# COMMAND ----------

ohe_cols = [feature + '_ohe' for feature in categorical_cols]

OHE = OneHotEncoder(inputCols = si_cols, 
                    outputCols = ohe_cols, 
                    dropLast = True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Numerical features

# COMMAND ----------

for feature in list(set(numerical_cols).union(set(binary_cols))):
  df_train = (df_train
              .withColumn(feature, f.col(feature).cast('double'))
             )
  
  df_test = (df_test
             .withColumn(feature, f.col(feature).cast('double'))
            )

# COMMAND ----------

# MAGIC %md
# MAGIC #### All features

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Vectorization

# COMMAND ----------

vectorization = VectorAssembler(inputCols = numerical_cols + binary_cols + ohe_cols,
                                outputCol = 'raw_features', 
                                handleInvalid = 'skip')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Scaling

# COMMAND ----------

std_scaler = StandardScaler(inputCol = 'raw_features', 
                            outputCol = 'features')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Fit feature pipeline

# COMMAND ----------

with mlflow.start_run(experiment_id = 784972351544861) as run:
  pipeline = (Pipeline(stages = [StrIndexer, OHE, vectorization, std_scaler])
              .fit(df_train)
             )
  
  mlflow.spark.log_model(pipeline,'features_pipeline')
#  mlflow.spark.save_model(pipeline)
  mlflow.log_param('model_name', 'feature_prep')
  
  df_train = pipeline.transform(df_train)           

  featuresNames = (
  [
  attr["name"] for attr in (df_train
                            .select("raw_features")
                            .schema[0]
                            .metadata["ml_attr"]["attrs"]["numeric"])
  ]+ 
  [
  attr["name"] for attr in (df_train
                            .select("raw_features")
                            .schema[0]
                            .metadata["ml_attr"]["attrs"]["binary"])
  ])


# COMMAND ----------

# MAGIC %md
# MAGIC #### Transform test df

# COMMAND ----------

df_test = pipeline.transform(df_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### PANDAS - Feature pipeline 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### One Hot Encoder

# COMMAND ----------

pd_X_train = pd.get_dummies(pd_X_train, columns=categorical_cols)
pd_X_test = pd.get_dummies(pd_X_test, columns=categorical_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Scaling

# COMMAND ----------

sklearn_scaler = sklearn_mms().fit(pd_X_train[[*numerical_cols, *binary_cols]])
pd_X_train[[*numerical_cols, *binary_cols]] = sklearn_scaler.transform(pd_X_train[[*numerical_cols, *binary_cols]])
pd_X_test[[*numerical_cols, *binary_cols]] = sklearn_scaler.transform(pd_X_test[[*numerical_cols, *binary_cols]])
