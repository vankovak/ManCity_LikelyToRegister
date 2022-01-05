# Databricks notebook source
# MAGIC %md
# MAGIC ## Predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run ntb

# COMMAND ----------

# MAGIC %run "./06_fit_model" $makingPredictions="True"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define functions

# COMMAND ----------

vectorElement = udf(lambda v:float(v[1]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Make predictions

# COMMAND ----------

# Load model
spark_logged_model = 'runs:/1fa8bf29c478469192593edbb62fbeef/log_Reg_model'
sklearn_logged_model = 'runs:/b644294d2699464e8beda495b18ba60b/sklearn_log_Reg_model'


# Load model as a Spark UDF.
spark_lr_model = mlflow.spark.load_model(model_uri=spark_logged_model)
sklearn_lr_model = mlflow.sklearn.load_model(model_uri=sklearn_logged_model)

# COMMAND ----------

# Transform test and train dataset
df_test_trans = spark_lr_model.transform(df_test)
df_train_trans = spark_lr_model.transform(df_train)

df_test_trans = df_test_trans.withColumn('prob_1', vectorElement('probability'))
df_train_trans = df_train_trans.withColumn('prob_1', vectorElement('probability'))

# COMMAND ----------

# Transform test and train dataset
pd_y_test_pred = sklearn_lr_model.predict(pd_X_test)
pd_y_train_pred = sklearn_lr_model.predict(pd_X_train)

pd_y_test_pred_proba = sklearn_lr_model.predict_proba(pd_X_test)
pd_y_train_pred_proba = sklearn_lr_model.predict_proba(pd_X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Probability distribution

# COMMAND ----------

pd_test_results = (df_test_trans
                   .select('label','prediction', f.round(f.col('prob_1'),6).alias('prob_1'))
                   .sort('prob_1', ascending=False)
                   .toPandas()
                  )

# COMMAND ----------

#Sklearn
import matplotlib.pyplot as plt
plt.hist(pd_y_test_pred_proba[:,1])
plt.show()

# COMMAND ----------

#Spark
import matplotlib.pyplot as plt
plt.hist(pd_test_results['prob_1'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### SKLEARN - Lift curve

# COMMAND ----------

pd_y_test_concat = pd.DataFrame([], columns=['probability', 'label'])
pd_y_test_concat['probability'] = pd_y_test_pred_proba[:,1]
pd_y_test_concat['label'] = np.array(pd_y_test)

# COMMAND ----------

display(lift_curve_sklearn(spark.createDataFrame(pd_y_test_concat), 10))

# COMMAND ----------

pd_test_lift = lift_curve_sklearn(spark.createDataFrame(pd_y_test_concat), 10).toPandas()
fig = plt.figure(1)
plt.bar(pd_test_lift['bucket'], pd_test_lift['cum_lift'])
plt.show()

# COMMAND ----------

pd_y_train_concat = pd.DataFrame([], columns=['probability', 'label'])
pd_y_train_concat['probability'] = pd_y_train_pred_proba[:,1]
pd_y_train_concat['label'] = np.array(pd_y_train)

# COMMAND ----------

display(lift_curve_sklearn(spark.createDataFrame(pd_y_train_concat), 10))

# COMMAND ----------

pd_train_lift = lift_curve_sklearn(spark.createDataFrame(pd_y_train_concat), 10).toPandas()
fig = plt.figure(1)
plt.bar(pd_train_lift['bucket'], pd_train_lift['cum_lift'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### SPARK - Lift curve

# COMMAND ----------

display(lift_curve(df_train_trans, 10))

# COMMAND ----------

pd_train_lift = lift_curve(df_train_trans, 10).toPandas()
fig = plt.figure(1)
plt.bar(pd_train_lift['bucket'], pd_train_lift['cum_lift'])
plt.show()

# COMMAND ----------

display(lift_curve(df_test_trans, 10))

# COMMAND ----------

pd_test_lift = lift_curve(df_test_trans, 10).toPandas()
fig = plt.figure(1)
plt.bar(pd_test_lift['bucket'], pd_test_lift['cum_lift'])
plt.show()
