# Databricks notebook source
# MAGIC %md
# MAGIC ## Model fitting

# COMMAND ----------

dbutils.widgets.text("makingPredictions", "False")
make_predictions = str(dbutils.widgets.get("makingPredictions"))
print(f"Making predictions: {make_predictions}.")

dbutils.widgets.text("sparkOrSklearn", "spark")
model_to_fit = str(dbutils.widgets.get("sparkOrSklearn"))
print(f"Fitting model: {model_to_fit}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup

# COMMAND ----------

import mlflow.sklearn

from pyspark.sql import Window
import matplotlib.pyplot as plt
import pandas as pd

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from sklearn.model_selection import GridSearchCV as sklearn_gscv
from sklearn.model_selection import StratifiedKFold as sklearn_skf
from sklearn.linear_model import LogisticRegression as sklearn_lr
from sklearn.metrics import roc_auc_score

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run ntb

# COMMAND ----------

# MAGIC %run "./05_feature_pipeline"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define functions

# COMMAND ----------

def lift_curve(predictions, 
               bin_count):
  
    vectorElement = udf(lambda v:float(v[1]))
    lift_df = (predictions
               .select(vectorElement('probability').cast('float').alias('probability'), 'label')
               .withColumn('rank', f.ntile(bin_count).over(Window.orderBy(f.desc("probability"))))
               .select('probability', 'rank', 'label')
               .groupBy('rank')
               .agg(f.count('label').alias('bucket_row_number'), 
                    f.sum('label').alias('bucket_lead_number'), 
                    f.avg('probability').alias('avg_model_lead_probability'))
               .withColumn('cum_avg_leads', 
                           f.avg('bucket_lead_number')
                           .over(Window.orderBy('rank').rangeBetween(Window.unboundedPreceding, 0)))
              )
    
    avg_lead_rate = (lift_df
                     .filter(f.col('rank')==bin_count)
                     .select('cum_avg_leads')
                     .collect()[0]
                     .cum_avg_leads
                    )
    
    cum_lift_df = (lift_df
                   .withColumn('cum_lift', f.col('cum_avg_leads').cast('float')/avg_lead_rate)
                   .selectExpr('rank as bucket', 
                               'bucket_row_number', 
                               'bucket_lead_number', 
                               'avg_model_lead_probability', 
                               'cum_avg_leads', 
                               'cum_lift')
                  )

    return cum_lift_df

# COMMAND ----------

def lift_curve_sklearn(predictions, 
               bin_count):
  
    vectorElement = udf(lambda v:float(v[1]))
    lift_df = (predictions
               .select('probability', 'label')
               .withColumn('rank', f.ntile(bin_count).over(Window.orderBy(f.desc("probability"))))
               .select('probability', 'rank', 'label')
               .groupBy('rank')
               .agg(f.count('label').alias('bucket_row_number'), 
                    f.sum('label').alias('bucket_lead_number'), 
                    f.avg('probability').alias('avg_model_lead_probability'))
               .withColumn('cum_avg_leads', 
                           f.avg('bucket_lead_number')
                           .over(Window.orderBy('rank').rangeBetween(Window.unboundedPreceding, 0)))
              )
    
    avg_lead_rate = (lift_df
                     .filter(f.col('rank')==bin_count)
                     .select('cum_avg_leads')
                     .collect()[0]
                     .cum_avg_leads
                    )
    
    cum_lift_df = (lift_df
                   .withColumn('cum_lift', f.col('cum_avg_leads').cast('float')/avg_lead_rate)
                   .selectExpr('rank as bucket', 
                               'bucket_row_number', 
                               'bucket_lead_number', 
                               'avg_model_lead_probability', 
                               'cum_avg_leads', 
                               'cum_lift')
                  )

    return cum_lift_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Target weight

# COMMAND ----------

target_ones = df_train.filter(f.col('Target')==1).count()
target_zeros = df_train.filter(f.col('Target')==0).count()

weight = target_ones/target_zeros

df_train = (df_train
            .withColumn('weight', f.when(f.col('Target')==1, 1-weight).otherwise(weight))
           )

print('Target == 1 obtains weight: ', round(1-weight,2), 'and Target == 0: ', round(1-round(1-weight, 2), 2))

# COMMAND ----------

df_train = df_train.withColumnRenamed('Target','label')
df_test = df_test.withColumnRenamed('Target','label')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit model

# COMMAND ----------

# MAGIC %md
# MAGIC #### Spark + Logistic Regression

# COMMAND ----------

if make_predictions == "False" and model_to_fit == 'spark':
  with mlflow.start_run(experiment_id = 784972351544863) as run:
    mlflow.log_param('model_name', 'logistic_regression')
    evaluator = BinaryClassificationEvaluator()
    lr = LogisticRegression(featuresCol='features', maxIter=20, weightCol='weight')
    paramGrid = (ParamGridBuilder()
                 .addGrid(lr.elasticNetParam,[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1])
                 .addGrid(lr.regParam, [0.5, 0.1, 0.01])
                 .build()
                )

    pipeline = Pipeline(stages = [lr])
    model_pipeline = CrossValidator(estimator = pipeline,
                                    estimatorParamMaps = paramGrid,
                                    evaluator = evaluator,
                                    numFolds = 5) 

    model = model_pipeline.fit(df_train)
    mlflow.spark.log_model(model,'log_Reg_model')    

    train_trans = model.transform(df_train)
    test_trans = model.transform(df_test)

    lift_df = lift_curve(model.transform(df_test), 10).toPandas()
    fig = plt.figure(1)
    plt.bar(lift_df['bucket'], lift_df['cum_lift'])
    lift_1 = lift_df.loc[0,'cum_lift']
    mlflow.log_metric('lift_1',lift_1) 
    mlflow.log_param('weight', weight)
    mlflow.log_metrics(dict(zip(featuresNames, model.bestModel.stages[0].coefficients )))

    #log AUC for train and test
    auc_train = (evaluator
                 .evaluate(train_trans
                           .withColumnRenamed('Target','label'), 
                           {evaluator.metricName: 'areaUnderROC'})
                )

    mlflow.log_metric('AUC_train', auc_train)

    auc_test = (evaluator
                .evaluate(test_trans
                          .withColumnRenamed('Target','label'), 
                          {evaluator.metricName: 'areaUnderROC'})
               )

    mlflow.log_metric('AUC_test', auc_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pandas + Logistic Regression

# COMMAND ----------

if make_predictions == "False" and model_to_fit == 'sklearn':
  with mlflow.start_run(experiment_id = 784972351544863) as run:
    mlflow.log_param('model_name', 'sklearn_logistic_regression')

    model = sklearn_lr(C = 0.01).fit(pd_X_train, pd_y_train)
  
    mlflow.sklearn.log_model(model,'sklearn_log_Reg_model')  
    
    # Calculate lift curve for test dataset and save it
    pd_y_test_pred_proba = model.predict_proba(pd_X_test)
    pd_y_train_pred_proba = model.predict_proba(pd_X_train)
    
    pd_y_test_concat = pd.DataFrame([], columns=['probability', 'label'])
    pd_y_test_concat['probability'] = pd_y_test_pred_proba[:,1]
    pd_y_test_concat['label'] = np.array(pd_y_test)
    
    pd_test_lift = lift_curve_sklearn(spark.createDataFrame(pd_y_test_concat), 10).toPandas()
    fig = plt.figure(1)
    plt.bar(pd_test_lift['bucket'], pd_test_lift['cum_lift'])
    lift_1 = lift_df.loc[0,'cum_lift']
    
    mlflow.log_metric('lift_1',lift_1) 
    
    # Save AUC metrics
    mlflow.log_metric('AUC_train', roc_auc_score(pd_y_train, pd_y_train_pred_proba[:,1]))
    mlflow.log_metric('AUC_test', roc_auc_score(pd_y_test, pd_y_test_pred_proba[:,1]))
    
    # Save features with their coefficients
    featuresNames = pd_X_train.columns
    mlflow.log_metrics(dict(zip(featuresNames, model.coef_[0])))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Coefficients

# COMMAND ----------

if make_predictions == "False" and model_to_fit == 'spark':
  pd_coefs = (pd.DataFrame(list(zip(model.bestModel.stages[-1].coefficients, 
                          featuresNames)), 
                 columns=['effect on model decision','name'])
              .sort_values('effect on model decision', ascending = False)
             )
  
  df_coefs = spark.createDataFrame(pd_coefs) 
  display(df_coefs)

# COMMAND ----------

if make_predictions == "False" and model_to_fit == 'sklearn':
  pd_coefs = (pd.DataFrame(list(zip(model.coef_[0], 
                          featuresNames)), 
                 columns=['effect on model decision','name'])
              .sort_values('effect on model decision', ascending = False)
             )
  
  df_coefs = spark.createDataFrame(pd_coefs) 
  display(df_coefs)
