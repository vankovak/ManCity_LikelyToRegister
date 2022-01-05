# Databricks notebook source
# MAGIC %md
# MAGIC ## SHAP application

# COMMAND ----------

dbutils.widgets.text("sampleSize", "2000")
sample_size = int(dbutils.widgets.get("sampleSize"))
print(f"Sample of size {sample_size} will be used for SHAP application.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run ntb

# COMMAND ----------

# MAGIC %run "./07_make_predictions"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Install shap

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install shap

# COMMAND ----------

import shap

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare data for shap explainer object

# COMMAND ----------

# Select sample for SHAP
sample_ratio = sample_size/pd_X_train.shape[0]

pd_X_trainsample, pd_X_rest, pd_y_trainsample, pd_y_rest = train_test_split(pd_X_train, 
                                                                            pd_y_train, 
                                                                            train_size = sample_ratio, 
                                                                            random_state = 42, 
                                                                            stratify = pd_y_train)

samples = pd_X_trainsample

# COMMAND ----------

explainer = shap.LinearExplainer(sklearn_lr_model, samples, feature_perturbation = 'interventional')

shap_values = explainer.shap_values(samples)

# COMMAND ----------

individual_person = 10
displayHTML(shap.getjs() + shap.force_plot(explainer.expected_value, shap_values[individual_person], samples.iloc[individual_person,:]).html())

# COMMAND ----------

shap.summary_plot(shap_values, samples, max_display = 60)

# COMMAND ----------

samples_edit = []
for col in samples.columns:
  col = col.replace('cleaned_', '')
  col = col.replace('grouped_', '')
  samples_edit.append(col)

# COMMAND ----------

pd_mean_shap = pd.DataFrame(zip(np.absolute(shap_values).mean(axis=0).tolist(), samples_edit), 
                            columns = ['Mean SHAP value', 'Feature'])


# COMMAND ----------

# Define feature impact
pd_mean_shap['Impact'] = pd_mean_shap['Feature'].map({
  'number_of_events' : 'positive',
 'number_of_news' : 'non-trivial',
 'number_of_videos' : 'non-trivial',
 'time_spent_on_videos_mins' : 'negative',
 'change_from_running_avg' : 'positive',
 'number_of_sessions' : 'negative',
 'isManchesterFlag' : 'non-trivial',
 'VisitedNewsFlag' : 'negative',
 'VisitedRetailFlag' : 'negative',
 'VisitedVideoFlag' : 'negative',
 'VisitedCityPlusFlag' : 'positive',
 'VisitedTicketingFlag' : 'negative',
 'VisitedMembershipFlag' : 'positive',
 'VisitedFixturesFlag' : 'negative',
 'isFixture' : 'positive',
 'device_category_desktop' : 'negative',
 'device_category_mobile' : 'positive',
 'device_category_tablet' : 'non-trivial',
 'device_operating_system_Android' : 'positive',
 'device_operating_system_Macintosh' : 'positive', 
 'device_operating_system_Windows' : 'negative',
 'device_operating_system_iOS' : 'non-trivial',
 'device_operating_system_other' : 'negative',
 'device_mobile_brand_name_Apple' : 'positive',
 'device_mobile_brand_name_Google' : 'non-trivial',
 'device_mobile_brand_name_Huawei' : 'negative',
 'device_mobile_brand_name_Microsoft' : 'positive',
 'device_mobile_brand_name_Samsung' : 'negative',
 'device_mobile_brand_name_Xiaomi' : 'non-trivial',
 'device_mobile_brand_name_other' : 'positive',
 'geo_country_India' : 'positive',
 'geo_country_South Africa' : 'positive',
 'geo_country_South Korea' : 'negative',
 'geo_country_United Kingdom' : 'non-trivial',
 'geo_country_United States' : 'negative',
 'geo_country_other' : 'positive',
 'previous_website_medium_cpc' : 'positive',
 'previous_website_medium_email' : 'negative',
 'previous_website_medium_link' : 'negative',
 'previous_website_medium_none' : 'positive',
 'previous_website_medium_organic' : 'negative',
 'previous_website_medium_other' : 'negative',
 'previous_website_medium_referral' : 'non-trivial',
 'previous_website_medium_social' : 'positive',
 'previous_website_name_direct' : 'positive',
 'previous_website_name_facebook' : 'positive',
 'previous_website_name_google' : 'positive',
 'previous_website_name_instagram' : 'positive',
 'previous_website_name_mancity_app' : 'negative',
 'previous_website_name_other' : 'positive',
 'previous_website_name_premier_league' : 'negative',
 'previous_website_name_twitter' : 'negative'
                                                      })

# COMMAND ----------

# Define what features will be included in the figure
pd_mean_shap['Include'] = pd_mean_shap['Feature'].map({
  'number_of_events' : 1,
 'number_of_news' : 0,
 'number_of_videos' : 0,
 'time_spent_on_videos_mins' : 0,
 'change_from_running_avg' : 1,
 'number_of_sessions' : 0,
 'isManchesterFlag' : 0,
 'VisitedNewsFlag' : 1,
 'VisitedRetailFlag' : 1,
 'VisitedVideoFlag' : 1,
 'VisitedCityPlusFlag' : 1,
 'VisitedTicketingFlag' : 1,
 'VisitedMembershipFlag' : 1,
 'VisitedFixturesFlag' : 1,
 'isFixture' : 1,
 'device_category_desktop' : 1,
 'device_category_mobile' : 1,
 'device_category_tablet' : 0,
 'device_operating_system_Android' : 1,
 'device_operating_system_Macintosh' : 1, 
 'device_operating_system_Windows' : 1,
 'device_operating_system_iOS' : 0,
 'device_operating_system_other' : 0,
 'device_mobile_brand_name_Apple' : 1,
 'device_mobile_brand_name_Google' : 0,
 'device_mobile_brand_name_Huawei' : 1,
 'device_mobile_brand_name_Microsoft' : 1,
 'device_mobile_brand_name_Samsung' : 1,
 'device_mobile_brand_name_Xiaomi' : 0,
 'device_mobile_brand_name_other' : 0,
 'geo_country_India' : 1,
 'geo_country_South Africa' : 1,
 'geo_country_South Korea' : 0,
 'geo_country_United Kingdom' : 0,
 'geo_country_United States' : 1,
 'geo_country_other' : 0,
 'previous_website_medium_cpc' : 1,
 'previous_website_medium_email' : 1,
 'previous_website_medium_link' : 1,
 'previous_website_medium_none' : 0,
 'previous_website_medium_organic' : 1,
 'previous_website_medium_other' : 0,
 'previous_website_medium_referral' : 0,
 'previous_website_medium_social' : 1,
 'previous_website_name_direct' : 1,
 'previous_website_name_facebook' : 1,
 'previous_website_name_google' : 1,
 'previous_website_name_instagram' : 1,
 'previous_website_name_mancity_app' : 1,
 'previous_website_name_other' : 0,
 'previous_website_name_premier_league' : 1,
 'previous_website_name_twitter' : 1
                                                      })

# COMMAND ----------

# Define groups
pd_mean_shap['Group'] = pd_mean_shap['Feature'].map({
  'number_of_events' : 'engagement',
 'number_of_news' : 'engagement',
 'number_of_videos' : 'engagement',
 'time_spent_on_videos_mins' : 'engagement',
 'change_from_running_avg' : 'engagement',
 'number_of_sessions' : 'engagement',
 'isManchesterFlag' : 'location',
 'VisitedNewsFlag' : 'content',
 'VisitedRetailFlag' : 'content',
 'VisitedVideoFlag' : 'content',
 'VisitedCityPlusFlag' : 'content',
 'VisitedTicketingFlag' : 'content',
 'VisitedMembershipFlag' : 'content',
 'VisitedFixturesFlag' : 'content',
 'isFixture' : 'fixture',
 'device_category_desktop' : 'device_category',
 'device_category_mobile' : 'device_category',
 'device_category_tablet' : 'device_category',
 'device_operating_system_Android' : 'device_os',
 'device_operating_system_Macintosh' : 'device_os', 
 'device_operating_system_Windows' : 'device_os',
 'device_operating_system_iOS' : 'device_os',
 'device_operating_system_other' : 'device_os',
 'device_mobile_brand_name_Apple' : 'device_brand',
 'device_mobile_brand_name_Google' : 'device_brand',
 'device_mobile_brand_name_Huawei' : 'device_brand',
 'device_mobile_brand_name_Microsoft' : 'device_brand',
 'device_mobile_brand_name_Samsung' : 'device_brand',
 'device_mobile_brand_name_Xiaomi' : 'device_brand',
 'device_mobile_brand_name_other' : 'device_brand',
 'geo_country_India' : 'location',
 'geo_country_South Africa' : 'location',
 'geo_country_South Korea' : 'location',
 'geo_country_United Kingdom' : 'location',
 'geo_country_United States' : 'location',
 'geo_country_other' : 'location',
 'previous_website_medium_cpc' : 'referrer_medium',
 'previous_website_medium_email' : 'referrer_medium',
 'previous_website_medium_link' : 'referrer_medium',
 'previous_website_medium_none' : 'referrer_medium',
 'previous_website_medium_organic' : 'referrer_medium',
 'previous_website_medium_other' : 'referrer_medium',
 'previous_website_medium_referral' : 'referrer_medium',
 'previous_website_medium_social' : 'referrer_medium',
 'previous_website_name_direct' : 'referrer_name',
 'previous_website_name_facebook' : 'referrer_name',
 'previous_website_name_google' : 'referrer_name',
 'previous_website_name_instagram' : 'referrer_name',
 'previous_website_name_mancity_app' : 'referrer_name',
 'previous_website_name_other' : 'referrer_name',
 'previous_website_name_premier_league' : 'referrer_name',
 'previous_website_name_twitter' : 'referrer_name'
                                                      })

# COMMAND ----------

pd_mean_shap['sign_numeric'] = \
pd_mean_shap['Impact'].map({'positive':1, 
                            'negative':-1, 
                            'non-trivial':1})

# COMMAND ----------

pd_mean_shap['Mean SHAP value w sign'] = np.exp(pd_mean_shap['sign_numeric'] * pd_mean_shap['Mean SHAP value'])-1

# COMMAND ----------

import plotly.express as px
pd_filtered_table = pd_mean_shap[
  (pd_mean_shap['Impact']=='positive') & 
  (pd_mean_shap['Include']==1) #&
  #(pd_mean_shap['Group']=='content')
]

fig = px.bar(pd_filtered_table, 
             x='Mean SHAP value w sign', 
             y = 'Feature', 
             color = 'Impact',
             text = np.round(pd_filtered_table['Mean SHAP value w sign'],3),
             color_discrete_map={'positive':'rgb(30,190,30)', 
                                 'negative':'rgb(254,46,60)', 
                                 'non-trivial': 'rgb(130,130,130)'})

fig.update_layout(yaxis={'categoryorder':'total ascending', 'title': ''},
                  xaxis = {'title':''},
                  template = 'plotly_white',
                  title = f'Average impact on the probability of registration',
                  font={'size':10},
                  width = 1100, 
                  height = 400)

fig.update_traces(textposition='outside',textfont_size=8)

fig.update_layout(legend=dict(yanchor="bottom",
                              y=0.05,
                              #y = 0.85,
                              xanchor="right",
                              x=0.95
                             #x = 0.15
                             )
                 )


fig.show()

# COMMAND ----------

# Visualizing top features

for feature in samples.columns:
    shap.dependence_plot(feature, shap_values, samples, interaction_index=None)

# COMMAND ----------

# Visualizing top features with interactions

for i, feature in enumerate(samples.columns):
  for int_feature in samples.columns[i:]:
    shap.dependence_plot(feature, shap_values, samples, interaction_index=int_feature)

# COMMAND ----------


