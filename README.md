# ManCity_LikelyToRegister
Code for the project Likely To Register in ManCity

There are 10 notebooks containing code for the preparation of the data, feature selection, model fitting and explanation of model outputs. The code was uploaded as is, i.e. including some additional analyses which were created for the purpose of presentation only (usually these blocks are located at the bottom of the file).

For more information about website personalisation (references, logic description, ...) see notion page at https://www.notion.so/datasentics/Web-Conversion-Rate-Optimisation-4249f66394a84ae38a8e3fc167ad4476#883ff398adcd4ef2a904fc569c522425.

- `01_data_preprocessing.py` preprocesses data (and selects relevant columns for feature selection etc.
- `02_create_data_model.py` creates a data matrix (i.e. each row corresponds to a person using some device during a specified time horizon, target is also calculated)
- `03_create_features.py` creates features
- `04_feature_selection.py` selects features (based on correlation, jaccard, information value)
- `05_feature_pipeline.py` creates a transformation pipeline for features (i.e. vectorization, scaling, one hot encoding etc.). there are two sections, first one creates pipeline in spark, second one in pandas
- `06_fit_model.py` fits model, again there are two sections: one fits spark model, another one fits sklearn model using mlflow
- `07_make_predictions.py` briefly evaluates model's predictive performance and lift curve
- `08_shap.py` applies shap library on sklearn model to extract insights
- `09_additional_analyses.py` performs factor analysis to group pre-selected features into factors (for the purpose of GA audiences creation)
- `10_evaluate_change.py` analyzes registration rates of users from control and treatment group
