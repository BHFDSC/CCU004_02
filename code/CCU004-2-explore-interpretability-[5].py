# Databricks notebook source
# MAGIC %md
# MAGIC **Description** This notebook explores applying methods to make models more interpretable for CCU004-2
# MAGIC  
# MAGIC **Project(s)** CCU004-2 - A nationwide deep learning pipeline to predict stroke and COVID-19 death in atrial fibrillation 
# MAGIC  
# MAGIC **Author(s)** Alex Handy
# MAGIC 
# MAGIC **Reviewer(s)** TBD
# MAGIC  
# MAGIC **Date last updated** 24-01-2022

# COMMAND ----------

# MAGIC %run /Workspaces/dars_nic_391419_j3w9t_collab/CCU004/CCU004_2/CCU004-2-global-helper-functions

# COMMAND ----------

#set upfront parameters

#INDIVIDUAL TEST SCENARIO FOR MOST PERFORMANT MODEL
outcomes = ["stroke"]
max_seq_lens = [100]
sample_ratios = [1]
runs = [1]

input_run_date = "240122"

# COMMAND ----------

#helper functions and packages

from datetime import datetime
import math
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, recall_score, precision_score, accuracy_score
import time
import xgboost as xgb
import shap

#to stop setting copy warning output - consider reviewing in refactoring
pd.set_option('chained',None)


##MACHINE LEARNING METHODS

def create_ml_features(x, feature_list, codelist, target_field, outcome):
    entry = {}
    for code in codelist:
        if code in x[target_field]:
            entry[code] = 1
        else:
            entry[code] = 0
    
    if outcome == "stroke":
        entry["age_at_af_diagnosis"] = x["age_at_af_diagnosis"]
    else:
        entry["age_at_cohort_start"] = x["age_at_cohort_start"]
      
    entry["female"] = x["female"]
    entry["white"] = x["white"]
    entry["asian_or_asian_british"] = x["asian_or_asian_british"]
    entry["black_or_black_british"] = x["black_or_black_british"]
    entry["mixed"] = x["mixed"]
    entry["other_ethnic_groups"] = x["other_ethnic_groups"]
    
    entry[outcome] = x[outcome]

    feature_list.append(entry)
    
def calc_ml_metrics(prediction, target):
    try:
      tn, fp, fn, tp = confusion_matrix(target, prediction).ravel()
      accuracy = (tp + tn) / (tp + tn + fp + fn)
      auc = roc_auc_score(target, prediction)
      sensitivity = tp / (tp + fn)
      specificity = tn / (tn + fp)
      precision = tp / (tp + fp)
    except ValueError:
      print("Predicting all one class")
      accuracy = 0
      auc =  0
      sensitivity = 0
      specificity = 0
      precision = 0
    
    return accuracy, auc, sensitivity, specificity, precision
    
def evaluate_ml_models(models, x_train, x_val, y_train, y_val, cohort_test_sub_non_n, all_codes_non_n, num_static_features, outcome, summary_data):
  for i, model in enumerate(models):
    entry = {}

    md = model.fit(x_train, y_train)

    pred = md.predict(x_val)
    
    if isinstance(model,sklearn.linear_model._logistic.LogisticRegression):
      model_name = "Logistic Regression"
    elif isinstance(model,sklearn.ensemble._forest.RandomForestClassifier):
      model_name = "Random Forest"
    elif isinstance(model, xgb.XGBRegressor):
      model_name = "XG Boost"
      pred = [ 1 if p >= 0.5 else 0 for p in pred ]
    else:
      print("Incorrect model type")    
    

    print("Validation sample results")
    accuracy, auc, sensitivity, specificity, precision = calc_ml_metrics(pred, y_val)
    
    print("Model:", model_name)
    
    print("Accuracy (val): ", accuracy)
    print("Auc (val): ", auc)
    print("Sensitivity (val):", sensitivity)
    print("Specificity (val):", specificity)
    print("Precision (val):", precision)
    
    
    print("Test sample results - whole group")
    pred_test = md.predict(cohort_test_sub_non_n.iloc[:, :(len(all_codes_non_n)+num_static_features)])
    y_test = cohort_test_sub_non_n[outcome]
    

    if isinstance(model, xgb.XGBRegressor):
      pred_test = [ 1 if p >= 0.5 else 0 for p in pred_test ]
    
    accuracy_test, auc_test, sensitivity_test, specificity_test, precision_test = calc_ml_metrics(pred_test, y_test)
    
    entry["model"] = model_name
    entry["accuracy"] = accuracy_test
    entry["auc"] = auc_test
    entry["sensitivity"] = sensitivity_test
    entry["specificity"] = specificity_test
    entry["precision"] = precision_test
    
    
    print("Accuracy (test): ", accuracy_test)
    print("Auc (test): ", auc_test)
    print("Sensitivity (test):", sensitivity_test)
    print("Specificity (test):", specificity_test)
    print("Precision (test):", precision_test)
    
    summary_data.append(entry)
    
    return md

# COMMAND ----------

#main script

start = time.time()
print("Script started ", datetime.fromtimestamp(start))

for outcome in outcomes:
  print("Outcome: ", outcome)
  
  #define static features
  if outcome == "stroke":
    STATIC_FEATURES = ["age_at_af_diagnosis", "female", "white", "asian_or_asian_british", "black_or_black_british", "mixed", "other_ethnic_groups"]
  else:
    STATIC_FEATURES = ["age_at_cohort_start", "female", "white", "asian_or_asian_british", "black_or_black_british", "mixed", "other_ethnic_groups"]
    
  NUM_STATIC_FEATURES = len(STATIC_FEATURES)
  print("Number of static features", NUM_STATIC_FEATURES)
  
  for run in runs:
    print("Run: ", run)
    #load the test table
    print("Load the test table for: ", outcome, " and run ", run)
    cohort_test_sub_py_export_table_name = "ccu004_2_cohort_" + outcome + "_seq_len_all_run_" + str(run) + "_test_sub_" + input_run_date
    cohort_test_sub_py = spark.table("dars_nic_391419_j3w9t_collab." + cohort_test_sub_py_export_table_name)
    cohort_test_sub = cohort_test_sub_py.toPandas()
    print("Test sub rows", len(cohort_test_sub))
    
    
    for sample_ratio in sample_ratios:
      print("Sample ratio: ", sample_ratio)
      #load the train table
      print("Load the train table for: ", outcome, " and run ", run, "and sample ratio ", sample_ratio)
      cohort_train_sub_py_export_table_name = "ccu004_2_cohort_" + outcome + "_seq_len_all_sr_" + str(sample_ratio) + "_run_" + str(run) + "_train_sub_" + input_run_date
      cohort_train_sub_py = spark.table("dars_nic_391419_j3w9t_collab." + cohort_train_sub_py_export_table_name)
      cohort_train_sub = cohort_train_sub_py.toPandas()
      print("Train sub rows", len(cohort_train_sub))
      
      
      for max_seq_len in max_seq_lens:
        print("Max seq len: ", max_seq_len)
        
        #setup summary data for each scenario
        print("Setup summary data for ", outcome, " and run ", run, " and sample ratio ", sample_ratio, "and max seq len ", max_seq_len)
        summary_data = []
        summary_data_sub = []
        
        #define max seq len field 
        if max_seq_len == 60:
          TARGET_FIELD = "med_hist_target_60"
        else:
          TARGET_FIELD = "med_hist_target_100"
          
        print("Target field: ", TARGET_FIELD)
        
        #create universal vocab of medical codes
        training_codes_non_n = [item for sublist in cohort_train_sub[TARGET_FIELD].values for item in sublist]
        training_codelist_non_n = list(set(training_codes_non_n))
        print("Codelist vocab length training non-neural", len(training_codelist_non_n))

        test_codes_non_n = [item for sublist in cohort_test_sub[TARGET_FIELD].values for item in sublist]
        test_codelist_non_n = list(set(test_codes_non_n))
        print("Codelist vocab length test non-neural", len(test_codelist_non_n))

        all_codes_non_n = list(set(training_codelist_non_n + test_codelist_non_n))
        print("Codelist vocab length non-neural", len(all_codes_non_n))
        
        #create features for ml models
        print("Create ml features")
        start_ml = time.time()
        
        print("ML started ", datetime.fromtimestamp(start_ml))
        train_features_non_n = []
        test_features_non_n = []
        cohort_train_sub.apply(create_ml_features, args=(train_features_non_n,all_codes_non_n, TARGET_FIELD, outcome), axis=1)
        cohort_test_sub.apply(create_ml_features, args=(test_features_non_n,all_codes_non_n, TARGET_FIELD, outcome), axis=1)

        cohort_train_sub_non_n = pd.DataFrame(train_features_non_n)
        cohort_test_sub_non_n = pd.DataFrame(test_features_non_n)
        
        print("Check dimensions of ml features")
        print("Ml train features shape: ", cohort_train_sub_non_n.shape)
        print("Ml test features shape: ", cohort_test_sub_non_n.shape)
        
        #generate test train splits
        x_train, x_val, y_train, y_val = train_test_split(cohort_train_sub_non_n.iloc[:, :(len(all_codes_non_n)+NUM_STATIC_FEATURES)], cohort_train_sub_non_n[outcome], test_size=0.20, random_state=run)
        ml_models = [xgb.XGBRegressor(objective="binary:logistic", random_state=run)]
        md = evaluate_ml_models(ml_models, x_train, x_val, y_train, y_val, cohort_test_sub_non_n, all_codes_non_n, NUM_STATIC_FEATURES, outcome, summary_data)
        
        end_ml = time.time()
        print("ML completed in %s minutes" % ( round(end_ml - start_ml,2) / 60) )
        
end = time.time()
print("Script completed in %s minutes" % ( round(end - start,2) / 60) )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calibration analysis

# COMMAND ----------

#mean calibration

print("Observed prevalence: ",cohort_test_sub_non_n["stroke"].values.mean())
x_test = cohort_test_sub_non_n.iloc[:, :-1]
y_test_pred = md.predict(x_test)

print("Predicted prevalence: ", y_test_pred.mean())

# COMMAND ----------

#calibration plots

from sklearn.calibration import calibration_curve
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from matplotlib.pyplot import figure

xg_y, xg_x = calibration_curve(cohort_test_sub_non_n["stroke"].values, y_test_pred, n_bins=10)

fig, ax = plt.subplots()
plt.plot(xg_x,xg_y)
line = mlines.Line2D([0, 1], [0, 1], color='black')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
fig.suptitle('Calibration plot for predicted probability of stroke vs the true probability of stroke')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability')
plt.autoscale(False)
fig.set_size_inches(9, 6, forward=True)
figure(dpi=300)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Interpretability analysis

# COMMAND ----------

snomed = spark.table("dss_corporate.snomed_sct2_description_full")
snomed_df = snomed.toPandas()
#snomed_df.head()

icd10 = spark.table("dss_corporate.icd10_group_chapter_v01")
icd10_df = icd10.toPandas()
#icd10_df.head()

# COMMAND ----------

#replace x_train columns with lookup term
#NOTE: slow using pyspark collect so converted into pandas which still seems to take a while to run

new_col_names = []
curr_col_names = x_test.columns
for col in curr_col_names:
  if col in STATIC_FEATURES:
    new_col_names.append(col)
  else:
    if len(col) > 4:
      try:
        snomed_name = snomed_df[snomed_df["conceptId"] == col]["term"].values[0]
        new_col_names.append(snomed_name)
      except:
        new_col_names.append(col)
    else:
      try:
        icd10_name = icd10_df[icd10_df["ALT_CODE"] == col]["ICD10_DESCRIPTION"].values[0]
        new_col_names.append(icd10_name)
      except:
        new_col_names.append(col)
      
print(new_col_names[-20:])

# COMMAND ----------

#shap values - aggregate
x_test.columns = new_col_names
xg_explainer = shap.Explainer(md)
xg_shap_values = xg_explainer(x_test)
shap.summary_plot(xg_shap_values, x_test, max_display=20)

# COMMAND ----------

#compare overall scores against feature importances
coef_dict = {}
for coef, feat in zip(md.feature_importances_,list(x_test.columns)):
    coef_dict[feat] = coef
    
sorted_dict = sorted(coef_dict.items(), key=lambda kv: kv[1], reverse=True)

top_20 = [sorted_dict[0:20]]
top_20

# COMMAND ----------

#select index of correct prediction at random
import random
stroke_idxs = [ i for i, x in enumerate(cohort_test_sub_non_n["stroke"].values) if (x == 1) & (y_test_pred[i] > 0.5) ]
no_stroke_idxs = [ i for i, x in enumerate(cohort_test_sub_non_n["stroke"].values) if (x == 0) & (y_test_pred[i] < 0.5) ]
s_idx = random.choice(stroke_idxs)
ns_idx = random.choice(no_stroke_idxs)

print(s_idx, ns_idx)

#check predictions were correct
print("Should be zero: ", cohort_test_sub_non_n["stroke"].values[ns_idx])
print("Prediction confidence: ", y_test_pred[ns_idx])

print("Should be one: ", cohort_test_sub_non_n["stroke"].values[s_idx])
print("Prediction confidence: ", y_test_pred[s_idx])

# COMMAND ----------

#shap values - individuals
#no stroke
shap.plots.waterfall(xg_shap_values[ns_idx])

# COMMAND ----------

#shap values - individuals
#stroke
shap.plots.waterfall(xg_shap_values[s_idx])
