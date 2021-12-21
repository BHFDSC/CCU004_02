# Databricks notebook source
# MAGIC %md
# MAGIC **Description** This notebook runs the model analysis pipeline for CCU004-2
# MAGIC  
# MAGIC **Project(s)** CCU004-2 - A nationwide deep learning pipeline to predict stroke and COVID-19 death in atrial fibrillation 
# MAGIC  
# MAGIC **Author(s)** Alex Handy
# MAGIC 
# MAGIC **Reviewer(s)** Chris Tomlinson, Hiu Yan (Samantha) Ip
# MAGIC  
# MAGIC **Date last updated** 20-12-2021

# COMMAND ----------

# MAGIC %run /Workspaces/dars_nic_391419_j3w9t_collab/CCU004/CCU004_2/CCU004-2-global-helper-functions

# COMMAND ----------

#set upfront parameters

#TARGET SCENARIOS
outcomes = ["stroke", "covid_death"]
max_seq_lens = [100]
sample_ratios = [1]
runs = [1,2,3]

input_run_date = "301121"
output_run_date = "301121"

scenarios = len(outcomes) * len(max_seq_lens) * len(sample_ratios) * len(runs)

print(scenarios)

SUB_GROUPS = ["female", "male", "lt_65", "gte_65", "white", "asian_or_asian_british", "black_or_black_british", "mixed", "other_ethnic_groups"]

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
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch import Tensor
from torch.utils.data import dataset
from typing import Tuple
import xgboost as xgb

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
    
def evaluate_ml_models(models, x_train, x_val, y_train, y_val, cohort_test_sub_non_n, all_codes_non_n, num_static_features, outcome, summary_data, summary_data_sub, sub_groups):
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
    
    print("Test sample results - sub groups")
    entry_sub = {}
    entry_sub["model"] = model_name
    
    if outcome == "stroke":
      age_col = "age_at_af_diagnosis"
    else:
      age_col = "age_at_cohort_start"
    
    for sub_group in sub_groups:
      if sub_group == "male":
        sub_group_test_df = cohort_test_sub_non_n[cohort_test_sub_non_n["female"] == 0]
      elif sub_group == "gte_65":
        sub_group_test_df = cohort_test_sub_non_n[cohort_test_sub_non_n[age_col] >=65]
      elif sub_group == "lt_65":
        sub_group_test_df = cohort_test_sub_non_n[cohort_test_sub_non_n[age_col] <65]
      else:
        sub_group_test_df = cohort_test_sub_non_n[cohort_test_sub_non_n[sub_group] == 1]
        
      pred_test_sub = md.predict(sub_group_test_df.iloc[:, :(len(all_codes_non_n)+num_static_features)])
      y_test_sub = sub_group_test_df[outcome]
      

      if isinstance(model, xgb.XGBRegressor):
        pred_test_sub = [ 1 if p >= 0.5 else 0 for p in pred_test_sub ]
      
      accuracy_test_sub, auc_test_sub, sensitivity_test_sub, specificity_test_sub, precision_test_sub = calc_ml_metrics(pred_test_sub, y_test_sub)
      
      entry_sub[str("accuracy" + "_" + sub_group)] = accuracy_test_sub
      entry_sub[str("auc" + "_" + sub_group)] = auc_test_sub
      entry_sub[str("sensitivity" + "_" + sub_group)] = sensitivity_test_sub
      entry_sub[str("specificity" + "_" + sub_group)] = specificity_test_sub
      entry_sub[str("precision" + "_" + sub_group)] = precision_test_sub
      
    
    summary_data_sub.append(entry_sub)


##DEEP LEARNING METHODS
    
#data preparation
def lookup_embeddings(x, code_to_ix, target_field):
    med_hist_entry = []
    for code in x[target_field]:
        emb = code_to_ix[code]
        med_hist_entry.append(emb)
    
    return med_hist_entry

def add_label(x, target_field):
  if x[target_field] == 1:
    return target_field
  else:
    return "No " + target_field

def create_dl_features(cohort_train_sub, cohort_test_sub, code_to_ix_nn, TARGET_FIELD, STATIC_FEATURES, outcome, run):
  #NOTE: ASSUMES SET PADDING IDX TO ZERO IN EMBEDDING LAYER
  seq_features_nn = cohort_train_sub.apply(lookup_embeddings, args=(code_to_ix_nn, TARGET_FIELD), axis=1)
  seq_features_nn_tn = [ torch.tensor(seq) for seq in seq_features_nn ]
  seq_features_nn_tn_pd = pad_sequence(seq_features_nn_tn, batch_first=True)

  #add static features
  static_df = cohort_train_sub[STATIC_FEATURES]
  static_features_nn = torch.tensor([ row for row in static_df.values ])

  #generate outcome labels for training data
  OUTCOME_CAT = outcome + "_cat"
  cohort_train_sub[OUTCOME_CAT] = cohort_train_sub.apply(add_label, args=(outcome,), axis=1)

  all_categories = list(cohort_train_sub[OUTCOME_CAT].unique())
  all_categories.sort()
  #CHECK SORT SO LABELLING MAKES SENSE FOR OUTCOMES e.g. 0=No stroke, 1=stroke
  print("DL categories: ", all_categories)
  n_categories = len(all_categories)
  labels_nn = cohort_train_sub[OUTCOME_CAT].apply(lambda x: all_categories.index(x))
  labels_nn_tn = torch.tensor(labels_nn)

  #create train, validation and test dataset
  x_seq_train, x_seq_val, x_static_train, x_static_val, y_train, y_val =  train_test_split(seq_features_nn_tn_pd, static_features_nn, labels_nn_tn,test_size=0.2, random_state=run)

  #test sample
  seq_features_test_nn = cohort_test_sub.apply(lookup_embeddings, args=(code_to_ix_nn, TARGET_FIELD), axis=1)
  seq_features_test_nn_tn = [ torch.tensor(seq) for seq in seq_features_test_nn ]
  seq_features_test_nn_tn_pd = pad_sequence(seq_features_test_nn_tn, batch_first=True)
  x_seq_test = seq_features_test_nn_tn_pd

  static_df_test = cohort_test_sub[STATIC_FEATURES]
  static_features_test_nn = torch.tensor([ row for row in static_df_test.values ])
  x_static_test = static_features_test_nn

  cohort_test_sub[OUTCOME_CAT] = cohort_test_sub.apply(add_label, args=(outcome,), axis=1)
  labels_test_nn = cohort_test_sub[OUTCOME_CAT].apply(lambda x: all_categories.index(x))
  labels_test_nn_tn = torch.tensor(labels_test_nn)
  y_test = labels_test_nn_tn

  print("x seq train", x_seq_train.size())
  print("x static train", x_static_train.size())
  print("y train", y_train.size())

  print("x seq val", x_seq_val.size())
  print("x static val", x_static_val.size())
  print("y val", y_val.size())

  print("x seq test", x_seq_test.size())
  print("x static test", x_static_test.size())
  print("y test", y_test.size())

  return x_seq_train, x_seq_val, x_static_train, x_static_val, y_train, y_val, x_seq_test, x_static_test, y_test, all_categories, n_categories, OUTCOME_CAT

def create_dl_sub_sample(cohort_test_sub, sub_group, code_to_ix_nn, TARGET_FIELD, STATIC_FEATURES, outcome):
  if outcome == "stroke":
    age_col = "age_at_af_diagnosis"
  else:
    age_col = "age_at_cohort_start"
    
  if sub_group == "male":
    cohort_test_sub_sample_df = cohort_test_sub[cohort_test_sub["female"] == 0]
  elif sub_group == "gte_65":
    cohort_test_sub_sample_df = cohort_test_sub[cohort_test_sub[age_col] >=65]
  elif sub_group == "lt_65":
    cohort_test_sub_sample_df = cohort_test_sub[cohort_test_sub[age_col] <65]
  else:
    cohort_test_sub_sample_df = cohort_test_sub[cohort_test_sub[sub_group] == 1]
  
  seq_features_test_nn_sub = cohort_test_sub_sample_df.apply(lookup_embeddings, args=(code_to_ix_nn, TARGET_FIELD), axis=1)
  seq_features_test_nn_tn_sub = [ torch.tensor(seq) for seq in seq_features_test_nn_sub ]
  seq_features_test_nn_tn_pd_sub = pad_sequence(seq_features_test_nn_tn_sub, batch_first=True)
  x_seq_test_sub = seq_features_test_nn_tn_pd_sub
  
  static_df_test_sub = cohort_test_sub_sample_df[STATIC_FEATURES]
  static_features_test_nn_sub = torch.tensor([ row for row in static_df_test_sub.values ])
  x_static_test_sub = static_features_test_nn_sub

  OUTCOME_CAT = outcome + "_cat"
  cohort_test_sub_sample_df[OUTCOME_CAT] = cohort_test_sub_sample_df.apply(add_label, args=(outcome,), axis=1)
  labels_test_nn_sub = cohort_test_sub_sample_df[OUTCOME_CAT].apply(lambda x: all_categories.index(x))
  labels_test_nn_tn_sub = torch.tensor(labels_test_nn_sub.values)
  y_test_sub = labels_test_nn_tn_sub
  
  
  return x_seq_test_sub, x_static_test_sub, y_test_sub

def create_dl_batches(batch_size, val_batch_size, x_seq_train, x_static_train, y_train, x_seq_val, x_static_val, y_val):
  train_data_nn = torch.utils.data.TensorDataset(
           x_seq_train, x_static_train, 
           y_train)

  val_data_nn = torch.utils.data.TensorDataset(
             x_seq_val, x_static_val, 
             y_val)

  train_loader_nn = torch.utils.data.DataLoader(
               train_data_nn, shuffle=True, 
               batch_size=batch_size, drop_last=True)

  val_loader_nn = torch.utils.data.DataLoader(
               val_data_nn, shuffle=False, 
               batch_size=val_batch_size, drop_last=True)
  
  return train_loader_nn, val_loader_nn


#training and evaluation 
def categoryFromOutput(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i]
  
def get_pred_label(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return category_i
  
def calc_dl_metrics(prediction, target, all_categories):
    predictions = []
    targets = []
    
    for i, sample in enumerate(prediction):
        pred_label = categoryFromOutput(sample, all_categories)
        pred = get_pred_label(sample)
        target_label = target[i].item()
        predictions.append(get_pred_label(sample))
        targets.append(target[i].item())
    
    try:
        tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
        accuracy = accuracy_score(targets, predictions)
        auc =  roc_auc_score(targets, predictions)
        sensitivity = recall_score(targets, predictions)
        specificity = tn / (tn + fp)
        precision = precision_score(targets, predictions, zero_division=0)
    except ValueError:
        print("Predicting all one class")
        accuracy = 0
        auc =  0
        sensitivity = 0
        specificity = 0
        precision = 0
    
    return accuracy, auc, sensitivity, specificity, precision
  
def run_dl_training_and_evaluation(net, opt, criterion, summary_data, max_seq_len, all_categories, epochs, train_loader_nn, val_batch_size, val_loader_nn, x_seq_test, x_static_test, y_test, cohort_test_sub, sub_groups, code_to_ix_nn, TARGET_FIELD, STATIC_FEATURES, outcome, summary_data_sub):
  losses_train = []
  accs_train = []
  aucs_train = []
  sens_train = []
  specs_train = []
  precs_train = []

  accs_val = []
  aucs_val = []
  sens_val = []
  specs_val = []
  precs_val = []

  accs_test = []
  aucs_test = []
  sens_test = []
  specs_test = []
  precs_test = []
  
  sub_group_res = []

  net_start = time.time()
  print(net.model_name, " started ", datetime.fromtimestamp(net_start))

  #loop through epochs
  for e in range(1,epochs+1):
      net.train()
      epoch_start = time.time()
      print("Epoch: ", str(e), " started ", datetime.fromtimestamp(epoch_start))

      #setup evaluation metrics
      epoch_loss = 0
      epoch_acc_train = 0
      epoch_auc_train = 0
      epoch_sen_train = 0
      epoch_spec_train = 0
      epoch_prec_train = 0

      epoch_acc_val = 0
      epoch_auc_val = 0
      epoch_sen_val = 0
      epoch_spec_val = 0
      epoch_prec_val = 0
      
      epoch_sub_group_res = {}
      
      if net.model_name == "LSTM":
        #initialize hidden layers
        h = net.init_hidden(batch_size)

      #training batches
      for batch_index, batch in enumerate(train_loader_nn):
          x_batch_seq = batch[0]
          x_batch_static = batch[1]
          y_batch = batch[2]
          
          if net.model_name == "LSTM":
            #generates hidden layer input for lstm
            h = tuple([l.data for l in h])
          elif net.model_name == "Transformer":
            #generates [max_seq_len, max_seq_len] square for transformer
            src_mask = generate_square_subsequent_mask(max_seq_len)
          else:
            print("Error, model type not available")

          #zero the gradient
          opt.zero_grad()

          #predict the output
          if net.model_name == "LSTM":
            y_batch_pred = net(x_batch_seq, x_batch_static, h)
          elif net.model_name == "Transformer":
            y_batch_pred = net(x_batch_seq, x_batch_static, src_mask)
          else:
            print("Error, model type not available")

          #calculate the loss
          loss = criterion(y_batch_pred, y_batch)

          #calculate evaluation metrics
          accuracy_train, auc_train, sensitivity_train, specificity_train, precision_train = calc_dl_metrics(y_batch_pred, y_batch, all_categories)

          #compute the gradient
          loss.backward()

          #update the weights
          opt.step()

          epoch_loss += loss.item()
          epoch_acc_train += accuracy_train
          epoch_auc_train += auc_train
          epoch_sen_train += sensitivity_train
          epoch_spec_train += specificity_train
          epoch_prec_train += precision_train


      #validation and test
      net.eval()
      with torch.no_grad():
          #validation
          if net.model_name == "LSTM":
            h_val = net.init_hidden(val_batch_size)
          for batch_val_index, batch_val in enumerate(val_loader_nn):
              x_batch_seq_val = batch_val[0]
              x_batch_static_val = batch_val[1]
              y_batch_val = batch_val[2]
              
              if net.model_name == "LSTM":
                #generates hidden layer input for lstm
                h_val = tuple([l_v.data for l_v in h_val])
                y_batch_pred_val = net(x_batch_seq_val, x_batch_static_val, h_val)
              elif net.model_name == "Transformer":
                #generates [max_seq_len, max_seq_len] square for transformer
                src_mask_val = generate_square_subsequent_mask(max_seq_len)
                y_batch_pred_val = net(x_batch_seq_val, x_batch_static_val, src_mask_val)
              else:
                print("Error, model type not available")
                            
              accuracy_val, auc_val, sensitivity_val, specificity_val, precision_val = calc_dl_metrics(y_batch_pred_val, y_batch_val, all_categories)

              epoch_acc_val += accuracy_val
              epoch_auc_val += auc_val
              epoch_sen_val += sensitivity_val
              epoch_spec_val += specificity_val
              epoch_prec_val += precision_val
          
          #test the output - whole group
          if net.model_name == "LSTM":
            #test batch prep lstm
            h_test = net.init_hidden(len(x_seq_test))
            h_test = tuple([l_t.data for l_t in h_test])
            y_pred_test = net(x_seq_test, x_static_test, h_test)
          elif net.model_name == "Transformer":
            #test batch prep transformer
            src_mask_test = generate_square_subsequent_mask(max_seq_len)
            y_pred_test = net(x_seq_test, x_static_test, src_mask_test)
          else:
            print("Error, model type not available")
        
          accuracy_test, auc_test, sensitivity_test, specificity_test, precision_test = calc_dl_metrics(y_pred_test, y_test, all_categories)
          
          #test the output - sub groups
          epoch_sub_group_res["model"] = net.model_name
          for sub_group in sub_groups:
            x_seq_test_sub, x_static_test_sub, y_test_sub = create_dl_sub_sample(cohort_test_sub, sub_group, code_to_ix_nn, TARGET_FIELD, STATIC_FEATURES, outcome)
            if net.model_name == "LSTM":
              #test batch prep lstm
              h_test_sub = net.init_hidden(len(x_seq_test_sub))
              h_test_sub = tuple([l_t.data for l_t in h_test_sub])
              y_pred_test_sub = net(x_seq_test_sub, x_static_test_sub, h_test_sub)
            elif net.model_name == "Transformer":
              #test batch prep transformer
              #NOTE: in sub groups, there is greater possibility that sample does not have an individual with max seq len (e.g. black british error) so mask with size of longest length
              mask_len = x_seq_test_sub.size()[1]
              src_mask_test_sub = generate_square_subsequent_mask(mask_len)
              y_pred_test_sub = net(x_seq_test_sub, x_static_test_sub, src_mask_test_sub)
            else:
              print("Error, model type not available")
              
            accuracy_test_sub, auc_test_sub, sensitivity_test_sub, specificity_test_sub, precision_test_sub = calc_dl_metrics(y_pred_test_sub, y_test_sub, all_categories) 
            epoch_sub_group_res[str("accuracy" + "_" + sub_group)] = accuracy_test_sub
            epoch_sub_group_res[str("auc" + "_" + sub_group)] = auc_test_sub
            epoch_sub_group_res[str("sensitivity" + "_" + sub_group)] = sensitivity_test_sub
            epoch_sub_group_res[str("specificity" + "_" + sub_group)] = specificity_test_sub
            epoch_sub_group_res[str("precision" + "_" + sub_group)] = precision_test_sub

      #accumulate metrics at epoch level (for charts and model reporting)

      reported_loss_train = epoch_loss / len(train_loader_nn)

      reported_acc_train = epoch_acc_train / len(train_loader_nn)
      reported_auc_train = epoch_auc_train / len(train_loader_nn)
      reported_sen_train = epoch_sen_train / len(train_loader_nn)
      reported_spec_train = epoch_spec_train / len(train_loader_nn)
      reported_prec_train = epoch_prec_train / len(train_loader_nn)

      reported_acc_val = epoch_acc_val / len(val_loader_nn)
      reported_auc_val = epoch_auc_val / len(val_loader_nn)
      reported_sen_val = epoch_sen_val / len(val_loader_nn)
      reported_spec_val = epoch_spec_val / len(val_loader_nn)
      reported_prec_val = epoch_prec_val / len(val_loader_nn)


      losses_train.append(epoch_loss)
      accs_train.append(reported_acc_train)
      aucs_train.append(reported_auc_train)
      sens_train.append(reported_sen_train)
      specs_train.append(reported_spec_train)
      precs_train.append(reported_prec_train)

      accs_val.append(reported_acc_val)
      aucs_val.append(reported_auc_val)
      sens_val.append(reported_sen_val)
      specs_val.append(reported_spec_val)
      precs_val.append(reported_prec_val)

      accs_test.append(accuracy_test)
      aucs_test.append(auc_test)
      sens_test.append(sensitivity_test)
      specs_test.append(specificity_test)
      precs_test.append(precision_test)
      
      sub_group_res.append(epoch_sub_group_res)

      epoch_end = time.time()

      print("Epoch: " + str(e) + " completed in %s seconds" % ( round(epoch_end - epoch_start,2) ) )
      #present epoch outputs
      print("Epoch: " + str(e) + " | Training Loss: " + str(round(reported_loss_train, 3)) + " | Training Accuracy: " + str(round(reported_acc_train, 3)) + " | Training AUC: " + str(round(reported_auc_train, 3))) 
      print("Epoch: " + str(e) + " | Training Sensitivity: " + str(round(reported_sen_train, 3)) + " | Training Specificity: " + str(round(reported_spec_train, 3)) + " | Training Precision: " + str(round(reported_prec_train, 3))) 

      print("Epoch: " + str(e) + "| Validation Accuracy: " + str(round(reported_acc_val, 3)) + " | Validation AUC: " + str(round(reported_auc_val, 3))) 
      print("Epoch: " + str(e) + " | Validation Sensitivity: " + str(round(reported_sen_val, 3)) + " | Validation Specificity: " + str(round(reported_spec_val, 3)) + " | Validation Precision: " + str(round(reported_prec_val, 3))) 

      print("Epoch: " + str(e) + "| Test Accuracy: " + str(round(accuracy_test, 3)) + " | Test AUC: " + str(round(auc_test, 3))) 
      print("Epoch: " + str(e) + " | Test Sensitivity: " + str(round(sensitivity_test, 3)) + " | Test Specificity: " + str(round(specificity_test, 3)) + " | Test Precision: " + str(round(precision_test, 3))) 

  net_end = time.time()
  print("Training completed in %s minutes" % ( round(net_end - net_start,2) / 60) )
  
  print("Get the summary results")
  max_auc_val = max(aucs_val)
  print("Max auc val", max_auc_val)
  max_auc_epoch_idx = aucs_val.index(max_auc_val)
  print("Max auc epoch", max_auc_epoch_idx)

  #load into summary data
  entry = {}
  entry["model"] = net.model_name
  entry["accuracy"] = accs_test[max_auc_epoch_idx]
  entry["auc"] = aucs_test[max_auc_epoch_idx]
  entry["sensitivity"] = sens_test[max_auc_epoch_idx]
  entry["specificity"] = specs_test[max_auc_epoch_idx]
  entry["precision"] = precs_test[max_auc_epoch_idx]
  summary_data.append(entry)
  
  summary_data_sub_entry = sub_group_res[max_auc_epoch_idx]
  summary_data_sub.append(summary_data_sub_entry)
  
##CHADSVASC

def list_medcodes(codelist_column_df):
  codelist = [item.code for item in codelist_column_df.select('code').collect()]
  return codelist
  
def load_chads_codelists(components):
  for comp in components:
    spark.sql(f"""CREATE OR REPLACE GLOBAL TEMP VIEW {comp}_codelist AS
    SELECT * FROM dars_nic_391419_j3w9t_collab.ccu020_20210816_2020_01_01_codelists WHERE codelist = '{comp}_chads'
    """)
    comp_table = 'global_temp.' + comp + '_codelist'
    comp_codelist = spark.table(comp_table)
    comp_codelist_py = list_medcodes(comp_codelist)
    component_codelists.append(comp_codelist_py)

def create_features_chads(x, feature_list, codelists, outcome):
  entry = {}

  #populate component fields
  for idx, codelist in enumerate(codelists):
      if idx == 0:
          comp_name = "vascular_disease"
      elif idx == 1:
          comp_name = "congestive_heart_failure"
      elif idx == 2:
          comp_name = "diabetes"
      else:
          comp_name = "hypertension"
      
      #NOTE: this field is different than ML and DL models which use most recent 100 codes as did not want to artificially disadvantage chadsvasc that does not use high dimensional sequence data
      for code in x["med_hist_uniq"]:
          if code in codelist:
              entry[comp_name] = 1
              break
          else:
              entry[comp_name] = 0


  if outcome == "stroke":
      entry["age"] = x["age_at_af_diagnosis"]
  else:
      entry["age"] = x["age_at_cohort_start"]

  entry["female"] = x["female"]
    
  entry["white"] = x["white"]
  entry["asian_or_asian_british"] = x["asian_or_asian_british"]
  entry["black_or_black_british"] = x["black_or_black_british"]
  entry["mixed"] = x["mixed"]
  entry["other_ethnic_groups"] = x["other_ethnic_groups"]

  entry[outcome] = x[outcome]

  feature_list.append(entry)
  
def create_chads_score(x):
  if x["age"] >=75:
      age = 2
  elif (x["age"] >=65) & (x["age"] <75):
      age = 1
  else:
      age = 0
  
      
  score = (x["vascular_disease"] + x["congestive_heart_failure"] + x["diabetes"] + x["hypertension"] + age + x["female"])
  
  return score


def run_chads_evaluation(cohort_test_sub_chads, outcome, summary_data, summary_data_sub, sub_groups):
  #whole group metrics
  y = cohort_test_sub_chads[outcome].values
  pred = cohort_test_sub_chads["pred_chads2"].values
  
  accuracy, auc, sensitivity, specificity, precision = calc_ml_metrics(pred, y)

  print("Accuracy: ", accuracy)
  print("Auc : ", auc)
  print("Sensitivity:", sensitivity)
  print("Specificity:", specificity)
  print("Precision:", precision)
  
  #load into summary data
  entry = {}
  entry["model"] = "CHA2DS2-VASc >=2"
  entry["accuracy"] = accuracy
  entry["auc"] = auc
  entry["sensitivity"] = sensitivity
  entry["specificity"] = specificity
  entry["precision"] = precision
  summary_data.append(entry)
  
  #sub group metrics
  entry_sub = {}
  entry_sub["model"] = "CHA2DS2-VASc >=2"
  
  #NOTE: different age interface for chads as the age parameter is already adjusted for stroke vs covid death in chads features (opportunity for tidying)
  for sub_group in sub_groups:
    if sub_group == "male":
      sub_group_test_df = cohort_test_sub_chads[cohort_test_sub_chads["female"] == 0]
    elif sub_group == "gte_65":
      sub_group_test_df = cohort_test_sub_chads[cohort_test_sub_chads["age"] >=65]
    elif sub_group == "lt_65":
      sub_group_test_df = cohort_test_sub_chads[cohort_test_sub_chads["age"] <65]
    else:
      sub_group_test_df = cohort_test_sub_chads[cohort_test_sub_chads[sub_group] == 1]
  
    y_test_sub = sub_group_test_df[outcome].values
    pred_test_sub = sub_group_test_df["pred_chads2"].values
  
    accuracy_test_sub, auc_test_sub, sensitivity_test_sub, specificity_test_sub, precision_test_sub = calc_ml_metrics(pred_test_sub, y_test_sub)
  
    entry_sub[str("accuracy" + "_" + sub_group)] = accuracy_test_sub
    entry_sub[str("auc" + "_" + sub_group)] = auc_test_sub
    entry_sub[str("sensitivity" + "_" + sub_group)] = sensitivity_test_sub
    entry_sub[str("specificity" + "_" + sub_group)] = specificity_test_sub
    entry_sub[str("precision" + "_" + sub_group)] = precision_test_sub
      
    
  summary_data_sub.append(entry_sub)
  

# COMMAND ----------

#LSTM model class
class MyLSTM(nn.Module):
  def __init__(self, output_size, vocab_size, embedding_dim, hidden_dim, n_layers, static_features_n, fc1_dim, dropout=0.2):
      super(MyLSTM, self).__init__()
      self.model_name = "LSTM"
      self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
      self.output_size = output_size
      self.n_layers = n_layers
      self.hidden_dim = hidden_dim
      self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
      self.fc_static = nn.Linear(static_features_n, hidden_dim)
      self.fc1 = nn.Linear((hidden_dim*2),fc1_dim)
      self.fc_out = nn.Linear(fc1_dim, output_size)
      self.softmax = nn.LogSoftmax(dim=1)
      self.dropout = nn.Dropout(dropout)

  def forward(self, seq_batch, static_batch, hidden):

      embeds = self.embeddings(seq_batch)
      lstm_out, (ht, ct) = self.lstm(embeds, hidden)
      lstm_ht = lstm_out[:,-1,:]
      lstm_ht_drop = self.dropout(lstm_ht)
      static = F.relu(self.fc_static(static_batch.float()))
      comb = torch.cat([lstm_ht_drop, static], dim=1)
      comb_drop = self.dropout(comb)
      fc1_out = F.relu(self.fc1(comb_drop))
      out = self.fc_out(fc1_out)
      out = self.softmax(out)
      return out

  def init_hidden(self, batch_size):
      weight = next(self.parameters()).data
      #initializes hidden state and cell state
      hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                    weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
      return hidden

# COMMAND ----------

class MyTransformer(nn.Module):

    def __init__(self, output_size, vocab_size, embedding_dim, hidden_dim, n_head, n_layers, n_static_features, static_dim, combo_dim, fc_int_dim, dropout = 0.2):
        super().__init__()
        self.model_name = "Transformer"
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_dim, n_head, hidden_dim, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        self.fc_static = nn.Linear(n_static_features, static_dim)
        self.fc_int = nn.Linear(combo_dim, fc_int_dim)
        self.embedding_dim = embedding_dim
        self.decoder = nn.Linear(fc_int_dim, output_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
      
    def forward(self, src_seq, src_static, src_mask):
        src_1 = self.encoder(src_seq) * math.sqrt(self.embedding_dim)
        src_2 = self.pos_encoder(src_1)
        seq_output = self.transformer_encoder(src_2, src_mask)
        seq_output_drop = self.dropout(seq_output)
        seq_sum_output = seq_output_drop.sum(dim=1) # pool over the time dimension
        static_output_1 = F.relu(self.fc_static(src_static.float()))
        comb_output_1 = torch.cat([static_output_1, seq_sum_output], dim=1)
        comb_output_2 = F.relu(self.fc_int(comb_output_1))
        comb_output_2_drop = self.dropout(comb_output_2)
        decoder_output = self.decoder(comb_output_2_drop)
        output = self.softmax(decoder_output)
        return output


def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim, dropout = 0.2, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        
        pe = torch.zeros(max_len, 1, embedding_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        #permute to change from seq_len first to batch size first
        pos_add = self.pe[:x.size(1)].permute(1,0,2)
        x = x + pos_add
        return self.dropout(x)

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

        # use same approach to cover training and test sub samples
        all_codes_nn = all_codes_non_n
        print("Codelist vocab length neural nets", len(all_codes_nn))
        code_to_ix_nn = {code: i+1 for i, code in enumerate(all_codes_nn)}
        
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
        
        #train, evaluate and report on ml models
        ml_models = [LogisticRegression(max_iter=3000, random_state=run), RandomForestClassifier(random_state=run),xgb.XGBRegressor(objective="binary:logistic", random_state=run)]
        x_train, x_val, y_train, y_val = train_test_split(cohort_train_sub_non_n.iloc[:, :(len(all_codes_non_n)+NUM_STATIC_FEATURES)], cohort_train_sub_non_n[outcome], test_size=0.20, random_state=run)
        evaluate_ml_models(ml_models, x_train, x_val, y_train, y_val, cohort_test_sub_non_n, all_codes_non_n, NUM_STATIC_FEATURES, outcome, summary_data, summary_data_sub, SUB_GROUPS)
        
        #NOTE: aim to free up memory here
        cohort_train_sub_non_n = None
        cohort_test_sub_non_n = None
        
        summary_data_df = pd.DataFrame(summary_data)
        print("Summary data after ML models for ", outcome, " and run ", run, " and sample ratio ", sample_ratio, " and max seq len ", max_seq_len, "\n", summary_data_df)
        
        summary_data_sub_df = pd.DataFrame(summary_data_sub)
        print("Summary data for sub groups after ML models for ", outcome, " and run ", run, " and sample ratio ", sample_ratio, " and max seq len ", max_seq_len, "\n", summary_data_sub_df)
        
        end_ml = time.time()
        print("ML completed in %s minutes" % ( round(end_ml - start_ml,2) / 60) )
        
        #setup features for dl models 
        print("Create dl features")
        start_dl = time.time()
      
        x_seq_train, x_seq_val, x_static_train, x_static_val, y_train, y_val, x_seq_test, x_static_test, y_test, all_categories, n_categories, OUTCOME_CAT = create_dl_features(cohort_train_sub, cohort_test_sub, code_to_ix_nn, TARGET_FIELD, STATIC_FEATURES, outcome, run)
        
        #train, evaluate and report on dl models 
        batch_size = 64
        val_batch_size = len(y_val)
        train_loader_nn, val_loader_nn = create_dl_batches(batch_size, val_batch_size, x_seq_train, x_static_train, y_train, x_seq_val, x_static_val, y_val)
        
        #setup dl model parameters
        
        output_size = n_categories
        vocab_size = len(all_codes_nn)+1
        embedding_dim = 200
        hidden_dim = 128
        n_layers = 2
        static_features_n = NUM_STATIC_FEATURES
        dropout = 0.2
        
        fc1_dim = 64 # lstm
        n_head = 2  # number of heads in nn.MultiheadAttention transformer
        static_dim = 64 # dimension for the transformer static data feedforward layer
        combo_dim = (static_dim + embedding_dim) #dimension for feedforward layer after concatenation in transformer
        fc_int_dim = int((combo_dim / 2)) #dimension for feedforward layer prior to decoder in transformer
        
        #setup training parameters
        epochs = 10
        learning_rate = 0.001
        iterations = int((len(y_train) / batch_size) * epochs)
        print("Number of iterations: ", iterations)
        
        lstm = MyLSTM(output_size, vocab_size, embedding_dim, hidden_dim, n_layers, static_features_n, fc1_dim, dropout)
        transformer = MyTransformer(output_size, vocab_size, embedding_dim, hidden_dim, n_head, n_layers, static_features_n, static_dim, combo_dim, fc_int_dim, dropout)
        nets = [lstm, transformer]
        #nets = [transformer]
        for net in nets:
          print(net.model_name)
          opt = optim.Adam(net.parameters(), lr=learning_rate)
          criterion = nn.NLLLoss()
          run_dl_training_and_evaluation(net, opt, criterion, summary_data, max_seq_len, all_categories, epochs, train_loader_nn, val_batch_size, val_loader_nn, x_seq_test, x_static_test, y_test, cohort_test_sub, SUB_GROUPS, code_to_ix_nn, TARGET_FIELD, STATIC_FEATURES, outcome, summary_data_sub)


        summary_data_df = pd.DataFrame(summary_data)
        print("Summary data after DL models for ", outcome, " and run ", run, " and sample ratio ", sample_ratio, " and max seq len ", max_seq_len, "\n", summary_data_df)
        
        summary_data_sub_df = pd.DataFrame(summary_data_sub)
        print("Summary data for sub groups after DL models for ", outcome, " and run ", run, " and sample ratio ", sample_ratio, " and max seq len ", max_seq_len, "\n", summary_data_sub_df)
        
        end_dl = time.time()
        print("DL completed in %s minutes" % ( round(end_dl - start_dl,2) / 60) )
        
        #setup features for chadsvasc
        print("Create chadsvasc baseline")
        chads_components = ["vascular_disease", "congestive_heart_failure", "diabetes", "hypertension"]
        component_codelists = []
        load_chads_codelists(chads_components)
        
        train_features_chads = []
        test_features_chads = []
        cohort_train_sub.apply(create_features_chads, args=(train_features_chads,component_codelists, outcome), axis=1)
        cohort_test_sub.apply(create_features_chads, args=(test_features_chads,component_codelists, outcome), axis=1)

        cohort_train_sub_chads = pd.DataFrame(train_features_chads)
        cohort_test_sub_chads = pd.DataFrame(test_features_chads)
        
        cohort_train_sub_chads["chads_score"] = cohort_train_sub_chads.apply(create_chads_score, axis=1)
        cohort_test_sub_chads["chads_score"] = cohort_test_sub_chads.apply(create_chads_score, axis=1)
        
        cohort_train_sub_chads["pred_chads2"] = np.where(cohort_train_sub_chads["chads_score"] >=2, 1, 0)
        cohort_test_sub_chads["pred_chads2"] = np.where(cohort_test_sub_chads["chads_score"] >=2, 1, 0)
        
        #evaluate and report on chadsvasc
        run_chads_evaluation(cohort_test_sub_chads, outcome, summary_data, summary_data_sub, SUB_GROUPS)
        
        #save summary tables
        summary_data_df = pd.DataFrame(summary_data)
        print("Final summary data table for ", outcome, " and run ", run, " and sample ratio ", sample_ratio, " and max seq len ", max_seq_len, "\n", summary_data_df)
        
        summary_data_sub_df = pd.DataFrame(summary_data_sub)
        print("Final summary data table for sub groups for ", outcome, " and run ", run, " and sample ratio ", sample_ratio, " and max seq len ", max_seq_len, "\n", summary_data_sub_df)
        
        summary_data_py = spark.createDataFrame(summary_data_df)
        summary_data_py_table_name = "ccu004_2_cohort_" + outcome + "_seq_len_" + str(max_seq_len) + "_sr_" + str(sample_ratio) + "_run_" + str(run) + "_summary_data_" + output_run_date
        create_table_pyspark(summary_data_py, summary_data_py_table_name)
        
        summary_data_sub_py = spark.createDataFrame(summary_data_sub_df)
        summary_data_sub_py_table_name = "ccu004_2_cohort_" + outcome + "_seq_len_" + str(max_seq_len) + "_sr_" + str(sample_ratio) + "_run_" + str(run) + "_summary_data_sub_" + output_run_date
        create_table_pyspark(summary_data_sub_py, summary_data_sub_py_table_name)
  
end = time.time()
print("Script completed in %s minutes" % ( round(end - start,2) / 60) )

# COMMAND ----------


