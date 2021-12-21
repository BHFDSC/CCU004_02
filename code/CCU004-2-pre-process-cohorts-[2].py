# Databricks notebook source
# MAGIC %md
# MAGIC **Description** This notebook applies the remaining cohort inclusion criteria and pre-processing for the cohort used in CCU004-2
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

#load pre-processing functions

from dateutil.relativedelta import relativedelta
from itertools import groupby

def calc_age_af(x): 
    return relativedelta(x["af_first_diagnosis_dt"], x["date_of_birth_dt"]).years

def calc_af_months_and_stroke(x):
  if pd.isna(x["stroke_first_diagnosis_dt"]):
    return None
  else:
    return ((x["stroke_first_diagnosis_dt"].year - x["af_first_diagnosis_dt"].year) * 12) + (x["stroke_first_diagnosis_dt"].month - x["af_first_diagnosis_dt"].month)
  
def calc_af_months_since_diagnosis(x): 
  end_date = pd.to_datetime("050121")
  return ((end_date.year - x["af_first_diagnosis_dt"].year) * 12) + (end_date.month - x["af_first_diagnosis_dt"].month)

def add_binary_stroke_variable(x):
  if pd.isna(x["stroke_first_diagnosis_dt"]):
    return 0
  else:
    return 1

def remove_short_term_stroke(x):
  if not (pd.isna(x["af_to_stroke_months"])) and (x["af_to_stroke_months"] < 2):
    return False
  else:
    return True

def remove_post_may21_stroke(x):
  end_date = pd.to_datetime("050121")
  if not (pd.isna(x["stroke_first_diagnosis_dt"])) and (x["stroke_first_diagnosis_dt"] > end_date):
    return False
  else:
    return True

def remove_post_may21_covid(x):
  end_date = pd.to_datetime("050121")
  if not (pd.isna(x["date_death_dt"])) and (x["date_death_dt"] > end_date):
    return False
  else:
    return True

#set gender variable
def female_flag(x):
  if x == "2":
      return 1
  else:
      return 0

#set ethnicity variables
def set_eth_label(x, eth_label):
    if x["ethnicity"] == eth_label:
        return 1
    else:
        return 0
      
      
#remove sequential duplicates
def remove_seq_dupes(x):
    return [v for i, v in enumerate(x["med_hist"]) if i == 0 or v != x["med_hist"][i-1]]


#set target codes based on a max length of sequence
def set_target_codes(x, target_field, max_len):
  codes = x[target_field]
  codes_len = len(codes)
  if codes_len > max_len:
    target_idx = (codes_len-max_len)
    target_codes = codes[target_idx:]
    return target_codes
  else:
    return x[target_field]

# COMMAND ----------

#set parameters
run_date = "301121"

# COMMAND ----------

#load cohort data

cohort_data_table = "dars_nic_391419_j3w9t_collab.ccu004_2_cohort_all_301121"
cohort_py = spark.table(cohort_data_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-process stroke cohort

# COMMAND ----------

#prepare stroke table
from pyspark.sql import functions as F

stroke_cohort_py = cohort_py.filter(cohort_py["date"] < cohort_py["af_first_diagnosis"])

# COMMAND ----------

#create table with list of medical codes for each individual and accompanying data
from pyspark.sql.functions import countDistinct, col

stroke_events = stroke_cohort_py.groupby("NHS_NUMBER_DEID").agg(F.sort_array(F.collect_list(F.struct("date", "code"))).alias("collected_list"), F.max("is_stroke_first_diagnosis"), F.max("date_of_birth"), F.max("date_of_death"), F.max("age_at_cohort_start"), F.max("sex"), F.max("ethnicity"), F.max("region_name"), F.max("imd_decile"), F.max("bmi"), F.max("af_first_diagnosis"), F.max("covid_infection"), F.max("covid_hospitalisation"), F.max("covid_death")).select("*").withColumn("sorted_list",col("collected_list.code")).withColumnRenamed("max(is_stroke_first_diagnosis)", "stroke_first_diagnosis").withColumnRenamed("max(date_of_birth)", "date_of_birth").withColumnRenamed("max(date_of_death)", "date_of_death").withColumnRenamed("max(age_at_cohort_start)", "age_at_cohort_start").withColumnRenamed("max(sex)", "sex").withColumnRenamed("max(ethnicity)", "ethnicity").withColumnRenamed("max(region_name)", "region_name").withColumnRenamed("max(imd_decile)", "imd_decile").withColumnRenamed("max(bmi)", "bmi").withColumnRenamed("max(af_first_diagnosis)", "af_first_diagnosis").withColumnRenamed("max(covid_infection)", "covid_infection").withColumnRenamed("max(covid_hospitalisation)", "covid_hospitalisation").withColumnRenamed("max(covid_death)", "covid_death").withColumnRenamed("sorted_list", "med_hist").drop("collected_list")

# COMMAND ----------

#convert cohort to pandas
stroke_cohort_df = stroke_events.toPandas()
print("Rows", len(stroke_cohort_df))
stroke_cohort_df.head()

# COMMAND ----------

#convert dates to pandas datetime
import pandas as pd
stroke_cohort_df["date_of_birth_dt"] = stroke_cohort_df["date_of_birth"].apply(lambda x: pd.to_datetime(x))
stroke_cohort_df["date_death_dt"] = stroke_cohort_df["date_of_death"].apply(lambda x: pd.to_datetime(x))
stroke_cohort_df["af_first_diagnosis_dt"] = stroke_cohort_df["af_first_diagnosis"].apply(lambda x: pd.to_datetime(x))
stroke_cohort_df["stroke_first_diagnosis_dt"] = stroke_cohort_df["stroke_first_diagnosis"].apply(lambda x: pd.to_datetime(x))

# COMMAND ----------

#create binary indicator for stroke
stroke_cohort_df["stroke"] = stroke_cohort_df.apply(add_binary_stroke_variable, axis=1)

# COMMAND ----------

#calculate months between af diagnosis and stroke
stroke_cohort_df["af_to_stroke_months"] = stroke_cohort_df.apply(calc_af_months_and_stroke, axis=1)
stroke_cohort_df["af_to_stroke_months"].describe()

# COMMAND ----------

#calculate months between af diagnosis and end of study -> median follow-up time
stroke_cohort_df["af_months_since_diagnosis"] = stroke_cohort_df.apply(calc_af_months_since_diagnosis, axis=1)
stroke_cohort_df["af_months_since_diagnosis"].describe()

# COMMAND ----------

#remove individuals with a stroke diagnosis less than 2 months after af diagnosis (to mitigate impact of delayed coding issues)
print("Length of cohort before filter: ", len(stroke_cohort_df))
stroke_cohort_df = stroke_cohort_df[stroke_cohort_df.apply(remove_short_term_stroke, axis=1)]
print("Length of cohort post filter: ", len(stroke_cohort_df))

#remove individuals with a stroke diagnosis after May 1st 2021
print("Length of cohort before filter: ", len(stroke_cohort_df))
stroke_cohort_df = stroke_cohort_df[stroke_cohort_df.apply(remove_post_may21_stroke, axis=1)]
print("Length of cohort post filter: ", len(stroke_cohort_df))

#remove individuals with a stroke diagnosis after date of death
print("Length of cohort before filter: ", len(stroke_cohort_df))
stroke_cohort_df = stroke_cohort_df[((stroke_cohort_df["stroke_first_diagnosis_dt"] < stroke_cohort_df["date_death_dt"]) | (pd.isna(stroke_cohort_df["date_death_dt"])) | (pd.isna(stroke_cohort_df["stroke_first_diagnosis_dt"])))]
print("Length of cohort post filter: ", len(stroke_cohort_df))

#review change in distribution
stroke_cohort_df["af_to_stroke_months"].describe()

# COMMAND ----------

#remove individuals with an af diagnosis prior to date of birth
print("Length of cohort before filter: ", len(stroke_cohort_df))
stroke_cohort_df = stroke_cohort_df[stroke_cohort_df["af_first_diagnosis_dt"] > stroke_cohort_df["date_of_birth_dt"]]
print("Length of cohort post filter: ", len(stroke_cohort_df))
#review change in distribution
stroke_cohort_df["af_months_since_diagnosis"].describe()

# COMMAND ----------

#set age at af diagnosis
stroke_cohort_df["age_at_af_diagnosis"] = stroke_cohort_df.apply(calc_age_af, axis=1)

# COMMAND ----------

#check stroke prevalence
stroke_cohort_df["stroke"].mean()

# COMMAND ----------

#convert NaN to 0 for covid outcomes

covid_outcomes = ["covid_infection", "covid_hospitalisation", "covid_death"]

for outcome in covid_outcomes:
    stroke_cohort_df[outcome] = stroke_cohort_df[outcome].fillna(value=0)

# COMMAND ----------

#check covid prevalence
stroke_cohort_df["covid_death"].mean()

# COMMAND ----------

#process gender and ethnicity variables
stroke_cohort_df["female"] = stroke_cohort_df["sex"].apply(female_flag)

# COMMAND ----------

#set binary variables for ethnicity
eth_labels = ["White", "Asian or Asian British", "Black or Black British", "Mixed", "Other Ethnic Groups"]
for eth_label in eth_labels:
  eth_label_clean = eth_label.replace(" ", "_").lower()
  print(eth_label_clean)
  stroke_cohort_df[eth_label_clean] = stroke_cohort_df.apply(set_eth_label, args=(eth_label, ), axis=1)

# COMMAND ----------

#add med hist sequence length
stroke_cohort_df["med_hist_len"] = stroke_cohort_df["med_hist"].apply(lambda x: len(x))
stroke_cohort_df["med_hist_len"].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

#get unique codes only
from collections import OrderedDict
stroke_cohort_df["med_hist_uniq"] = stroke_cohort_df["med_hist"].apply(lambda x: list(OrderedDict.fromkeys(x)))

# COMMAND ----------

#review distribution of unique codes
stroke_cohort_df["med_hist_uniq_len"] = stroke_cohort_df["med_hist_uniq"].apply(lambda x: len(x))
stroke_cohort_df["med_hist_uniq_len"].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

#filter individuals for sequence length
stroke_cohort_ln_df = stroke_cohort_df[stroke_cohort_df["med_hist_uniq_len"] >= 5]
stroke_cohort_ln_df["med_hist_uniq_len"].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

#set target codes field for modelling - max len 60
target_field = "med_hist_uniq"
stroke_cohort_ln_df["med_hist_target_60"] = stroke_cohort_ln_df.apply(set_target_codes, args=(target_field, 60), axis=1)

# COMMAND ----------

#review distribution of target codes
stroke_cohort_ln_df["med_hist_target_60_len"] = stroke_cohort_ln_df["med_hist_target_60"].apply(lambda x: len(x))
stroke_cohort_ln_df["med_hist_target_60_len"].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

#set target codes field for modelling - max len 100
target_field = "med_hist_uniq"
stroke_cohort_ln_df["med_hist_target_100"] = stroke_cohort_ln_df.apply(set_target_codes, args=(target_field, 100), axis=1)

# COMMAND ----------

#review distribution of target codes
stroke_cohort_ln_df["med_hist_target_100_len"] = stroke_cohort_ln_df["med_hist_target_100"].apply(lambda x: len(x))
stroke_cohort_ln_df["med_hist_target_100_len"].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

#convert back to pyspark table
stroke_cohort_ln_df_sp = spark.createDataFrame(stroke_cohort_ln_df)

# COMMAND ----------

#save stroke table
stroke_export_table_name = "ccu004_2_cohort_stroke_seq_len_all_" + run_date
create_table_pyspark(stroke_cohort_ln_df_sp, stroke_export_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-process covid cohort

# COMMAND ----------

#prepare covid table
from pyspark.sql import functions as F

covid_cohort_py = cohort_py.filter(cohort_py["date"] < cohort_py["covid_infection_date"])

# COMMAND ----------

#create table with list of medical codes for each individual and accompanying data

from pyspark.sql.functions import countDistinct, col

covid_events = covid_cohort_py.groupby("NHS_NUMBER_DEID").agg(F.sort_array(F.collect_list(F.struct("date", "code"))).alias("collected_list"), F.max("is_stroke_first_diagnosis"), F.max("date_of_birth"), F.max("date_of_death"), F.max("age_at_cohort_start"), F.max("sex"), F.max("ethnicity"), F.max("region_name"), F.max("imd_decile"), F.max("bmi"), F.max("af_first_diagnosis"), F.max("covid_infection"), F.max("covid_hospitalisation"), F.max("covid_death")).select("*").withColumn("sorted_list",col("collected_list.code")).withColumnRenamed("max(is_stroke_first_diagnosis)", "stroke_first_diagnosis").withColumnRenamed("max(date_of_birth)", "date_of_birth").withColumnRenamed("max(date_of_death)", "date_of_death").withColumnRenamed("max(age_at_cohort_start)", "age_at_cohort_start").withColumnRenamed("max(sex)", "sex").withColumnRenamed("max(ethnicity)", "ethnicity").withColumnRenamed("max(region_name)", "region_name").withColumnRenamed("max(imd_decile)", "imd_decile").withColumnRenamed("max(bmi)", "bmi").withColumnRenamed("max(af_first_diagnosis)", "af_first_diagnosis").withColumnRenamed("max(covid_infection)", "covid_infection").withColumnRenamed("max(covid_hospitalisation)", "covid_hospitalisation").withColumnRenamed("max(covid_death)", "covid_death").withColumnRenamed("sorted_list", "med_hist").drop("collected_list")

# COMMAND ----------

#convert cohort to pandas
covid_cohort_df = covid_events.toPandas()
print("Rows", len(covid_cohort_df))
covid_cohort_df.head()

# COMMAND ----------

#convert dates to pandas datetime
import pandas as pd
covid_cohort_df["date_of_birth_dt"] = covid_cohort_df["date_of_birth"].apply(lambda x: pd.to_datetime(x))
covid_cohort_df["date_death_dt"] = covid_cohort_df["date_of_death"].apply(lambda x: pd.to_datetime(x))
covid_cohort_df["af_first_diagnosis_dt"] = covid_cohort_df["af_first_diagnosis"].apply(lambda x: pd.to_datetime(x))
covid_cohort_df["stroke_first_diagnosis_dt"] = covid_cohort_df["stroke_first_diagnosis"].apply(lambda x: pd.to_datetime(x))

# COMMAND ----------

#create binary indicator for stroke
covid_cohort_df["stroke"] = covid_cohort_df.apply(add_binary_stroke_variable, axis=1)

# COMMAND ----------

#calculate months between af diagnosis and stroke
covid_cohort_df["af_to_stroke_months"] = covid_cohort_df.apply(calc_af_months_and_stroke, axis=1)
covid_cohort_df["af_to_stroke_months"].describe()

# COMMAND ----------

#calculate months between af diagnosis and end of study
covid_cohort_df["af_months_since_diagnosis"] = covid_cohort_df.apply(calc_af_months_since_diagnosis, axis=1)
covid_cohort_df["af_months_since_diagnosis"].describe()

# COMMAND ----------

#remove individuals with a stroke diagnosis after May 1st 2021
print("Length of cohort before filter: ", len(covid_cohort_df))
covid_cohort_df = covid_cohort_df[covid_cohort_df.apply(remove_post_may21_stroke, axis=1)]
print("Length of cohort post filter: ", len(covid_cohort_df))

#remove individuals with a death diagnosis after May 1st 2021
print("Length of cohort before filter: ", len(covid_cohort_df))
covid_cohort_df = covid_cohort_df[covid_cohort_df.apply(remove_post_may21_covid, axis=1)]
print("Length of cohort post filter: ", len(covid_cohort_df))

#remove individuals with a stroke diagnosis after date of death
print("Length of cohort before filter: ", len(covid_cohort_df))
covid_cohort_df = covid_cohort_df[((covid_cohort_df["stroke_first_diagnosis_dt"] < covid_cohort_df["date_death_dt"]) | (pd.isna(covid_cohort_df["date_death_dt"])) | (pd.isna(covid_cohort_df["stroke_first_diagnosis_dt"])))]
print("Length of cohort post filter: ", len(covid_cohort_df))

# COMMAND ----------

#remove individuals with an af diagnosis prior to date of birth
print("Length of cohort before filter: ", len(covid_cohort_df))
covid_cohort_df = covid_cohort_df[covid_cohort_df["af_first_diagnosis_dt"] > covid_cohort_df["date_of_birth_dt"]]
print("Length of cohort post filter: ", len(covid_cohort_df))

# COMMAND ----------

#set age at af diagnosis
covid_cohort_df["age_at_af_diagnosis"] = covid_cohort_df.apply(calc_age_af, axis=1)

# COMMAND ----------

#check stroke prevalence
covid_cohort_df["stroke"].mean()

# COMMAND ----------

#convert NaN to 0 for covid outcomes

covid_outcomes = ["covid_infection", "covid_hospitalisation", "covid_death"]

for outcome in covid_outcomes:
    covid_cohort_df[outcome] = covid_cohort_df[outcome].fillna(value=0)

# COMMAND ----------

#check covid prevalence
covid_cohort_df["covid_death"].mean()

# COMMAND ----------

#process gender variables
covid_cohort_df["female"] = covid_cohort_df["sex"].apply(female_flag)

# COMMAND ----------

#set binary variables for ethnicity
eth_labels = ["White", "Asian or Asian British", "Black or Black British", "Mixed", "Other Ethnic Groups"]
for eth_label in eth_labels:
  eth_label_clean = eth_label.replace(" ", "_").lower()
  print(eth_label_clean)
  covid_cohort_df[eth_label_clean] = covid_cohort_df.apply(set_eth_label, args=(eth_label, ), axis=1)

# COMMAND ----------

#add med hist sequence length
covid_cohort_df["med_hist_len"] = covid_cohort_df["med_hist"].apply(lambda x: len(x))
covid_cohort_df["med_hist_len"].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

#get unique codes only
from collections import OrderedDict
covid_cohort_df["med_hist_uniq"] = covid_cohort_df["med_hist"].apply(lambda x: list(OrderedDict.fromkeys(x)))

# COMMAND ----------

#review distribution of unique codes
covid_cohort_df["med_hist_uniq_len"] = covid_cohort_df["med_hist_uniq"].apply(lambda x: len(x))
covid_cohort_df["med_hist_uniq_len"].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

#filter individuals for sequence length
covid_cohort_ln_df = covid_cohort_df[covid_cohort_df["med_hist_uniq_len"] >= 5]
covid_cohort_ln_df["med_hist_uniq_len"].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

#set target codes field for modelling - 60 seq len
target_field = "med_hist_uniq"
covid_cohort_ln_df["med_hist_target_60"] = covid_cohort_ln_df.apply(set_target_codes, args=(target_field, 60), axis=1)

# COMMAND ----------

#review distribution of target codes
covid_cohort_ln_df["med_hist_target_60_len"] = covid_cohort_ln_df["med_hist_target_60"].apply(lambda x: len(x))
covid_cohort_ln_df["med_hist_target_60_len"].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

#set target codes field for modelling - 100 seq len
target_field = "med_hist_uniq"
covid_cohort_ln_df["med_hist_target_100"] = covid_cohort_ln_df.apply(set_target_codes, args=(target_field, 100), axis=1)

# COMMAND ----------

#review distribution of target codes
covid_cohort_ln_df["med_hist_target_100_len"] = covid_cohort_ln_df["med_hist_target_100"].apply(lambda x: len(x))
covid_cohort_ln_df["med_hist_target_100_len"].describe().apply(lambda x: format(x, 'f'))

# COMMAND ----------

#convert back to pyspark table
covid_cohort_ln_df_sp = spark.createDataFrame(covid_cohort_ln_df)
covid_cohort_ln_df_sp.show()

# COMMAND ----------

#save covid table
covid_export_table_name = "ccu004_2_cohort_covid_death_seq_len_all_" + run_date
create_table_pyspark(covid_cohort_ln_df_sp, covid_export_table_name)
