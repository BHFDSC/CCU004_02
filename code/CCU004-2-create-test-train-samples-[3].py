# Databricks notebook source
# MAGIC %md
# MAGIC **Description** This notebook creates the test and train samples used in CCU004-2
# MAGIC  
# MAGIC **Project(s)** CCU004-2 - A nationwide deep learning pipeline to predict stroke and COVID-19 death in atrial fibrillation 
# MAGIC  
# MAGIC **Author(s)** Alex Handy
# MAGIC 
# MAGIC **Reviewer(s)** Chris Tomlinson, Hiu Yan (Samantha) Ip
# MAGIC  
# MAGIC **Date last updated** 24-01-2022

# COMMAND ----------

# MAGIC %run /Workspaces/dars_nic_391419_j3w9t_collab/CCU004/CCU004_2/CCU004-2-global-helper-functions

# COMMAND ----------

#TARGET SCENARIOS
outcomes = ["stroke", "covid_death"]
max_seq_lens = [100]
sample_ratios = [1]
runs = [1,2,3]

input_run_date = "240122"
output_run_date = "240122"

scenarios = len(outcomes) * len(max_seq_lens) * len(sample_ratios) * len(runs)

print(scenarios)

# COMMAND ----------

#create all test / train samples at once
for outcome in outcomes:
  print("Outcome: ", outcome)

  #load the cohort data
  cohort_table_name = "dars_nic_391419_j3w9t_collab.ccu004_2_cohort_" + outcome + "_seq_len_all_" + input_run_date
  cohort_py = spark.table(cohort_table_name)
  cohort_df = cohort_py.toPandas()
  print("Cohort rows", len(cohort_df))

  #create the test data and save it
  FRAC_TRAIN = 0.8
  TRAIN_SAMPLE_NUM = 10000
  TEST_SAMPLE_NUM = 10000
    
  for run in runs:
    print("Run: ", run)
    #training data
    cohort_train = cohort_df.sample(frac = FRAC_TRAIN, random_state=run)
    print("Length of train", len(cohort_train))

    #test data ("hold out" from population distribution)
    cohort_test = cohort_df.drop(cohort_train.index)
    print("Length of test", len(cohort_test))

    # Subsample test data to fit in memory
    cohort_test_sub = cohort_test.sample(n=TEST_SAMPLE_NUM, random_state=run)
    cohort_test_sub = cohort_test_sub.reset_index()
    print("Length of test sample", len(cohort_test_sub))

    cohort_test_sub_py = spark.createDataFrame(cohort_test_sub)
    cohort_test_sub_py_export_table_name = "ccu004_2_cohort_" + outcome + "_seq_len_all_run_" + str(run) + "_test_sub_" + output_run_date
    create_table_pyspark(cohort_test_sub_py, cohort_test_sub_py_export_table_name)
    print("Created test table for: ", outcome, " + ", run)
        
    for sample_ratio in sample_ratios:
      print("Sample ratio: ", sample_ratio)

      if sample_ratio == "pop":
        print(sample_ratio)
        cohort_train_sub = cohort_train.sample(n=TRAIN_SAMPLE_NUM, random_state=run)
        print("Length of training sample", len(cohort_train_sub))
      else:
        print(sample_ratio)
        pos = cohort_train[cohort_train[outcome] == 1]
        neg = cohort_train[cohort_train[outcome] == 0]

        pos_len = len(pos)
        print("Length of train pos class: ", pos_len)

        neg_sample = neg.sample(n=int(pos_len * sample_ratio), random_state=run)
        print("Length of train neg class: ", len(neg_sample))

        cohort_train_resample = pos.append(neg_sample)

        print("Length of resampled train data: ", len(cohort_train_resample))

        cohort_train_resample = cohort_train_resample.reset_index()
        cohort_train_sub = cohort_train_resample.sample(n=TRAIN_SAMPLE_NUM, random_state=run)
        cohort_train_sub = cohort_train_sub.reset_index()
        print("Length of training sample", len(cohort_train_sub))

      cohort_train_sub_py = spark.createDataFrame(cohort_train_sub)
      cohort_train_sub_py_export_table_name = "ccu004_2_cohort_" + outcome + "_seq_len_all_sr_" + str(sample_ratio) + "_run_" + str(run) + "_train_sub_" + output_run_date
      create_table_pyspark(cohort_train_sub_py, cohort_train_sub_py_export_table_name)

      print("Created train table for: ", outcome, " + ", sample_ratio, " + ", run)
