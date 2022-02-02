# Databricks notebook source
# MAGIC %md
# MAGIC **Description** This notebook creates the base cohort for CCU004-2
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

run_date = "301121"

# COMMAND ----------

#get existing AF evaluation cohort
af_eval = spark.table("dars_nic_391419_j3w9t_collab.ccu020_20210816_2020_01_01_study_population_cov")
af_eval.createOrReplaceGlobalTempView("af_eval")

# COMMAND ----------

#assemble ischaemic stroke codelist
spark.sql(f"""
  CREATE OR REPLACE GLOBAL TEMP VIEW is_stroke_codelist AS
  SELECT DISTINCT code as code, term as term, terminology as system, "is_stroke" as codelist
  FROM bhf_cvd_covid_uk_byod.bhf_covid_uk_phenotypes_20210127 
  WHERE name == 'stroke_IS'
  """)

# COMMAND ----------

#get date of first IS stroke diagnosis
spark.sql(f"""
  CREATE OR REPLACE GLOBAL TEMP VIEW is_stroke_ids AS
  SELECT DISTINCT NHS_NUMBER_DEID, MIN(DATE) AS is_stroke_first_diagnosis
  FROM dars_nic_391419_j3w9t_collab.gdppr_dars_nic_391419_j3w9t_archive 
  WHERE BatchId = '5ceee019-18ec-44cc-8d1d-1aac4b4ec273' 
  AND CODE IN (SELECT code FROM global_temp.is_stroke_codelist WHERE codelist = 'is_stroke' AND system = 'SNOMED')
  GROUP BY NHS_NUMBER_DEID
  """)

# COMMAND ----------

#get date of first any stroke diagnosis
spark.sql(f"""
  CREATE OR REPLACE GLOBAL TEMP VIEW any_stroke_ids AS
  SELECT DISTINCT NHS_NUMBER_DEID, MIN(DATE) AS any_stroke_first_diagnosis
  FROM dars_nic_391419_j3w9t_collab.gdppr_dars_nic_391419_j3w9t_archive 
  WHERE BatchId = '5ceee019-18ec-44cc-8d1d-1aac4b4ec273' 
  AND CODE IN (SELECT code FROM dars_nic_391419_j3w9t_collab.ccu020_20210816_2020_01_01_codelists WHERE codelist = 'stroke_hasbled' AND system = 'SNOMED')
  GROUP BY NHS_NUMBER_DEID
  """)

# COMMAND ----------

#join stroke diagnosis data
spark.sql(f"""
  CREATE OR REPLACE GLOBAL TEMPORARY VIEW af_eval_cohort AS
  SELECT af.*, 
  is_st.is_stroke_first_diagnosis as is_stroke_first_diagnosis,
  any_st.any_stroke_first_diagnosis as any_stroke_first_diagnosis
  FROM global_temp.af_eval AS af
  LEFT JOIN global_temp.is_stroke_ids AS is_st ON af.NHS_NUMBER_DEID = is_st.NHS_NUMBER_DEID
  LEFT JOIN global_temp.any_stroke_ids AS any_st ON af.NHS_NUMBER_DEID = any_st.NHS_NUMBER_DEID
  """)

# COMMAND ----------

#filter out individuals where stroke < af diagnosis
af_eval_cohort = spark.table("global_temp.af_eval_cohort")
print("Row count pre stroke diagnosis filter", af_eval_cohort.count())

af_eval_cohort_tgt = af_eval_cohort.filter( (af_eval_cohort["af_first_diagnosis"] < af_eval_cohort["any_stroke_first_diagnosis"]) | (af_eval_cohort["any_stroke_first_diagnosis"].isNull())  ) 

print("Row count post stroke diagnosis filter", af_eval_cohort_tgt.count())

af_eval_cohort_tgt_table = "af_eval_tgt_table"  
af_eval_cohort_tgt.createOrReplaceGlobalTempView(af_eval_cohort_tgt_table)

# COMMAND ----------

#join GP and HES APC
spark.sql(f"""
  CREATE OR REPLACE GLOBAL TEMPORARY VIEW gp_events AS
  SELECT NHS_NUMBER_DEID as id, CODE as code, DATE as date
  FROM dars_nic_391419_j3w9t_collab.gdppr_dars_nic_391419_j3w9t_archive
  WHERE DATE IS NOT NULL
  AND BatchId = '5ceee019-18ec-44cc-8d1d-1aac4b4ec273'
  """)

# Primary (.._01) HES APC Diagnosis 
spark.sql(f"""
  CREATE OR REPLACE GLOBAL TEMPORARY VIEW hes_events AS
  SELECT PERSON_ID_DEID as id, DIAG_4_01 as code, ADMIDATE as date
  FROM dars_nic_391419_j3w9t_collab.hes_apc_all_years_archive 
  WHERE ADMIDATE IS NOT NULL 
  AND BatchId = '5ceee019-18ec-44cc-8d1d-1aac4b4ec273'
  """)

spark.sql(f"""
  CREATE OR REPLACE GLOBAL TEMPORARY VIEW all_events AS
  SELECT *
  FROM global_temp.gp_events
  UNION ALL
  SELECT *
  FROM global_temp.hes_events
  """)

# COMMAND ----------

#join GP and HES APC data (all events) with AF evaluation cohort
spark.sql(f"""
  CREATE OR REPLACE GLOBAL TEMPORARY VIEW af_eval_all AS
  SELECT af.*, 
  all.code as code, 
  all.date as date
  FROM global_temp.af_eval_tgt_table AS af
  LEFT JOIN global_temp.all_events AS all ON af.NHS_NUMBER_DEID = all.id
  ORDER BY date
  """)

# COMMAND ----------

#save table
af_eval_all = spark.table("global_temp.af_eval_all")
export_table_name = "ccu004_2_cohort_all_" + run_date
create_table_pyspark(af_eval_all, export_table_name)
