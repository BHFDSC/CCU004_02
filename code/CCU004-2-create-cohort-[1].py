# Databricks notebook source
# MAGIC %md
# MAGIC # add documentation..
# MAGIC 
# MAGIC ## Build AF cohort using AT evaluation study cohort

# COMMAND ----------

# MAGIC %run /Workspaces/dars_nic_391419_j3w9t_collab/CCU004/CCU004_2/CCU004-2-global-helper-functions

# COMMAND ----------

#get existing AT evaluation cohort
af_eval = spark.table("dars_nic_391419_j3w9t_collab.ccu020_20210816_2020_01_01_study_population_cov")
af_eval.createOrReplaceGlobalTempView("af_eval")

# COMMAND ----------

#get date of first stroke diagnosis
spark.sql(f"""
  CREATE OR REPLACE GLOBAL TEMP VIEW stroke_ids AS
  SELECT DISTINCT NHS_NUMBER_DEID, MIN(DATE) AS stroke_first_diagnosis
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
  st.stroke_first_diagnosis as stroke_first_diagnosis
  FROM global_temp.af_eval AS af
  LEFT JOIN global_temp.stroke_ids AS st ON af.NHS_NUMBER_DEID = st.NHS_NUMBER_DEID
  """)

# COMMAND ----------

#filter out individuals where stroke < af diagnosis
af_eval_cohort = spark.table("global_temp.af_eval_cohort")

af_eval_cohort_tgt = af_eval_cohort.filter( (af_eval_cohort["af_first_diagnosis"] < af_eval_cohort["stroke_first_diagnosis"]) | (af_eval_cohort["stroke_first_diagnosis"].isNull())  ) 

af_eval_cohort_tgt_table = "af_eval_tgt_table"  
af_eval_cohort_tgt.createOrReplaceGlobalTempView(af_eval_cohort_tgt_table)

# COMMAND ----------

#combine GP and HES APC

spark.sql(f"""
  CREATE OR REPLACE GLOBAL TEMPORARY VIEW gp_events AS
  SELECT NHS_NUMBER_DEID as id, CODE as code, DATE as date
  FROM dars_nic_391419_j3w9t_collab.gdppr_dars_nic_391419_j3w9t_archive
  WHERE DATE IS NOT NULL
  AND BatchId = '5ceee019-18ec-44cc-8d1d-1aac4b4ec273'
  """)

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
export_table_name = "ccu004_2_cohort_all_021121"
create_table_pyspark(af_eval_all, export_table_name)
