#clear environment
rm(list = ls())

# set target folder
setwd("/mnt/efs/a.handy/dars_nic_391419_j3w9t_collab/CCU004_02")

#Load packages
library(odbc)
library(dplyr)
library(data.table)
library(DBI)

today_date = format(Sys.time(), "%d_%m_%Y")

# connect to databricks instance
con = dbConnect( odbc::odbc(), "Databricks", timeout = 60, 
                 PWD=rstudioapi::askForPassword("Please enter your Databricks personal access token"))

outcomes = c("stroke", "covid_death")
input_run_date = "240122"

for (outcome in outcomes){
  query = paste('SELECT * FROM dars_nic_391419_j3w9t_collab.ccu004_2_cohort_', outcome, '_seq_len_all_', input_run_date, sep="")
  data = dbGetQuery(con,query)
  
  #create function to generate clean labels for n and pct columns
  clean_table_text = function(col_n, col_pct) {
    clean_var = paste(col_n, " (", round(col_pct,3)*100, "%)", sep="")
    return(clean_var)
  }
  
  #create function to generate clean labels for mean and sd columns
  clean_cont_text = function(col_mean, col_sd) {
    clean_var = paste(round(col_mean, 1), " (+/- ", round(col_sd,1), ")", sep="")
    return(clean_var)
  }
  
  #individuals
  n_individuals = nrow(data)
  
  #age at cohort start
  mean_age = mean(data$age_at_cohort_start)
  age_sd = sd(data$age_at_cohort_start)
  age_clean = clean_cont_text(mean_age, age_sd)
  
  #age at AF diagnosis
  mean_age_af = mean(data$age_at_af_diagnosis)
  age_af_sd = sd(data$age_at_af_diagnosis)
  age_af_clean = clean_cont_text(mean_age_af, age_af_sd)
  
  #follow-up time
  data$af_yrs_since_af_diagnosis = data$af_months_since_diagnosis / 12
  mean_follow_up = mean(data$af_yrs_since_af_diagnosis)
  follow_up_sd = sd(data$af_yrs_since_af_diagnosis)
  follow_up_clean = clean_cont_text(mean_follow_up, follow_up_sd)
  
  #sex
  female_n = sum(data$female)
  female_pct = female_n / nrow(data)
  female_clean = clean_table_text(female_n, female_pct)
  
  #ethnicity categories
  
  #add ethnicity flags
  data = mutate(data, eth_white = if_else(ethnicity == "White",1,0), 
                eth_asian = if_else(ethnicity == "Asian or Asian British",1,0), 
                eth_black = if_else(ethnicity == "Black or Black British",1,0),
                eth_mixed = if_else(ethnicity == "Mixed",1,0),
                eth_other = if_else(ethnicity == "Other Ethnic Groups",1,0)
  )
  
  eth_white_n = sum(data$eth_white)
  eth_white_pct = eth_white_n / nrow(data)
  eth_white_clean = clean_table_text(eth_white_n, eth_white_pct)
  
  eth_asian_n = sum(data$eth_asian)
  eth_asian_pct = eth_asian_n / nrow(data)
  eth_asian_clean = clean_table_text(eth_asian_n, eth_asian_pct)
  
  eth_black_n = sum(data$eth_black)
  eth_black_pct = eth_black_n / nrow(data)
  eth_black_clean = clean_table_text(eth_black_n, eth_black_pct)
  
  eth_mixed_n = sum(data$eth_mixed)
  eth_mixed_pct = eth_mixed_n / nrow(data)
  eth_mixed_clean = clean_table_text(eth_mixed_n, eth_mixed_pct)
  
  eth_other_n = sum(data$eth_other)
  eth_other_pct = eth_other_n / nrow(data)
  eth_other_clean = clean_table_text(eth_other_n, eth_other_pct)
  
  #medical history length
  mean_med_hist = mean(data$med_hist_len)
  med_hist_sd = sd(data$med_hist_len)
  med_hist_clean = clean_cont_text(mean_med_hist, med_hist_sd)
  
  #medical history length unique
  mean_med_hist_uniq = mean(data$med_hist_uniq_len)
  med_hist_uniq_sd = sd(data$med_hist_uniq_len)
  med_hist_uniq_clean = clean_cont_text(mean_med_hist_uniq, med_hist_uniq_sd)
  
  #stroke
  stroke_n = sum(data$stroke)
  stroke_pct = stroke_n / nrow(data)
  stroke_clean = clean_table_text(stroke_n, stroke_pct)
  
  #covid event
  covid_event_n = sum(data$covid_infection)
  covid_event_pct = covid_event_n / nrow(data)
  covid_event_clean = clean_table_text(covid_event_n, covid_event_pct)
  
  #covid death
  covid_death_n = sum(data$covid_death)
  covid_death_pct = covid_death_n / nrow(data)
  covid_death_clean = clean_table_text(covid_death_n, covid_death_pct)

  
  summary_table = data.frame(individuals = n_individuals, 
                             age_at_cohort_start = age_clean, 
                             age_at_first_af_diagnosis = age_af_clean,
                             follow_up_time_yrs = follow_up_clean,
                             female = female_clean,
                             eth_white = eth_white_clean,
                             eth_asian = eth_asian_clean,
                             eth_black = eth_black_clean,
                             eth_mixed = eth_mixed_clean,
                             eth_other = eth_other_clean,
                             med_hist_len = med_hist_clean, 
                             med_hist_uniq_len = med_hist_uniq_clean,
                             stroke = stroke_clean,
                             covid_event = covid_event_clean,
                             covid_death = covid_death_clean
  )
  
  summary_table_t = t(summary_table)
  print(summary_table_t)
  
  summary_table_filename = paste("cohort_summary_table_",outcome,"_",today_date, ".csv", sep="")
  write.csv(summary_table_t, summary_table_filename, row.names=T, quote=F)
  
}




