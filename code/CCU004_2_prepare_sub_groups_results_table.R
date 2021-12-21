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

#calculate the average and sd for each metric
create_entry_text = function(metrics) {
  avg_num = round(mean(metrics),2)
  sd_num = round(sd(metrics),2)
  lower_ci = round(avg_num - (1.96*sd_num),2)
  upper_ci = round(avg_num + (1.96*sd_num),2)
  
  if (lower_ci < 0 || is.na(lower_ci)) {
    lower_ci = 0
  }
  
  if (upper_ci > 1 || is.na(upper_ci)) {
    upper_ci = 1
  }
  
  entry_text = paste(avg_num, " (", lower_ci, "-", upper_ci, ") ", sep="")
  
  return(entry_text)
}


#setup loop for scenarios
#outcomes = c("stroke")
outcomes = c("covid_death")
max_lens = c(100)
sample_ratios = c(1)
runs = c(1,2,3)
output_date = "301121"

model_names = c("CHA2DS2-VASc >=2","Logistic Regression","LSTM","Random Forest","Transformer","XG Boost")

for (outcome in outcomes){
  for (max_len in max_lens){
    for (sample_ratio in sample_ratios){
      scenario = paste("outcome_", outcome, "_seq_len_", max_len, "_sample_ratio_", sample_ratio, sep="")
      print(paste("Scenario: ", scenario, sep=""))
      
      #select the AUC data from the 3 sub groups 
      
      #load in the results tables for runs
      query_1 = paste('SELECT * FROM dars_nic_391419_j3w9t_collab.ccu004_2_cohort_', outcome, '_seq_len_', max_len, '_sr_', sample_ratio, '_run_', runs[1], '_summary_data_sub_', output_date, sep="")
      data_1 = dbGetQuery(con,query_1)
      #sort by model name
      data_1 = data_1[order(data_1$model),]
      #select auc cols only
      auc_data_1 = data_1 %>% select(c("model", "auc_female", "auc_male", "auc_lt_65", "auc_gte_65", "auc_white", "auc_asian_or_asian_british", "auc_black_or_black_british", "auc_mixed", "auc_other_ethnic_groups"))
      
      #load in the results tables for runs
      query_2 = paste('SELECT * FROM dars_nic_391419_j3w9t_collab.ccu004_2_cohort_', outcome, '_seq_len_', max_len, '_sr_', sample_ratio, '_run_', runs[2], '_summary_data_sub_', output_date, sep="")
      data_2 = dbGetQuery(con,query_2)
      #sort by model name
      data_2 = data_2[order(data_2$model),]
      #select auc cols only
      auc_data_2 = data_2 %>% select(c("model", "auc_female", "auc_male", "auc_lt_65", "auc_gte_65", "auc_white", "auc_asian_or_asian_british", "auc_black_or_black_british", "auc_mixed", "auc_other_ethnic_groups"))
      
      #load in the results tables for runs
      query_3 = paste('SELECT * FROM dars_nic_391419_j3w9t_collab.ccu004_2_cohort_', outcome, '_seq_len_', max_len, '_sr_', sample_ratio, '_run_', runs[3], '_summary_data_sub_', output_date, sep="")
      data_3 = dbGetQuery(con,query_3)
      #sort by model name
      data_3 = data_3[order(data_3$model),]
      #select auc cols only
      auc_data_3 = data_3 %>% select(c("model", "auc_female", "auc_male", "auc_lt_65", "auc_gte_65", "auc_white", "auc_asian_or_asian_british", "auc_black_or_black_british", "auc_mixed", "auc_other_ethnic_groups"))
      
      #create an individual table with all metrics
      run_data_comb = data.frame()
      
      for (row in 1:length(model_names)){
        model_name = auc_data_1[row, 1]
        
        female_aucs = c(auc_data_1[row, 2], auc_data_2[row, 2], auc_data_3[row, 2])
        male_aucs = c(auc_data_1[row, 3], auc_data_2[row, 3], auc_data_3[row, 3])
        lt_65_aucs = c(auc_data_1[row, 4], auc_data_2[row, 4], auc_data_3[row, 4])
        gte_65_aucs = c(auc_data_1[row, 5], auc_data_2[row, 5], auc_data_3[row, 5])
        white_aucs = c(auc_data_1[row, 6], auc_data_2[row, 6], auc_data_3[row, 6])
        asian_aucs = c(auc_data_1[row, 7], auc_data_2[row, 7], auc_data_3[row, 7])
        black_aucs = c(auc_data_1[row, 8], auc_data_2[row, 8], auc_data_3[row, 8])
        mixed_aucs = c(auc_data_1[row, 9], auc_data_2[row, 9], auc_data_3[row, 9])
        other_aucs = c(auc_data_1[row, 10], auc_data_2[row, 10], auc_data_3[row, 10])
        
        female_aucs_comb = create_entry_text(female_aucs) 
        male_aucs_comb = create_entry_text(male_aucs)
        lt_65_aucs_comb = create_entry_text(lt_65_aucs) 
        gte_65_aucs_comb = create_entry_text(gte_65_aucs) 
        white_aucs_comb = create_entry_text(white_aucs) 
        asian_aucs_comb = create_entry_text(asian_aucs)
        black_aucs_comb = create_entry_text(black_aucs)
        mixed_aucs_comb = create_entry_text(mixed_aucs)
        other_aucs_comb = create_entry_text(other_aucs)
        
        new_entry_ind = data.frame(
          scenario = scenario,
          model = model_name,
          female_auc = female_aucs_comb,
          male_auc = male_aucs_comb,
          lt_65_auc = lt_65_aucs_comb,
          gte_65_auc = gte_65_aucs_comb,
          white_auc = white_aucs_comb,
          asian_aucs = asian_aucs_comb,
          black_aucs = black_aucs_comb,
          mixed_aucs = mixed_aucs_comb,
          other_aucs = other_aucs_comb
        )
        
        run_data_comb = rbind(run_data_comb, new_entry_ind)
        print(run_data_comb)
        
      }
      
      #save individual tables as a csv
      run_data_comb_filename = paste("summary_data_sub_",scenario, "_", today_date, ".csv", sep="")
      write.csv(run_data_comb, run_data_comb_filename, row.names=F, quote=F)
      
    }
  }
}



