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
outcomes = c("stroke")
#outcomes = c("covid_death")
max_lens = c(60,100)
sample_ratios = c(1,3,"pop")
runs = c(1,2,3)
output_date = "021121"

comp_table = as.data.frame(matrix(0, ncol = 0, nrow = 6))
comp_table$model = c("CHA2DS2-VASc >=2","Logistic Regression","LSTM","Random Forest","Transformer","XG Boost")

for (outcome in outcomes){
  for (max_len in max_lens){
    for (sample_ratio in sample_ratios){
      scenario = paste("outcome_", outcome, "_seq_len_", max_len, "_sample_ratio_", sample_ratio, sep="")
      print(paste("Scenario: ", scenario, sep=""))
      
      #load in the results tables for runs
      query_1 = paste('SELECT * FROM dars_nic_391419_j3w9t_collab.ccu004_2_cohort_', outcome, '_seq_len_', max_len, '_sr_', sample_ratio, '_run_', runs[1], '_summary_data_', output_date, sep="")
      data_1 = dbGetQuery(con,query_1)
      #sort by model name
      data_1 = data_1[order(data_1$model),]
      #print(data_1)
      
      query_2 = paste('SELECT * FROM dars_nic_391419_j3w9t_collab.ccu004_2_cohort_', outcome, '_seq_len_', max_len, '_sr_', sample_ratio, '_run_', runs[2], '_summary_data_', output_date, sep="")
      data_2 = dbGetQuery(con,query_2)
      #sort by model name
      data_2 = data_2[order(data_2$model),]
      #print(data_2)
      
      query_3 = paste('SELECT * FROM dars_nic_391419_j3w9t_collab.ccu004_2_cohort_', outcome, '_seq_len_', max_len, '_sr_', sample_ratio, '_run_', runs[3], '_summary_data_', output_date, sep="")
      data_3 = dbGetQuery(con,query_3)
      #sort by model name
      data_3 = data_3[order(data_3$model),]
      #print(data_3)
      
      #create an individual table with all metrics
      run_data_comb = data.frame()
      
      for (row in 1:length(data_1)){
        model_name = data_1[row, 1]
        
        accuracies = c(data_1[row, 2], data_2[row, 2], data_3[row, 2])
        aucs = c(data_1[row, 3], data_2[row, 3], data_3[row, 3])
        sensitivities = c(data_1[row, 4], data_2[row, 4], data_3[row, 4])
        specificities = c(data_1[row, 5], data_2[row, 5], data_3[row, 5])
        precisions = c(data_1[row, 6], data_2[row, 6], data_3[row, 6])
        
        accuracy_comb = create_entry_text(accuracies) 
        auc_comb = create_entry_text(aucs)
        sensitivity_comb = create_entry_text(sensitivities) 
        specificity_comb = create_entry_text(specificities) 
        precision_comb = create_entry_text(precisions) 
        
        new_entry_ind = data.frame(
          scenario = scenario,
          model = model_name,
          accuracy = accuracy_comb,
          auc = auc_comb,
          sensitivity = sensitivity_comb,
          specificity = specificity_comb,
          precision = precision_comb
        )
      
        run_data_comb = rbind(run_data_comb, new_entry_ind)
        
      }
      
      comp_table[[scenario]] = run_data_comb$auc
      
      #save individual tables as a csv
      run_data_comb_filename = paste("summary_data_",scenario, "_", today_date, ".csv", sep="")
      write.csv(run_data_comb, run_data_comb_filename, row.names=F, quote=F) 
      
    }
  }

  print(comp_table)
  
  #save comparison table as csv
  comp_table_filename = paste("summary_data_comparison_", outcome, "_", today_date, ".csv", sep="")
  write.csv(comp_table, comp_table_filename, row.names=F, quote=F)
}






