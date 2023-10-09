# XAI-for-drug-response-prediction
A repository to hold all material related to my MSc research project investigating the use of explainable AI models to predict drug response in cancer patients

The following data files are too large to upload but can be supplied upon request:

  '41467_2021_22170_MOESM4_ESM';  
  '41586_2022_5575_MOESM5_ESM';  
  'GDSC1_fitted_dose_response_24Jul22';  
  'short_41586_2022_5575_MOESM5_ESM';

Dataframes:

- X : dataframe containing -omic data for each cell line  
- y : dataframe containing GDSC IC50 values  
- x_all : dataframe containing just -omic data for cell line/drug combinations  
- x_drug : dataframe containing the drug one hot encoding for cell line/drug combinations. Combined with x_all to produce X_main in early integration models,  
  also used for late integration models where the dataframes are combined in the NN architecture  
- X_main : Concatenation of x_all and x_drug that is used in early integration models  
- y_main : contains the GDSC IC50 values for cell line/drug combinations  

N.B. suffixes such as _phos and _prot were added to the end of these to differentiate data type, or _L1000 which would differentiate the data source (LINCS consortium data)

