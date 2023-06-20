import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

from utils import *

import sklearn
from sklearn.preprocessing import OneHotEncoder


#creates the X dataframe and the cell_line list
def x_maker(phospho_data, dtype='phospho'):
    
    if dtype == 'phospho':

        #read in the excel sheet with phospho data
        phospho_df = pd.read_excel(phospho_data, index_col=0, sheet_name=1)

        #remove . from cell lines
        phospho_df.columns = [c.replace('.', '-') for c in phospho_df.columns]

        #make each row a cell line
        X = phospho_df.T

        #make a list of phospho features with dates for names
        to_drop = []
        for f in X.columns:
            if f[0].isdigit() and f[1].isdigit():
                to_drop.append(f)

        #drop phospho features with dates for names
        X = X.drop(columns=to_drop)

        #format the table
        X = X.rename_axis('PHOSPHO SYMBOLS', axis=1)

        #create a cell line list
        cell_lines = X.index.tolist()
        for i in range(len(cell_lines)):
            cell_lines[i] = cell_lines[i].replace('.','-')
            
    elif dtype == 'proteomic':
        phospho_df = pd.read_excel(phospho_data, index_col=0)        
        #remove . from cell lines
        phospho_df.columns = [c.replace('.', '-') for c in phospho_df.columns]        
        
        #make each row a cell line
        X = phospho_df.T
        
        #remove metadata rows
        meta = ['Unnamed: 145', 'Name', 'Accessions', 'Mascot Score', 'No Unique Peptides', 'No Pept Identifications']
        
        #make a list of features with dates for names
        to_drop = []
        for f in X.columns:
            if f[0].isdigit() and f[1].isdigit():
                to_drop.append(f)

        #drop phospho features with dates for names
        X = X.drop(columns=to_drop)
        X = X.drop(index=meta)
        
        #remove '_HUMAN' from the end of the protein names
        col_names_dict = {f: f.split('_H')[0] for f in X.columns}
        X.rename(columns=col_names_dict, inplace=True)

        #format the table
        X = X.rename_axis('PROTEIN SYMBOLS', axis=1)

        #create a cell line list
        cell_lines = X.index.tolist()
        for i in range(len(cell_lines)):
            cell_lines[i] = cell_lines[i].replace('.','-')
    
    return X, cell_lines

#creates the y dataframe
def y_maker(GDSC_data):
    y = pd.read_excel(GDSC_data)
    frame = {}
    for d in np.unique(y['CELL_LINE_NAME']):
        cellDf = y[y['CELL_LINE_NAME'] == d]
        cellDf.index = cellDf['DRUG_NAME']
        frame[d] = cellDf['LN_IC50']

    def remove_repeats_mean_gdsc1(frame, y): 
        new_frame = {}
        for cell_line in np.unique(y['CELL_LINE_NAME']):
            temp_subset = frame[cell_line].groupby(frame[cell_line].index).mean()
            new_frame[cell_line] = temp_subset
        return new_frame  

    new_frame = remove_repeats_mean_gdsc1(frame, y)
    y = pd.DataFrame(new_frame).T
    
    return y

#function that creates drug lists
def dlMaker(y_main, noRepeats = False):
    dl = []
    if noRepeats == True:
        for i in y_main.index:
            if i.split('::')[1] not in dl:
                dl.append(i.split('::')[1])
    else:
        for i in y_main.index:
            dl.append(i.split('::')[1])
    return dl

#produces the one hot dataframe 
def one_hot_maker(y):    
    #define the one hot encoder
    encoder = OneHotEncoder(sparse=False)

    #define the drugs needing encoding
    drugList = list(y.columns)

    #enforce 2D array format
    oneHotList = [[i] for i in drugList]        

    #create the one hot data
    onehot = encoder.fit_transform(oneHotList)
    
    # create a dictionary assigning drug name to one hot value
    hotDrugs = {}
    for i in range(len(onehot)):
        hotDrugs[oneHotList[i][0]] = onehot[i]

    hotdrugsDF = pd.DataFrame.from_dict(hotDrugs)

    hotdrugsDF = hotdrugsDF.T 
    
    return hotdrugsDF.T

def X_main_maker(X, drugs, short = False): 
    
    if short == False:
        #concatenate X and one hot drugs
        X_main = pd.concat([X, drugs], axis=1)
        return X_main
    
    if short == True:
        #concatenate x_drug_short and hotDF
        X_main = pd.concat([X, drugs], axis=1)
        #shorten the x_drug df for model training
        X_main_short = X_main[0:1000] 
        return X_main_short

#produces final dataframes
def create_all_drugs(x, xd, y, cells):
    drug_inds = []
    x_dfs = []
    x_drug_dfs = []
    y_final = []
    
    #only consdier cell lines that are required. 
    y = y.loc[cells]
    x = x.loc[cells]
    x.astype(np.float16)
    for i, d in enumerate(xd.columns):
        #find cell lines without missing truth values
        y_temp = y[d]
        nona_cells = y_temp.index[~np.isnan(y_temp)]
        #finds the index for the start / end of each drug
        ind_high = len(nona_cells) + i
        drug_inds.append((d, i, ind_high))
        i += len(nona_cells)

        #store vals of the cell lines with truth values
        x_pp = x.loc[nona_cells] 
        x_dfs.append(x_pp)
        X_drug = pd.DataFrame([xd[d]] * len(x_pp))
        x_drug_dfs.append(X_drug)
        y_final.append(y_temp.dropna())

    #combine values for all drugs  
    x_final = pd.concat(x_dfs, axis=0)
    x_drug_final = pd.concat(x_drug_dfs, axis=0)
    y_final = pd.concat(y_final, axis=0)
    
    #format number type for dataframes
    #x_final = x_final.astype(np.float32)
    #x_drug_final = x_drug_final.astype(np.float16)

    #combine the drug and cell line names into an index
    cls_drugs_index = x_final.index + '::' + x_drug_final.index
    
    #re-index all arrays with this
    x_final.index = cls_drugs_index
    x_drug_final.index = cls_drugs_index
    y_final.index = cls_drugs_index
    
    return x_final, x_drug_final, y_final

#make unique cell_line list
def clMaker(X, y):
    cl = []
    for i, val in enumerate(X.index):
        if val in y.index:
            if (i == 0) or (i%3==0):
                cl.append(val)
    return cl

def landmark_X_maker(X, landmarkGenes):
    #reindex X with only the landmark genes
    L1000 = []
    for i in X.columns:
        if i.split('(')[0] in landmarkGenes:
            L1000.append(i)
    X_L1000 = X.reindex(L1000,axis="columns")  
    return X_L1000

def drugData(X_main):    
    
    #create initial dataframe from Anticancer database file
    antiC_file = "data/cancerdrugsdb.txt"
    antiC_df = pd.read_csv(antiC_file, delimiter='\t', index_col=0)

    #only related cancers and drug targets are required
    antiC_df = antiC_df[['Indications','Targets']]

    #create a list of drugs in our data
    dl = []
    for i in X_main.index:
        cl, drug = i.split('::')
        if drug not in dl:
            dl.append(drug)
    common_list = set(dl).intersection(list(antiC_df.index)) #find the overlap between our drug data and AntiCancer file
    antiC_df = antiC_df.reindex(common_list) #reindex for only overlapping drugs
    
    return antiC_df

def read_KEGG_json(file_name):
    
    import json
    #open JSON file
    file = open(f'data/KEGG/{file_name}.json')

    #create json object
    KEGG_MAPK = json.load(file)

    #access the specific gene list
    KEGG_MAPK_targets = KEGG_MAPK[file_name]['geneSymbols']
    
    return KEGG_MAPK_targets

#function that edits X and y to include only drugs and targets taken from the Anti-cancer database
#option to filter both drugs and features (ctype=all), just drugs (ctype=drug), or just features (ctype=feature)

def drug_target_maker(X_main, X, y, dtype = 'proteomic', ctype = 'all'):
    #read in drug data
    dd = drugData(X_main)
    
    if ctype == 'all':
        #remove unwanted drugs from y 
        dd_y = y[list(dd.index)]

        #create targets list from drug data and select only these as features for X
        target_lists = [i.split('; ') for i in dd['Targets']] 
        target_list = [item for sublist in target_lists for item in sublist] #takes the nested lists formed above and turns them into one list
        targets = list(set(target_list)) #removes duplicate values from target_list
        if dtype == 'proteomic':
            tars = set(targets).intersection(list(X.columns))
        elif dtype == 'phospho':
            targs = kinase_targets(X, X_main)
            tars = set(targs).intersection(list(X.columns))
        dd_X = X[tars]
        
    elif ctype == 'drug':
        #remove unwanted drugs from y 
        #does not differ between phospho and proteomic data
        dd_y = y[list(dd.index)]

        #create a dd_X
        dd_X = X
        
    elif ctype == 'feature':
        #create dd_y
        dd_y = y

        #create targets list from drug data and select only these as features for X
        target_lists = [i.split(';') for i in dd['Targets']] 
        target_list = [item for sublist in target_lists for item in sublist] #takes the nested lists formed above and turns them into one list
        targets = list(set(target_list)) #removes duplicate values from target_list
        targs = [s.strip() for s in targets] #remove unwanted whitespace
        if dtype == 'proteomic':
            tars = set(targs).intersection(list(X.columns))
        elif dtype == 'phospho':
            targs = kinase_targets(X, X_main)
            tars = set(targs).intersection(list(X.columns))
        dd_X = X[tars]
        
    
    return dd_X, dd_y
