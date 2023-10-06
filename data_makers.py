import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

from sklearn.preprocessing import StandardScaler

from utils import *

import sklearn
from sklearn.preprocessing import OneHotEncoder


#creates the X dataframe and the cell_line list
def x_maker(phospho_data, dtype='phospho'):
    
    if dtype == 'phospho':

        phospho_df = pd.read_excel(phospho_data, index_col=0, sheet_name=1) 

        #full stop in cell line names produced issues later on
        phospho_df.columns = [c.replace('.', '-') for c in phospho_df.columns]

        #make cell lines the rows 
        X = phospho_df.T

        #produce a list of phospho features with dates for names
        to_drop = []
        for f in X.columns:
            if f[0].isdigit() and f[1].isdigit():
                to_drop.append(f)
        X = X.drop(columns=to_drop) #remove these from the data 

        #format the table
        X = X.rename_axis('PHOSPHO SYMBOLS', axis=1)

        #create a cell line list as a secondary function output
        cell_lines = X.index.tolist()
        for i in range(len(cell_lines)):
            cell_lines[i] = cell_lines[i].replace('.','-')
            
    elif dtype == 'proteomic': #repeat for proteomic data
        phospho_df = pd.read_excel(phospho_data, index_col=0)   

        phospho_df.columns = [c.replace('.', '-') for c in phospho_df.columns]        
        
        X = phospho_df.T
        
        #remove metadata rows
        meta = ['Unnamed: 145', 'Name', 'Accessions', 'Mascot Score', 'No Unique Peptides', 'No Pept Identifications']
        
        to_drop = []
        for f in X.columns:
            if f[0].isdigit() and f[1].isdigit():
                to_drop.append(f)

        X = X.drop(columns=to_drop)
        X = X.drop(index=meta)
        
        #remove '_HUMAN' from the end of the protein names
        col_names_dict = {f: f.split('_H')[0] for f in X.columns}
        X.rename(columns=col_names_dict, inplace=True)

        #format the table
        X = X.rename_axis('PROTEIN SYMBOLS', axis=1)

        cell_lines = X.index.tolist()
        for i in range(len(cell_lines)):
            cell_lines[i] = cell_lines[i].replace('.','-')
    
    return X, cell_lines

#creates the y dataframe
def y_maker(GDSC_data):
    y = pd.read_excel(GDSC_data)

    
    frame = {} #dictionary with cell lines holding the IC50 values for each drug
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
    if noRepeats == True: #returns every drug included in the study
        for i in y_main.index:
            if i.split('::')[1] not in dl:
                dl.append(i.split('::')[1])
    else:  
        for i in y_main.index:
            dl.append(i.split('::')[1])
    return dl

#produces the one hot dataframe 

# can one hot any dataframe as long as the data you want encoded are the column headings (X.T or y)

def one_hot_maker(df, X_df = False):    
    
    if X_df:
        df = df.T

    #define the one hot encoder
    encoder = OneHotEncoder(sparse=False)

    #define the drugs needing encoding
    drugList = list(df.columns)

    #enforce 2D array format
    oneHotList = [[i] for i in drugList]        

    #create the one hot data
    onehot = encoder.fit_transform(oneHotList)

    # create a dictionary assigning drug name to one hot value
    hotDrugs = {}
    for i in range(len(onehot)):
        hotDrugs[oneHotList[i][0]] = onehot[i]

    hotdrugsDF = pd.DataFrame.from_dict(hotDrugs)

    if X_df:
        hotdrugsDF = hotdrugsDF.T
        
    return hotdrugsDF

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

def drugData(X_main, isPhospho = False):    
    
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
    dd = antiC_df # common df name

    if isPhospho:
        #kinase targets from SIGNOR database
        signor_df = pd.read_csv('data/human_phosphorylations_26_05_23.txt', sep='\t')
        signor_df = signor_df[['ENTITYA', 'MECHANISM', 'ENTITYB', 'RESIDUE']]
        signor_df = signor_df[signor_df['MECHANISM']=='phosphorylation']
        
        # create target list to filter signor
        target_lists = [i.split('; ') for i in dd['Targets']] 
        target_list = [item for sublist in target_lists for item in sublist] #takes the nested lists formed above and turns them into one list
        targets = list(set(target_list)) #removes duplicate values from target_list
        
        #filter signor using targets
        signor_df = signor_df[signor_df['ENTITYA'].isin(targets)]
        signor_df = signor_df.reset_index()    
        
        # turn the drug-target lists into phospho lists per drug so that a drug can be removed if it has no associated phospho
        phospho_target_lists = []
        for i, lis in enumerate(target_lists):
            phospho_target_lists.append([]) # create a phospho list for each drug-target list
            for val in lis: #iterate through the lists of proteins
                signor_phos = signor_df[signor_df['ENTITYA'] == val]
                for row in signor_phos.iterrows(): # for each protein we can have multiple phosphos so we must iterate again through these
                    phospho_target_lists[i].append(f'{row[1][3]}({row[1][4]});') # add the phosphos to the lists for each drug
            phospho_target_lists[i] = residue_changer(phospho_target_lists[i])   

        # reindex the dd dataframe to have phospho targets as the 'Targets' column value
        index_list = dd.index # drug list from original dataframe
        new_indexes = []
        for i1 in range(len(phospho_target_lists)):
            phospho_target_lists[i1] = [x for x in phospho_target_lists[i1] if x in list(X_main.columns)] # only keep phosphos present in X
            # if the lists still contain phosphos after filtering then join them into a string and make them the new value for dd
            if len(phospho_target_lists[i1]) > 0:
                drug = index_list[i1]
                l1 = ', '.join(phospho_target_lists[i1])
                dd.loc[drug, 'Targets'] = l1
                new_indexes.append(drug) # make a note of the drugs that contain phospho data to reindex dd

        dd = dd.reindex(new_indexes) # reindex dd
        
    return dd

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

def phospho_target_maker(X_main, X, y, drugs, source = 'ACF', ctype = 'all'):

    dd = drugData(X_main, isPhospho=True)
    
    #create targets list from drug data and select only these as features for X
    phos_target_lists = [i.split(', ') for i in dd['Targets']] 
    phos_target_list = [item for sublist in phos_target_lists for item in sublist] #takes the nested lists formed above and turns them into one list
    phos_targets = list(set(phos_target_list)) #removes duplicate values from target_list

    if ctype == 'all':
        #create the new X and Y
        dd_y = y[list(dd.index)]
        dd_X = X[phos_targets]
        dd_drugs = drugs[list(dd.index)]
    elif ctype == 'drug':
        #create the new X and Y
        dd_y = y[list(dd.index)]
        dd_drugs = drugs[list(dd.index)]
        dd_X = X
    elif ctype == 'feature':
        #create the new X and Y
        dd_y = y
        dd_X = X[phos_targets]
        dd_drugs = drugs
    
    
    return dd_X, dd_y, dd_drugs

#function that edits X and y to include only drugs and targets taken from the Anti-cancer database
#option to filter both drugs and features (ctype=all), just drugs (ctype=drug), or just features (ctype=feature)

def proteomic_target_maker(X_main, X, y, drugs, ctype = 'all'):
    #read in drug data
    dd = drugData(X_main)
    
    #create targets list from drug data and select only these as features for X
    target_lists = [i.split('; ') for i in dd['Targets']] 
    target_list = [item for sublist in target_lists for item in sublist] #takes the nested lists formed above and turns them into one list
    targets = list(set(target_list)) #removes duplicate values from target_list
    tars = set(targets).intersection(list(X.columns))
    
    if ctype == 'all':
        #remove unwanted drugs from y 
        dd_y = y[list(dd.index)]
        dd_X = X[tars]
        dd_drugs = drugs[list(dd.index)]
        
    elif ctype == 'drug':
        #remove unwanted drugs from y 
        dd_y = y[list(dd.index)]
        #create a dd_X
        dd_X = X
        dd_drugs = drugs[list(dd.index)]
        
    elif ctype == 'feature':
        #create dd_y
        dd_y = y
        dd_X = X[tars]
        dd_drugs = drugs
        
    
    return dd_X, dd_y, dd_drugs

def mixed_set_maker(X, y):
    
    #one hot representations of drugs from y
    hotdrugsDF = one_hot_maker(y)
    #one hot representations of cell lines from X
    onehotX = one_hot_maker(X, X_df=True)
    
    #produce X-main and y_main
    cl = clMaker(onehotX, y)
    x_all, x_drug, y_main = create_all_drugs(x=onehotX, xd=hotdrugsDF, y=y, cells=cl)
    X_main = X_main_maker(x_all, x_drug, short = False)
    
    return X_main, y_main