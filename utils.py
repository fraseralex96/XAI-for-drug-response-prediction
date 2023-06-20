from sklearn.model_selection import train_test_split
from statistics import mean, median, mode, quantiles
import pandas as pd

from data_makers import *

def cell_line_split(X_main, y_main, test_size, random_state):
    cl = []
    for i in X_main.index:
        cell = i.split('::')[0]
        if cell not in cl:
            cl.append(cell)
    cl_train, cl_test = train_test_split(cl, test_size=test_size, random_state=random_state)
    train_indexes = []
    test_indexes = []
    for i in X_main.index:
        cell_line = i.split('::')[0]
        if cell_line in cl_train:
            train_indexes.append(i)
        if cell_line in cl_test:
            test_indexes.append(i)
    X_train = X_main.reindex(train_indexes)
    y_train = y_main.reindex(train_indexes)
    X_test = X_main.reindex(test_indexes)
    y_test = y_main.reindex(test_indexes)
    
    return X_train, X_test, y_train, y_test

#function to find the largest elements
def Nmaxelements(list1, N):
    list2 = list1[:]
    max_list=[]
    for i in range(N):
        max_list.append(max(list2))
        list2.remove(max(list2))
    return max_list

#take features from model
def ebmFeatures(model, topX = 1000, N = 20):
    
    #create the explainer object 
    explainer = ebm.explain_global()
    
    #extract data to remove one hot features
    explainer_names = explainer.data()['names'][:topX]
    explainer_scores = explainer.data()['scores'][:topX]
       
    #select the highest scoring features
    explainer_dict_scores = {val: i for i, val in enumerate(explainer_scores)}
    explainer_dict_names = {i: val for i, val in enumerate(explainer_names)}
    
    #produce ordered lists of largest scores and names
    largest_scores = Nmaxelements(explainer_scores, N)
    largest_indexes = [explainer_dict_scores[s] for s in largest_scores]
    largest_names = [explainer_dict_names[i] for i in largest_indexes]
            
    return largest_names, largest_scores

def feat_finder(file_list, topX = 20):
    dict_list = []
    for f in file_list:
        feature_dict = {}
        with open(f, "r") as features:
            lines = features.readlines()
            for i in lines:
                phospho = i.split(':')[0]
                score = i.split(':')[1]
                score = score.split("\n")[0]
                feature_dict[phospho] = float(score)
        dict_list.append(feature_dict)
        
    feat_list = [list(fd)[:topX] for fd in dict_list]   
    
    scores = {}
    for i1, dic in enumerate(feat_list):
        if len(feat_list) > 2:
            check_list = feat_list[:i1] + feat_list[i1+1:]
            check_list = check_list[0] + check_list[1]
            for i2, item in enumerate(dic):
                if item in check_list and item not in scores:
                    scores[item]=[[i2], [dict_list[i1][item]]]
                elif item in check_list and item in scores:
                    scores[item][0].append(i2)
                    scores[item][1].append(dict_list[i1][item])  
        elif len(feat_list) == 2:
            check_list = feat_list[:i1] + feat_list[i1+1:]
            for i2, item in enumerate(dic):
                if item in check_list[0] and item not in scores:
                    scores[item]=[[i2], [dict_list[i1][item]]]
                elif item in check_list[0] and item in scores:
                    scores[item][0].append(i2)
                    scores[item][1].append(dict_list[i1][item])  
                    
    ranker = []
    for i in scores:
        scores[i].insert(1, mean(scores[i][0]))
        scores[i].insert(3, mean(scores[i][2]))

    sorted_dict = dict(sorted(scores.items(), key=lambda item: item[1][1]))

    return sorted_dict

def cancer_lines(X_main):
    cl_data = pd.read_excel("data/41467_2021_22170_MOESM3_ESM.xlsx")
    
    # remove unwanted columns, duplicates, reset index
    cl_to_cancer = cl_data[['Cancer', 'Cell line']]
    cl_to_cancer = cl_to_cancer.drop_duplicates(subset=["Cell line"], keep='first')
    cl_to_cancer = cl_to_cancer.reset_index(drop=True)
    
    # create a list of cell lines that are present in our final dataset
    unique_cls = []
    for i in X_main.index:
        cl = i.split('::')[0]
        if cl not in unique_cls:
            unique_cls.append(cl)
            
    # remove the cell lines not present in final dataset
    for i, cl in enumerate(cl_to_cancer['Cell line']):
        if cl not in unique_cls:
            cl_to_cancer = cl_to_cancer.drop(i)
            
    #sort by cancer
    cl_to_cancer = cl_to_cancer.sort_values(by='Cancer')
    cl_to_cancer = cl_to_cancer.reset_index(drop=True)
    
    #separate all the cell lines by cancer type
    AML_lines = cl_to_cancer[cl_to_cancer['Cancer']=='AML']

    hepato_lines = cl_to_cancer[cl_to_cancer['Cancer']=='Hepatocellular']

    esophag_lines = cl_to_cancer[cl_to_cancer['Cancer']=='Esophageal']
    
    return AML_lines, hepato_lines, esophag_lines

#function that turns arrays containing model output descriptive stats into dataframe
#title must be string
#data is 2d array with all arrays with descriptive stats
#headers is an array of the same length as data but with header names for the descriptive stats

def table_make(title, data, headers, file):
    title = {}
    for index, array in enumerate(data):
        title[headers[index]] = array
    df = pd.DataFrame.from_dict(title)
    df.index = df[headers[0]]
    df = df.drop([headers[0]], axis=1)
    if file:
        df.to_csv(file)
    return df

def rfrFeatures(classify, X_main, topX = 22786, N = 22786):
    
    #extract data to at least remove one hot features 
    rfr_names = list(X_main[:topX].columns)
    rfr_scores = list(classify[:topX])

    #create dictionaries to allow for easy sorting
    rfr_dict_scores = {val: i for i, val in enumerate(rfr_scores)}
    rfr_dict_names = {i: val for i, val in enumerate(rfr_names)}

    #produce ordered lists of largest scores and names
    rfr_largest_scores = Nmaxelements(rfr_scores, N)
    rfr_largest_indexes = [rfr_dict_scores[s] for s in rfr_largest_scores]
    rfr_largest_names = [rfr_dict_names[i] for i in rfr_largest_indexes]
    
    return rfr_largest_names, rfr_largest_scores

def kinase_targets(X, X_main, file_name = 'KEGG_MTOR_SIGNALING_PATHWAY', source = 'ACF'):
    
    #kinase targets from SIGNOR database
    signor_df = pd.read_csv('data/human_phosphorylations_26_05_23.txt', sep='\t')
    signor_df = signor_df[['ENTITYA', 'MECHANISM', 'ENTITYB', 'RESIDUE']]
    signor_df = signor_df[signor_df['MECHANISM']=='phosphorylation']
    
    if source == 'ACF':
        from data_makers import drugData
        dd = drugData(X_main)

        #create targets list from drug data and select only these as features for X
        target_lists = [i.split(';') for i in dd['Targets']] 
        target_list = [item for sublist in target_lists for item in sublist] #takes the nested lists formed above and turns them into one list
        targets = list(set(target_list)) #removes duplicate values from target_list
        targs = [s.strip() for s in targets] #remove unwanted whitespace
        
    elif source == 'KEGG':
        from data_makers import read_KEGG_json
        targs = read_KEGG_json(file_name)
    
    #filter for signor proteins that are cancer targets and reset index
    signor_df = signor_df[signor_df['ENTITYA'].isin(targs)]
    signor_df = signor_df.reset_index()
    
    #check if the features are targets of the kinases that are drug targets
    kinase_targets = []
    for c in X.columns:

        #extract features from the columns and split them into proteins and residues
        feat = c.split(';')[0]
        prot, residue = feat.split('(')

        #replace the one letter symbol for the residue for the three letter symbol seen in the SIGNOR database
        residue_dict = {'S':'Ser', 'T':'Thr', 'M':'Met', 'Y':'Tyr', 'R':'Arg', 'K':'Lys'}
        residue = residue.split(')')[0]
        new_symbol = residue_dict[residue[0]]
        new_residue = new_symbol+residue[1:]

        # search for features that are targets in the database
        row = signor_df[(signor_df['ENTITYB']==prot) & (signor_df['RESIDUE']==new_residue)]
        if len(row) > 0:
            kinase_targets.append(f'{prot}({residue});')
    return kinase_targets

def cancer_filter(X, y, cancer = 'AML'):
    cl_data = pd.read_excel("data/41467_2021_22170_MOESM3_ESM.xlsx")

    # remove unwanted columns, duplicates, reset index
    cl_to_cancer = cl_data[['Cancer', 'Cell line']]
    cl_to_cancer = cl_to_cancer.drop_duplicates(subset=["Cell line"], keep='first')
    cl_to_cancer = cl_to_cancer.reset_index(drop=True)

    #sort by cancer
    cl_to_cancer = cl_to_cancer.sort_values(by='Cancer')
    cl_to_cancer = cl_to_cancer.reset_index(drop=True)

    #isolate a list of only AML lines
    cancer_lines = cl_to_cancer[cl_to_cancer['Cancer']==cancer]
    cl = list(cancer_lines['Cell line'])
    
    # reindex the dataframes for these targets
    X = X.reindex(cl)
    y = y.reindex(cl)
    
    return X, y

def residue_changer(phos_list):
    residue_dict = {'Ser':'S', 'Thr':'T', 'Met':'M', 'Tyr':'Y', 'Arg':'R', 'Lys':'K'}
    new_phos_list = []
    count = 0
    for i, feat in enumerate(phos_list):
        prot, residue = feat.split('(')
        residue = residue.split(')')[0]
        
        if residue[:3] in residue_dict.keys():
            #replace the one letter symbol for the residue for the three letter symbol seen in the SIGNOR database
            new_symbol = residue_dict[residue[:3]]
            new_residue = new_symbol+residue[3:]

            new_phos = f'{prot}({new_residue});'

            new_phos_list.append(new_phos)
        
        elif isinstance(residue[0], str) and isinstance(residue[1], int):
            phos = f'{prot}({residue});'
            new_phos_list.append(phos)
        
    return new_phos_list
