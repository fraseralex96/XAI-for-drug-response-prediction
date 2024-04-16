from sklearn.model_selection import train_test_split
from statistics import mean, median, mode, quantiles
import pandas as pd

from data_makers import *

import itertools

# tts across cell lines
def cell_line_split(X, y, test_size=0.2, random_state=0):   
    cl = []
    #loop to extract cell lines from X
    for i in X.index:
        cell = i.split('::')[0] 
        if cell not in cl: # remove repeats
            cl.append(cell)
    cl_train, cl_test = train_test_split(cl, test_size=test_size, random_state=random_state) ## tts the cell lines
    
    assert len(set(cl_train).intersection(cl_test)) == 0
    
    train_indexes = []
    test_indexes = []
    
    # split indexes in X df by the previous cell lines tts
    for i in X.index:
        cell_line = i.split('::')[0]
        if cell_line in cl_train:
            train_indexes.append(i)
        if cell_line in cl_test:
            test_indexes.append(i)
            
    # perform individual splits for the NN inputs
    if y is None:
        X_train = X.reindex(train_indexes)
        X_test = X.reindex(test_indexes)
        
        return X_train, X_test        
    else:
        X_train = X.reindex(train_indexes)
        X_test = X.reindex(test_indexes)
        y_train = y.reindex(train_indexes)
        y_test = y.reindex(test_indexes)
    
        return X_train, X_test, y_train, y_test
    
#function to find the largest elements
def Nmaxelements(list1, N):
    list2 = list1[:] #produce temporary list for the function
    max_list=[]
    for i in range(N):
        max_list.append(max(list2))
        list2.remove(max(list2)) 
    return max_list

# function to order the features by 'importance'
def topFeatures(classify, X_main, topX = 22786, N = 22786):
    
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

#returns the top specified number of features
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

#separates the X dataframe by cancer type
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

#filter X for cancer type
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

# changes the residue style
def residue_changer(phos_list):
    residue_dict = {'Ser':'S', 'Thr':'T', 'Met':'M', 'Tyr':'Y', 'Arg':'R', 'Lys':'K'}
    new_phos_list = []
    
    for i, feat in enumerate(phos_list):
        
        #isolate the residue from the phosphopeptide name
        prot, residue = feat.split('(')
        residue = residue.split(')')[0]
        
        if residue[:3] in residue_dict.keys():
            #replace the one letter symbol for the residue for the three letter symbol seen in the SIGNOR database
            new_symbol = residue_dict[residue[:3]]
            new_residue = new_symbol+residue[3:]

            new_phos = f'{prot}({new_residue});'

            new_phos_list.append(new_phos)
        
        # double check if format is S939 rather than Ser939
        elif isinstance(residue[0], str) and isinstance(residue[1], int):
            phos = f'{prot}({residue});'
            new_phos_list.append(phos)
        
    return new_phos_list

# returns the phosphopeptides associated with a given list of kinases
def kinase_target_finder(kinases):
    #kinase targets from SIGNOR database
    signor_df = pd.read_csv('data/human_phosphorylations_26_05_23.txt', sep='\t')
    signor_df = signor_df[['ENTITYA', 'MECHANISM', 'ENTITYB', 'RESIDUE']]
    signor_df = signor_df[signor_df['MECHANISM']=='phosphorylation']
        
    #filter signor using targets
    signor_df = signor_df[signor_df['ENTITYA'].isin(kinases)]
    signor_df = signor_df.reset_index()

    targs_phospho = [f'{i[1][3]}({i[1][4]});' for i in signor_df.iterrows()]
    targs_phospho = residue_changer(targs_phospho)
    
    return targs_phospho

# returns a number of metrics regarding significant drug targets 
def SHAP_targets(X_main, explainer, X_test, y_test, dtype = 'phospho', strict = False):
    indexes = []
    total_feats = []
    percent_of_total = []
    total_sig = []
    percent_of_sig = []
    targets = []
    percent_sig = []
    sig_targs = []
    insig_targs = []

            
    if dtype == 'phospho':
        # dictionary of the drugs and the targets of them
        from data_makers import drugData
        dd = drugData(X_main, isPhospho = True)
        target_lists = [i.split(', ') for i in dd['Targets']]
        drug_dict = {dd.index[i]:target_lists[i] for i in range(len(target_lists))}

    elif dtype == 'proteomic':    
        # dictionary of the drugs and the targets of them
        from data_makers import drugData
        dd = drugData(X_main)
        target_lists = [i.split('; ') for i in dd['Targets']]
        drug_dict = {dd.index[i]:target_lists[i] for i in range(len(target_lists))}
        
    elif dtype == 'mix':
        from data_makers import drugData
        # Load drug datasets
        dd_phos = drugData(X_main, isPhospho=True)
        dd_prot = drugData(X_main)

        # Split target lists
        target_lists_phos = [i.split(', ') for i in dd_phos['Targets']]
        target_lists_prot = [i.split('; ') for i in dd_prot['Targets']]

        # Create a dictionary with drug indices as keys and their corresponding target lists as values
        drug_dict = {dd_prot.index[i]: target_lists_prot[i] for i in range(len(target_lists_prot))}

        # Merge target lists for drugs present in both datasets
        for i, row in dd_phos.iterrows():
            if i in drug_dict:
                targs = row['Targets'].split(', ')
                drug_dict[i].extend(targs)
    
    for i in range(len(X_test)):
        try:
            if dtype == 'proteomic':
                #calculate shap scores
                shaps = explainer.shap_values(X_test.iloc[i:i+1][:386], y_test[i], check_additivity=False)
                if strict:
                    upper_quartile = np.quantile(shaps[0], 0.90)
                else:
                    upper_quartile = np.quantile(shaps[0], 0.75)
                sig_shaps = [x for x in shaps[0] if x > upper_quartile]


            elif dtype == 'phospho':
                #calculate shap scores
                shaps = explainer.shap_values(X_test.iloc[i:i+1][:130], y_test[i], check_additivity=False)
                if strict:
                    upper_quartile = np.quantile(shaps[0], 0.90)
                else:
                    upper_quartile = np.quantile(shaps[0], 0.75)
                sig_shaps = [x for x in shaps[0] if x > upper_quartile]


            elif dtype == 'mix':
                #calculate shap scores
                shaps = explainer.shap_values(X_test.iloc[i:i+1][:516], y_test[i], check_additivity=False)
                if strict:
                    upper_quartile = np.quantile(shaps[0], 0.90)
                else:
                    upper_quartile = np.quantile(shaps[0], 0.75)
                sig_shaps = [x for x in shaps[0] if x > upper_quartile]


            #targets for the specific drug
            cl, drug = y_test.index[i].split('::')
            shap_targets = drug_dict[drug]
            print(shap_targets)

            #find where the targets are situated in the shap list
            index = [i2 for i2, v in enumerate(X_test.iloc[i:i+1]) if v in shap_targets]

            #find the shap values for the targets
            shap_vals = [shaps[0][i3] for i3 in index]

            #METRICS
            significant_targets = [shap_targets[i4] for i4, sh in enumerate(shap_vals) if sh > upper_quartile]
            insignificant_targets = set(shap_targets).difference(significant_targets)
            percent_significant = (len(significant_targets)/len(shap_targets))*100
            t_feats = len(shaps[0])
            p_of_total = (len(shap_targets)/t_feats)*100
            t_sig = len(sig_shaps)
            p_of_sig = (len(significant_targets)/t_sig)*100

            
            indexes.append(f'{cl}::{drug}')
            total_feats.append(t_feats)
            percent_of_total.append(p_of_total)
            total_sig.append(t_sig)
            percent_of_sig.append(p_of_sig)
            targets.append(shap_targets)
            percent_sig.append(percent_significant)
            sig_targs.append(significant_targets)
            insig_targs.append(insignificant_targets)
        except KeyError as e:
            print(f"No targets found for {cl}{drug} : {e}")
        
    return indexes, (total_feats, percent_of_total, total_sig, percent_of_sig), (targets, percent_sig, sig_targs, insig_targs)


# returns a number of metrics regarding significant drug targets 
def SHAP_targets_NN(X_main, explainer, X_test, y_test, dtype = 'phospho', strict=False):
    indexes = []
    total_feats = []
    percent_of_total = []
    total_sig = []
    percent_of_sig = []
    targets = []
    percent_sig = []
    sig_targs = []
    insig_targs = []

    xo_test, xd_test = X_test
            
    if dtype == 'phospho':
        # dictionary of the drugs and the targets of them
        from data_makers import drugData
        dd = drugData(X_main, isPhospho = True)
        target_lists = [i.split(', ') for i in dd['Targets']]
        drug_dict = {dd.index[i]:target_lists[i] for i in range(len(target_lists))}

    elif dtype == 'proteomic':    
        # dictionary of the drugs and the targets of them
        from data_makers import drugData
        dd = drugData(X_main)
        target_lists = [i.split('; ') for i in dd['Targets']]
        drug_dict = {dd.index[i]:target_lists[i] for i in range(len(target_lists))}
        
    elif dtype == 'mix':
        from data_makers import drugData
        # Load drug datasets
        dd_phos = drugData(X_main, isPhospho=True)
        dd_prot = drugData(X_main)

        # Split target lists
        target_lists_phos = [i.split(', ') for i in dd_phos['Targets']]
        target_lists_prot = [i.split('; ') for i in dd_prot['Targets']]

        # Create a dictionary with drug indices as keys and their corresponding target lists as values
        drug_dict = {dd_prot.index[i]: target_lists_prot[i] for i in range(len(target_lists_prot))}

        # Merge target lists for drugs present in both datasets
        for i, row in dd_phos.iterrows():
            if i in drug_dict:
                targs = row['Targets'].split(', ')
                drug_dict[i].extend(targs)

    for i in range(len(xo_test)):
        try:
            shaps = explainer.shap_values([np.array([xo_test.iloc[i]]), np.array([xd_test.iloc[i]])])
            
            if strict:
                upper_quartile = np.quantile(shaps[0][0][0], 0.90)
            else:
                upper_quartile = np.quantile(shaps[0][0][0], 0.75)   
                
            sig_shaps = [x for x in shaps[0][0][0] if x > upper_quartile]

            #targets for the specific drug
            cl, drug = y_test.index[i].split('::')
            drug_targets = drug_dict[drug]
            

            #find where the targets are situated in the shap list
            index = [i2 for i2, v in enumerate(xo_test.iloc[i:i+1]) if v in drug_targets]

            #find the shap values for the targets
            shap_vals = [shaps[0][0][0][i3] for i3 in index]

            #METRICS
            significant_targets = [drug_targets[i4] for i4, sh in enumerate(shap_vals) if sh > upper_quartile and sh > 0]
            insignificant_targets = set(drug_targets).difference(significant_targets)
            percent_significant = (len(significant_targets)/len(drug_targets))*100
            t_feats = len(shaps[0][0][0])
            p_of_total = (len(drug_targets)/t_feats)*100
            t_sig = len(sig_shaps)
            p_of_sig = (len(significant_targets)/t_sig)*100

            
            indexes.append(f'{cl}::{drug}')
            total_feats.append(t_feats)
            percent_of_total.append(p_of_total)
            total_sig.append(t_sig)
            percent_of_sig.append(p_of_sig)
            targets.append(drug_targets)
            percent_sig.append(percent_significant)
            sig_targs.append(significant_targets)
            insig_targs.append(insignificant_targets)
        except KeyError as e:
            print(f"No targets found for {cl}{drug} : {e}")
        
    return indexes, (total_feats, percent_of_total, total_sig, percent_of_sig), (targets, percent_sig, sig_targs, insig_targs)


# returns a number of metrics regarding significant drug targets 
def IG_targets_NN(X_main, explainer, X_test, y_test, dtype = 'phospho', strict = False):
    indexes = []
    total_feats = []
    percent_of_total = []
    total_sig = []
    percent_of_sig = []
    targets = []
    percent_sig = []
    sig_targs = []
    insig_targs = []

    xo_test, xd_test = X_test
            
    if dtype == 'phospho':
        # dictionary of the drugs and the targets of them
        from data_makers import drugData
        dd = drugData(X_main, isPhospho = True)
        target_lists = [i.split(', ') for i in dd['Targets']]
        drug_dict = {dd.index[i]:target_lists[i] for i in range(len(target_lists))}

    elif dtype == 'proteomic':    
        # dictionary of the drugs and the targets of them
        from data_makers import drugData
        dd = drugData(X_main)
        target_lists = [i.split('; ') for i in dd['Targets']]
        drug_dict = {dd.index[i]:target_lists[i] for i in range(len(target_lists))}
        
    elif dtype == 'mix':
        from data_makers import drugData
        # Load drug datasets
        dd_phos = drugData(X_main, isPhospho=True)
        dd_prot = drugData(X_main)

        # Split target lists
        target_lists_phos = [i.split(', ') for i in dd_phos['Targets']]
        target_lists_prot = [i.split('; ') for i in dd_prot['Targets']]

        # Create a dictionary with drug indices as keys and their corresponding target lists as values
        drug_dict = {dd_prot.index[i]: target_lists_prot[i] for i in range(len(target_lists_prot))}

        # Merge target lists for drugs present in both datasets
        for i, row in dd_phos.iterrows():
            if i in drug_dict:
                targs = row['Targets'].split(', ')
                drug_dict[i].extend(targs)
                
    for i in range(len(xo_test)):
        try:
            #explainer instance
            explanation = explainer.explain([xo_test[i:(i+1)].values, np.array(xd_test[i:(i+1)])],
                                     baselines=None,
                                     target=None)
            attributions = explanation.attributions #top features for this explanation
            
            if strict:
                upper_quartile = np.quantile(attributions[0][0], 0.90)
            else:
                upper_quartile = np.quantile(attributions[0][0], 0.75)
                
            sig_shaps = [x for x in attributions[0][0] if x > upper_quartile]
                    
            #targets for the specific drug
            cl, drug = y_test.index[i].split('::')
            drug_targets = drug_dict[drug]

            #find where the targets are situated in the shap list
            index = [i2 for i2, v in enumerate(xo_test.iloc[i:i+1]) if v in drug_targets]

            #find the shap values for the targets
            ig_vals = [attributions[0][0][i3] for i3 in index]

            #METRICS
            significant_targets = [drug_targets[i4] for i4, sh in enumerate(ig_vals) if sh > upper_quartile and sh > 0]
            insignificant_targets = set(drug_targets).difference(significant_targets)
            percent_significant = (len(significant_targets)/len(drug_targets))*100
            t_feats = len(attributions[0][0])
            p_of_total = (len(drug_targets)/t_feats)*100
            t_sig = len(sig_shaps)
            p_of_sig = (len(significant_targets)/t_sig)*100

            
            indexes.append(f'{cl}::{drug}')
            total_feats.append(t_feats)
            percent_of_total.append(p_of_total)
            total_sig.append(t_sig)
            percent_of_sig.append(p_of_sig)
            targets.append(drug_targets)
            percent_sig.append(percent_significant)
            sig_targs.append(significant_targets)
            insig_targs.append(insignificant_targets)
        except KeyError as e:
            print(f"No targets found for {cl}{drug} : {e}")
        
    return indexes, (total_feats, percent_of_total, total_sig, percent_of_sig), (targets, percent_sig, sig_targs, insig_targs)

# returns index of a target
def index_finder(targ, df):
    index = 0
    for i, v in df.iterrows():
        if i == targ:
            return index
        index+=1
    print('No index found')
    
#standardise by removing the mean from each point and dividing by sd
def standardiser(train, test):
    #set the mean
    mean = train.mean(axis=0)
    std = train.std(axis=0)

    # standardise both the train and test on the training set mean
    train -= mean
    train /= std

    test -= mean
    test /= std

    return train, test

# turns the IC50 values into classes
def classyFire(y):
    for i in range(len(y)):
        if y[i] < 2.36: 
            y[i] = 0
        elif y[i] > 2.36 and y[i] < 5.26: 
            y[i] = 1
        elif y[i] > 5.26: 
            y[i] = 2

    print(f'0 : high responsiveness\n1 : intermediate responsiveness\n2 : low responsiveness')
    return y

# removes the one hot from the outputs for explainability techniques
def drug_feat_remover(exp_list):
    # remove drug features
    feat_list = []
    for i, LIME in enumerate(exp_list):
        matches = re.findall(r'[A-Z]', LIME[0])
        if matches and LIME[1]>0:
            feat_list.append(f'{LIME[0]}: {LIME[1]}')

    return feat_list

# calculate electronic markers for drug response
def EMDR_finder(drug, dtype='phospho', rawData = True):
    #read in data types
    if dtype == 'phospho' and rawData == False:
        EMDR_df = pd.read_csv('EMDR/EMDR_phos/EMDR_vals.csv')
    elif dtype == 'phospho' and rawData:   
        EMDR_df = pd.read_csv('EMDR/raw_EMDR_phos/normal_EMDR_vals.csv', index_col=0)
    elif dtype == 'proteomic' and rawData == False:
        print('Only raw data available')
    elif dtype == 'proteomic' and rawData:   
        EMDR_df = pd.read_csv('EMDR/raw_EMDR_prot/normal_EMDR_vals.csv', index_col=0)
        
    EMDR_feature_split = EMDR_df['EMDRs'][drug].split("'") # split the string that is outputted into an array
    
    even_feats = itertools.islice(EMDR_feature_split, 1, None, 2) # remove odd indexes in arrays as these contain "'" and no features due to split
    EMDR_features = list(itertools.chain(even_feats)) # chain this together into an array again
    
    return EMDR_features


def drug_or_cl_sig(target, df, isDrug=True):
    indexes = []
    sig_list = []
    if isDrug:
        for row in df.iterrows():
            drug = row[1]['cl:drug'].split('::')[1]
            if drug == target:
                sig_list.append(row[1]['percent significant'])
                indexes.append(row[0])
    elif not isDrug:
        for row in df.iterrows():
            cell_line = row[1]['cl:drug'].split('::')[0]
            if cell_line == target:
                sig_list.append(row[1]['percent significant'])
                indexes.append(row[0])
    return indexes, sig_list
