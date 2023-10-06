import statistics
from statistics import mean

class meanModel():
    def __init__(self, y_train, dl):
        self.y_train = y_train
        self.dl = dl
        
        #create the list with all IC50 values
        ml = []
        for drug in self.dl:
            tempList = []
            for i in self.y_train.index:
                cl, drug_name = i.split('::')
                if drug_name == drug:
                    tempList.append(self.y_train.loc[f'{cl}::{drug_name}'])
            ml.append(tempList)

        #take the mean for these values
        meanList = []
        for i in range(len(ml)):
            meanList.append(mean(ml[i]))
        
        #create a dictionary for the model
        meanModel = {}
        for i in range(len(dl)):
            meanModel.update( {dl[i] : meanList[i]} )
            
        self.meanModel = meanModel
        
    def predict(self, y_test):
        self.y_test = y_test
        
        meanPredictions = {}
        
        for index in y_test:
            cl, drug = index.split('::') # split by double colon as this separates cl and drug in dataframe
            try:
                meanPredictions.update({index : self.meanModel[drug]}) #outputs mean of drug if drug exists
            except:
                meanPredictions.update({index : mean(list(self.meanModel.values()))}) # outputs mean of all drugs if drug doesn't exist to avoid issues when calculating the r-squared
                      
                    
        self.meanPredictions = meanPredictions
        return self.meanPredictions
        
        