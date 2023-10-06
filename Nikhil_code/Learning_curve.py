import keras_tuner as kt 
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import time as time
import numpy as np
import sklearn
import scipy
import pandas as pd

def run_lc_ucl(model_func, xtrain, ytrain, xval, yval, xtest, ytest, 
               train_sizes, num_trails=1, epochs=100, 
               direc='my_dir', proj_name='test'):

    test_results = {'r2': [], 'mse': [], 'rho' : []}
    t1 = time.time()
    best_models, best_hps = [], []
    num_train_pairs = [] 
    for i, train_size in enumerate(train_sizes):
        print((i / len(train_sizes)))
        
        
        
        #create tuner object 
        tuner = kt.RandomSearch(
            hypermodel=model_func,
            objective="val_loss",
            max_trials=num_trails,
            executions_per_trial=1,
            overwrite=True,
            directory=direc,
            project_name=proj_name,
        )
    
        #run search 
        callbacks = [EarlyStopping(monitor='val_loss', patience=5, 
                                   restore_best_weights=True)]
        

        
        tuner.search(
            [xtrain[0].iloc[: train_size], xtrain[1].iloc[: train_size]], 
            ytrain.iloc[: train_size],
            validation_data=([xval[0], xval[1]], yval),
            epochs=epochs,
            callbacks=callbacks,
            batch_size=64
            )
        
        #make predctions using the opt model
        pre = tuner.get_best_models()[0].predict(xtest)
        pre = pre.reshape(len(pre))
        test_results['rho'].append(scipy.stats.pearsonr(ytest, pre))
        test_results['r2'].append(sklearn.metrics.r2_score(ytest, pre))
        test_results['mse'].append(sklearn.metrics.mean_squared_error(ytest, pre))
        
        best_models.append(tuner.get_best_models()[0])
        best_hps.append(tuner.get_best_hyperparameters()[0])
        
    delta_t = t1 - time.time()
    test_results = pd.DataFrame(test_results)
    test_results['train size'] = train_sizes
    print('total time elapsed (s)')
    print(delta_t)
                                   
    
    return test_results, best_models, best_hps