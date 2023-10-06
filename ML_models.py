import utils
from utils import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBRegressor

from scipy.stats import sem

from sklearn.metrics import mean_squared_error, r2_score

from scipy.stats import pearsonr

# Required tensorflow keras imports
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import initializers

def rfr(X, y, test_size=0.1, random_state = 0, iterations = 5):

    r2_mean_list = []
    MSE_mean_list = []
    pearson_mean_list = []

    for i in range(iterations):

        print(f'Iteration: {i+1}')

        X_train, X_val, y_train, y_val = cell_line_split(X, y, test_size, random_state) #train-validation split

        classify = RandomForestRegressor(n_jobs=-1, max_depth=300, n_estimators=200)
        classify.fit(X_train.values, y_train)

        y_pred = classify.predict(X_val)

        #metrics
        r2 = r2_score(y_val, y_pred)
        MSE = mean_squared_error(y_val.values, y_pred)
        pearson = pearsonr(y_val.values, y_pred)
        r2_mean_list.append(r2)
        MSE_mean_list.append(MSE)
        pearson_mean_list.append(pearson[0])

    standard_error = sem(r2_mean_list)
    r2_mean = mean(r2_mean_list)
    MSE_mean = mean(MSE_mean_list)
    pearson_mean = mean(pearson_mean_list)

    return r2_mean, MSE_mean, pearson_mean, standard_error

def classy_rfr(X, y, test_size=0.1, random_state = 0, iterations = 5):

    for i in range(iterations):

        print(f'Iteration: {i+1}')

        X_train, X_val, y_train, y_val = cell_line_split(X, y, test_size, random_state) #train-validation split

        classify = RandomForestClassifier(n_jobs=-1, max_depth=300, n_estimators=200)
        classify.fit(X_train.values, y_train)

        y_pred = classify.predict(X_val)

        accuracy = classify.score(X_val.values, y_val)

    return y_pred, accuracy

def rfr_standard(X, y, test_size=0.1, random_state = 0, iterations = 5):
    
    r2_mean_list = []
    MSE_mean_list = []
    pearson_mean_list = []
    
    for i in range(iterations):
        print(f'Iteration: {i+1}')
        X_train, X_val, y_train, y_val = cell_line_split(X, y, test_size, random_state) #train-validation split
        
        #standardise
        y_train, y_val = standardiser(y_train, y_val)
        
        classify = RandomForestRegressor(n_jobs=-1, max_depth=300, n_estimators=200)
        classify.fit(X_train.values, y_train)

        y_pred = classify.predict(X_val)
        
        #metrics
        r2 = r2_score(y_val, y_pred)
        MSE = mean_squared_error(y_val.values, y_pred)
        pearson = pearsonr(y_val.values, y_pred)
        r2_mean_list.append(r2)
        MSE_mean_list.append(MSE)
        pearson_mean_list.append(pearson[0])
        
    standard_error = sem(r2_mean_list)
    r2_mean = mean(r2_mean_list)
    MSE_mean = mean(MSE_mean_list)
    pearson_mean = mean(pearson_mean_list)
    
    return r2_mean, MSE_mean, pearson_mean, standard_error

def xgb(X, y, test_size=0.1, random_state = 0, iterations = 5):

    r2_mean_list = []
    MSE_mean_list = []
    pearson_mean_list = []

    for i in range(iterations):

        print(f'Iteration: {i+1}')

        X_train, X_val, y_train, y_val = cell_line_split(X, y, test_size, random_state) #train-validation split

        classify = XGBRegressor(max_depth = 75, 
                   n_estimators = 300, 
                   seed = 42, 
                   min_child_weight = 3, 
                   gamma = 0, 
                   colsample_bytree = 0.3, 
                   reg_alpha = 0.1,
                   n_jobs=-1)
        classify.fit(X_train.values, y_train)

        y_pred = classify.predict(X_val)

        #metrics
        r2 = r2_score(y_val, y_pred)
        MSE = mean_squared_error(y_val.values, y_pred)
        pearson = pearsonr(y_val.values, y_pred)
        r2_mean_list.append(r2)
        MSE_mean_list.append(MSE)
        pearson_mean_list.append(pearson[0])

    standard_error = sem(r2_mean_list)
    r2_mean = mean(r2_mean_list)
    MSE_mean = mean(MSE_mean_list)
    pearson_mean = mean(pearson_mean_list)

    return r2_mean, MSE_mean, pearson_mean, standard_error

def rfrFeatSelect(X, y, test_size=0.1, random_state = 0, iterations = 1):

    r2_mean_list = []
    MSE_mean_list = []
    pearson_mean_list = []

    for i in range(iterations):

        print(f'Iteration: {i+1}')

        X_train, X_val, y_train, y_val = cell_line_split(X, y, test_size, random_state) #train-validation split

        classify = RandomForestRegressor(n_jobs=-1)
        classify.fit(X_train.values, y_train)

        y_pred = classify.predict(X_val.values)

        featSelect = classify.feature_importances_
        r2 = r2_score(y_val, y_pred)
        MSE = mean_squared_error(y_val, y_pred)
        pearson = pearsonr(y_val.values, y_pred)
        r2_mean_list.append(r2)
        MSE_mean_list.append(MSE)
        pearson_mean_list.append(pearson[0])

    r2_mean = mean(r2_mean_list)
    MSE_mean = mean(MSE_mean_list)
    pearson_mean = mean(pearson_mean_list)

    return r2_mean, MSE_mean, pearson_mean, featSelect

def xgbFeatSelect(X, y, test_size=0.1, random_state = 0, iterations = 1):

    r2_mean_list = []
    MSE_mean_list = []
    pearson_mean_list = []

    for i in range(iterations):

        print(f'Iteration: {i+1}')

        X_train, X_val, y_train, y_val = cell_line_split(X, y, test_size, random_state) #train-validation split

        classify = XGBRegressor(max_depth = 75, 
                   n_estimators = 300, 
                   seed = 42, 
                   min_child_weight = 3, 
                   gamma = 0, 
                   colsample_bytree = 0.3, 
                   reg_alpha = 0.1,
                   n_jobs=-1)
        classify.fit(X_train.values, y_train)

        y_pred = classify.predict(X_val.values)

        featSelect = classify.feature_importances_
        r2 = r2_score(y_val, y_pred)
        MSE = mean_squared_error(y_val, y_pred)
        pearson = pearsonr(y_val.values, y_pred)
        r2_mean_list.append(r2)
        MSE_mean_list.append(MSE)
        pearson_mean_list.append(pearson[0])

    r2_mean = mean(r2_mean_list)
    MSE_mean = mean(MSE_mean_list)
    pearson_mean = mean(pearson_mean_list)
    
    return r2_mean, MSE_mean, pearson_mean, featSelect

## CNN using functional API
def build_CNN(xo_train, xd_train, learning_rate, momentum, seed, mtype = 'regression'):

    # set layer weights initialiser
    initializer = keras.initializers.GlorotUniform(seed=seed)
    
    # phosphoproteomics data input
    x_input = layers.Input(shape=(xo_train.shape[1],1))
    # 1st convolution layer
    x = layers.Conv1D(filters=8, kernel_size=4, kernel_initializer=initializer, activation='relu')(x_input) 
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Dropout(0.1)(x)
    # 2nd convolution layer
    x = layers.Conv1D(filters=16, kernel_size=4, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Flatten()(x)

    # dense layers for xo_train
    x = layers.Dense(256, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # one-hot encoded drug data input
    y_input = layers.Input(shape=(xd_train.shape[1]))
    #y = Dense(128, kernel_initializer=initializer, activation="relu")(y_input) 
    y = y_input
    
    # Concatenate phosphoproteomics and one-hot drug data + dense layers & activation layer
    z = layers.concatenate([x, y])
    z = layers.Dense(64, kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dense(64, kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    
    if mtype == 'regression':
        z = layers.Dense(1, kernel_initializer=initializer)(z) # actiation layer is 1 neuron for single regression prediction value
        
        model = keras.Model([x_input, y_input], z)
    
        model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate, 
                                                        momentum=momentum), 
                                                        loss='mse', metrics=['mae'])
    elif mtype == 'classifier':
        z = layers.Dense(3, activation = 'softmax',activity_regularizer=keras.regularizers.l2())(z) # actiation layer is 3 neuron for three classes

        model = keras.Model([x_input, y_input], z)
    
        model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    
    return model

## NN using functional API
#complexity ranges from 1-5

def build_Deep_NN(xo_train, xd_train, learning_rate, momentum, seed, complexity = 4):

    if complexity == 5:
        c = 512
    elif complexity == 4:
        c = 256
    elif complexity == 3:
        c = 128
    elif complexity == 2:
        c = 64
    elif complexity == 1:
        c = 32
    
    # set layer weights initialiser
    initializer = keras.initializers.GlorotUniform(seed=seed)
    
    # phosphoproteomics data input
    x_input = layers.Input(shape=(xo_train.shape[1]))

    # dense layers for xo_train
    x = layers.Dense(c, kernel_initializer=initializer, activation='relu')(x_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(c, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # one-hot encoded drug data input
    y_input = layers.Input(shape=(xd_train.shape[1]))
    y = Dense((c/2), kernel_initializer=initializer, activation="relu")(y_input) 
    
    # Concatenate phosphoproteomics and one-hot drug data + dense layers & activation layer
    z = layers.concatenate([x, y])
    z = layers.Dense((c/4), kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dense((c/4), kernel_initializer=initializer, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dense(1, kernel_initializer=initializer)(z) # actiation layer is 1 neuron for single regression prediction value

    model = keras.Model([x_input, y_input], z)
    
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate, 
                                                        momentum=momentum), 
                                                        loss='mse', metrics=['mae'])
    return model

# Learning rate schedule function - exponential decay after epoch 10
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
    
## CNN using functional API
def DeepIC50(X_train, learning_rate, momentum, seed, mtype='regression'):

    # set layer weights initialiser
    initializer = keras.initializers.GlorotUniform(seed=seed)
    
    # drug-cell line data input
    x_input = layers.Input(shape=(X_train.shape[1],1))
    
    # 1st convolution layer
    x = layers.Conv1D(filters=16, kernel_size=11, kernel_initializer=initializer, activation='relu')(x_input) 
    x = layers.BatchNormalization()(x)
    #x = layers.MaxPooling1D()(x)
    #x = layers.Dropout(0.1)(x)
    
    # 2nd convolution layer
    x = layers.Conv1D(filters=16, kernel_size=11, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D()(x)
    #x = layers.Dropout(0.1)(x)
    
    # 3rd convolution layer
    x = layers.Conv1D(filters=32, kernel_size=11, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    #x = layers.MaxPooling1D()(x)
    #x = layers.Dropout(0.1)(x)
    
    # 4th convolution layer
    x = layers.Conv1D(filters=32, kernel_size=11, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D()(x)
    #x = layers.Dropout(0.1)(x)
    
    # 5th convolution layer
    x = layers.Conv1D(filters=64, kernel_size=11, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D()(x)
    #x = layers.Dropout(0.1)(x)
    
    # 6th convolution layer
    x = layers.Conv1D(filters=64, kernel_size=11, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D()(x)
    #x = layers.Dropout(0.1)(x)
    
    x = layers.Flatten()(x)
    
    # 5 fully connected layers with batch norm and dropout
    x = layers.Dense(128, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(256, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(512, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(256, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    if mtype == 'classifier':
        output = layers.Dense(3, activation = 'softmax',activity_regularizer=keras.regularizers.l2())(x) # actiation layer is 3 neuron for each class

        model = keras.Model(x_input, output)

        model.compile( loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    
    if mtype == 'regression':
        output = layers.Dense(1, kernel_initializer=initializer)(x) # actiation layer is 1 neuron for single regression prediction value
        
        model = keras.Model(x_input, output)
    
        model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate, 
                                                        momentum=momentum), 
                                                        loss='mse', metrics=['mae'])
    
    return model

## CNN using functional API
def build_early_integration_DNN(X_train, learning_rate, momentum, seed, mtype='regression'):

    # set layer weights initialiser
    initializer = keras.initializers.GlorotUniform(seed=seed)
    
    # drug-cell line data input
    x_input = layers.Input(shape=(X_train.shape[1]))

    # 5 fully connected layers with batch norm and dropout
    x = layers.Dense(512, kernel_initializer=initializer, activation='relu')(x_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(1024, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(2048, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(1024, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(512, kernel_initializer=initializer, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    if mtype == 'classifier':
        output = layers.Dense(3, activation = 'softmax',activity_regularizer=keras.regularizers.l2())(x) # actiation layer is 3 neuron for each class

        model = keras.Model(x_input, output)

        model.compile( loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    
    if mtype == 'regression':
        output = layers.Dense(1, kernel_initializer=initializer)(x) # actiation layer is 1 neuron for single regression prediction value
        
        model = keras.Model(x_input, output)
    
        model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate, 
                                                        momentum=momentum), 
                                                        loss='mse', metrics=['mae'])
    
    return model