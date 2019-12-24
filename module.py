import os
import sys
from math import fabs
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pandas import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import requests
import json

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D, Activation, MaxPooling2D, BatchNormalization, Input
from tensorflow.keras import models
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.utils import plot_model, to_categorical


def Get_and_Plot_correlations(df,size=10, min_cor=0.0):
    '''
    Purpose: Function plots a graphical correlation matrix for each pair of columns in the dataframe.
    
    Input:
           - df             : the input pandas DataFrame
           - size           : vertical and horizontal size of the plot
           - min_cor        : minimum correlation (absolute value) desired
    Output:
           - corr           : a pandas dataframe containing the correlation matrix of the input df
           - df_corr_sorted : a pandas dataframe containing the sorted correlations for which
                              absolute value is > min_corr
    '''
        
    corr = df.corr()
    
    features_list = corr.columns.tolist()
   
    #Plot of the correlations
    plt.gcf().clear()
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, vmin=-1, vmax=1,cmap='RdYlGn')
    plt.xticks(range(len(features_list)), features_list)
    plt.yticks(range(len(features_list)), features_list)  
    #ax.set_xticklabels(features_list)
    #ax.set_yticklabels(features_list)      
    plt.xticks(rotation='vertical')
    plt.colorbar(cax)
    plt.show()
     
    #Computation of a df which lists the correlations in a sorted way
    
    corr = corr.round(decimals=2)
    df_corr_sorted = pd.DataFrame(columns=['field vs field','correlation'])
    for f in features_list:
        for g in features_list:
            if g > f and fabs(corr[f][g]) > min_cor:
                df_corr_sorted.loc[len(df_corr_sorted)]=[f + ' vs ' + g, corr[f][g]] 

    df_corr_sorted['correlation_abs'] = df_corr_sorted['correlation'].abs()
    df_corr_sorted = df_corr_sorted.sort_values(by='correlation_abs',ascending=False)
    
    return corr,df_corr_sorted 


def downsample(df, label_col_name, random_state=42):
    '''
    This function allows to balcance the different classes of a feature label_col_name by
    downsampling the largest classes towards the smallest one. During the process,
    the dataframe rows are shuffled.
    Inspired from https://rensdimmendaal.com/notes/howto-downsample-with-pandas/ 
    
    Input:
           - df             : the input pandas DataFrame
           - label_col_name : the name of the feature containing the classes to be balanced
           - random_state   : seed used for the sampn
    Output:
           - df_out         : the pandas DataFrame with balanced feature, and shuffled rows   
    '''

    # find the number of observations in the smallest group
    nmin = df[label_col_name].value_counts().min()
    df_balanced = (df
            # split the dataframe per group
            .groupby(label_col_name)
            # sample nmin observations from each group
            .apply(lambda x: x.sample(nmin))
            # recombine the dataframes 
            )
    df_out = df_balanced.sample(len(df_balanced), random_state=random_state).reset_index(drop=True)
    return df_out


def transfer_learning_model(height,width):
    '''
    This function defines the multi-task transfer learning model: 
    * The task 1 is a multi-class classification of the cc3 classes. The output for this task 
      is a dense(5) layer, to be matched with a one-hot-encoded version of the cc3 columns (containing 5 classes). 
      As it is a multi-class problem, the loss function for that task is a softmax, i.e., the probabilities of all 
      classes sum to 1.
    * The task 2 is a multi-label classification of the polka dot, floral, checker columns. The output for this task
      is a dense(3) layer, to be matched with these 3 columns (already in one-hot-encoded format). 
      As it is a multi-label problem, the loss function for that task is a sigmoid, i.e., the probabilities 
      for each label is independent from the other labels.
    
    Input:
           - height : the height dimension of the pictures
           - width  : the width dimension of the pictures
    Output:
           - model : the model defined, ready for compilation     
    '''
    
    # ====================================  
    # Definition of the convolutional base
    # ====================================
    
    # On available models: https://keras.io/applications/#available-models
    # On transfer learning: https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e
    
    from tensorflow.keras.applications import VGG16
    conv_base = VGG16(weights='imagenet',
    #from tensorflow.keras.applications import VGG19
    #conv_base = VGG19(weights='imagenet',                  
                      include_top=False,
                      input_shape=(height,width,3))
    conv_base.trainable = False

    #from tensorflow.keras.applications.xception import Xception
    #conv_base = Xception(weights='imagenet',
    #                  include_top=False,
    #                  input_shape=(height,width,3))
    #conv_base.trainable = False

    #from tensorflow.keras.applications.resnet import ResNet50
    #conv_base = ResNet50(weights='imagenet',
    #                  include_top=False,
    #                  input_shape=(height,width,3))
    #conv_base.trainable = False
    

    # =======================
    # Definition of the model
    # =======================
    
    # On multi-output models: 
    # https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
    # https://medium.com/@vijayabhaskar96/multi-label-image-classification-tutorial-with-keras-imagedatagenerator-cd541f8eaf24
    
    inp = Input(shape = (height,width,3), name='input')
    x = conv_base(inp)
    #x = GlobalMaxPooling2D()(x)
    #x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)

    #option1
    #x = Dense(512, activation='relu')(x)
    #x = Dropout(0.25)(x)
    #x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)
    output1 = Dense(5, activation = 'softmax', name='cc3')(x)
    output2 = Dense(3, activation = 'sigmoid', name='tags')(x)

    #option2
    #branch1
    #x1 = Dense(512, activation='relu')(x)
    #x1 = Dropout(0.5)(x1)
    #x1 = Dense(64, activation='relu')(x1)
    #x1 = Dropout(0.5)(x1)
    #output1 = Dense(5, activation = 'softmax', name='cc3')(x1)
    #branch2
    #x2 = Dense(512, activation='relu')(x)
    #x2 = Dropout(0.5)(x2)
    #x2 = Dense(64, activation='relu')(x2)
    #x2 = Dropout(0.5)(x2)
    #output2 = Dense(3, activation = 'sigmoid', name='tags')(x2)

    model = Model(inp,[output1,output2])        
    return model