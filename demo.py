# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 08:34:01 2022
Main script for experiments with multi-label classification for Unit Commitment
@author: jpeeples, Dillon Richards
"""

## Import libraries
from __future__ import print_function
from __future__ import division
import argparse
import numpy as np
import random
import pandas as pd
import pickle
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (multilabel_confusion_matrix, hamming_loss, 
                             zero_one_loss, classification_report)

## Local external libraries
from Demo_Parameters import Parameters
from Utils.crossval import cross_validation
from Utils.PR_Curves import generate_PR_Curves


def load_data(directory):
    
    #Load features and labels
    df_labels = pd.read_csv('{}{}.csv'.format(directory,'Labels'))
    df_features = pd.read_csv('{}{}.csv'.format(directory,'Features'))
    
    #Set index values for data frames based on number of samples
    #7000 bus case needs data frame updated
    # if 
    try:
        df_labels = df_labels.set_index("Unnamed: 0")
        df_features = df_features.set_index("Unnamed: 0")
    except:
        df_labels = df_labels.set_index("Time")
        df_features = df_features.set_index("Time")
    
    #Convert to numpy
    X = df_features.to_numpy()
    y = df_labels.to_numpy()
    
    return X, y

def generate_directory(Params):
    
    #Create folder based on experimental settings
    folder = '{}/{}/RS_{}_Train_{}/{}/'.format(Params['folder'], Params['Dataset'],
                                               Params['random_state'],
                                               str(Params['train_percent']*100).split('.')[0],
                                               Params['mode'])
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    return folder

def main(Params):
    
    
    print('Starting Experiments...')
        
    #Set random state for reproducibility
    np.random.seed(Params['random_state'])
    random.seed(Params['random_state'])
    
    #Load data
    X, y = load_data(Params['data_dir'])
    
    #Split data into training and testing paritions
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=Params['train_percent'],
                                                        random_state=Params['random_state'])
    
    #Normalize the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    #Generate directory for saved results
    results_location = generate_directory(Params)
    
    #Perform cross validation to find best model(s)
    classifiers = cross_validation(X, y, Params,results_location)
  
    #Use best classifier on test data
    print('Testing best models...')
    start = time.time()
    
    #Create Precision-Recall Curve based on fuel type if default settings are true
    if ((np.round(Params['train_percent'],2) == 0.70) and (Params['Dataset'] == '7000 Bus') and (Params['random_state'] == 42)):
        plot_fuel_type = True
    else:
        plot_fuel_type = False
   
    for key, value in classifiers.items():
        best_model = value
        
        #Retrain model based on best parameters
        #If using saved model, do not retrain (replicate results in paper)
        if (Params['use_pretrained'] and (Params['Dataset'] == '7000 Bus')):
            pass
        else:
            best_model.fit(X_train_norm,y_train)
        
        #Get output predictions on test data
        y_predict = best_model.predict(X_test_norm)
        
        #Use function to generate results and save into folder
        # Take in y_true, y_predict, classifier
        target_names = []
        num_gens = y.shape[-1]
        
        for gen in range(0,num_gens):
            target_names.append('Generator {}'.format(gen))
            
        cm = multilabel_confusion_matrix(y_test, y_predict)
        hamming_score = hamming_loss(y_test,y_predict)
        zero_one_loss_val = zero_one_loss(y_test,y_predict)
        
        #Generate metrics for classifier
        report = classification_report(y_test, y_predict, target_names=target_names, output_dict=True)
          
        if Params['save_results']:
            
            #Generate directory for classier
            classifier_dir = '{}{}'.format(results_location,key)
            
            if not os.path.exists(classifier_dir):
                os.makedirs(classifier_dir)
            pd.DataFrame(y_predict).to_csv('{}/{}'.format(classifier_dir,'Predictions.csv')) 
            pd.DataFrame(y_test).to_csv('{}/{}'.format(classifier_dir,'Ground_Truth.csv'))
            
            df = pd.DataFrame(report)
            df_trans = df.transpose()
            
            #Save to CSV
            df_trans.to_csv('{}/{}.csv'.format(classifier_dir,'Classification_Report'))
            
            #MODEL EXPORT
            filename = '{}/{}.sav'.format(classifier_dir,key)
            pickle.dump(best_model, open(filename, 'wb'))
            
            #Save metrics and confusion matrix
            with open((classifier_dir + '/Hamming_Loss.txt'), "w") as output:
                output.write(str(hamming_score))
                
            with open((classifier_dir + '/Zero_One_Loss.txt'), "w") as output:
                output.write(str(zero_one_loss_val))
                
            np.save('{}/{}'.format(classifier_dir,'multi_label_confusion_matrix.'),cm)
            
            #Generate Precision-Recall Curves
            generate_PR_Curves(X_test_norm,y_test,best_model,key,classifier_dir,
                               Params['data_dir'],plot_fuel_types=plot_fuel_type)
    
    
    time_elapsed = time.time() - start
    print('Testing completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    print("************* Completed Unit Commitment Machine Learning Experiments *************")
         
def parse_args():
    parser = argparse.ArgumentParser(description='Run UC ML experiments for bus case')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction,
                        help='Save results of experiments (default: True), --no-save_results to set to False')
    parser.add_argument('--data_selection', type=int, default=1,
                        help='Dataset selection:  1: 7000 Bus Case, 2: 600 Bus Case, 3: 14 Bus Case')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Amount of details to print out during experiments')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for experiments (default: 42)')
    parser.add_argument('--n_iter_search', type=int, default=15,
                        help='Number of parameter settings that are sampled.Trades off runtime vs quality of the solution (default: 15)')
    parser.add_argument('--use_pretrained', default=True, action=argparse.BooleanOptionalAction,
                        help='Use saved weights of models, only for 7000 bus case (default: True), --no-use_pretrained to set to False')
    parser.add_argument('--folder', type=str, default='Saved_Models/Test/',
                        help='Location to save models')
    parser.add_argument('--train_percent', type=float, default=.70,
                        help='Train percentage for division of initial dataset between 0 and 1 (default: 70%)')
    parser.add_argument('-numRuns', type=int, default=1,
                        help='Number of experimental runs of random intialization (default: 3)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    params = Parameters(args)
    main(params)




