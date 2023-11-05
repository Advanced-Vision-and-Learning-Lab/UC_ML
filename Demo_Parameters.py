# -*- coding: utf-8 -*-
"""
Parameters for histogram layer experiments
Only change parameters in this file before running
demo.py
@author: jpeeples 
"""
import os
import sys

def Parameters(args):
    ######## ONLY CHANGE PARAMETERS BELOW ########
    #Flag for if results are to be saved out
    #Set to True to save results out and False to not save results
    save_results = args.save_results
    
    #Location to store trained models
    #Always add slash (/) after folder name
    folder = args.folder
    
    #Select dataset. Set to number of desired dataset
    data_selection = args.data_selection
    Dataset_names = {1: '7000 Bus', 2: '600 Bus', 3: '14 Bus'}
    
    #Flag to use pretrained models from our experiments or train from scratch (default: True)
    use_pretrained = args.use_pretrained
    
    #Set training, random state, and number of parameters settings for experiments
    train_percent = args.train_percent
    random_state = args.random_state
    n_iter_search = args.n_iter_search
    
    #Set amount of details to print out (anything non-zero will result in printing)
    verbose = args.verbose
    
    #Set number of experimental runs/trials
    numRuns = args.numRuns
   
    #Visualization of results parameters
    #Visualization parameters for figures
    fig_size = 12
    font_size = 24
    
    #Flag for TSNE visuals, set to True to create TSNE visual of features
    #Set to false to not generate TSNE visuals
    #Number of images to view for TSNE (defaults to all training imgs unless
    #value is less than total training images).
    TSNE_visual = True
    Num_TSNE_images = 10000
    
    ######## ONLY CHANGE PARAMETERS ABOVE ########
    #Location of texture datasets
    Data_dirs = {'7000 Bus': './Datasets/7000 Bus/',
                 '600 Bus': './Datasets/600 Bus/',
                 '14 Bus': './Datasets/14 Bus/'}
  
    Dataset = Dataset_names[data_selection]
    data_dir = Data_dirs[Dataset]
    
    if ('use_pretrained' and (Dataset == '7000 Bus')):
        mode = 'Pretrained'
    else:
        mode = 'Scratch'
    
    #Return dictionary of parameters
    Params = {'save_results': save_results,'folder': folder,
              'Dataset': Dataset, 'data_dir': data_dir,
              'use_pretrained': use_pretrained,
              'random_state': random_state,
              'train_percent': train_percent,
              'mode': mode,'TSNE_visual': TSNE_visual,
              'Num_TSNE_images': Num_TSNE_images,'fig_size': fig_size,
              'font_size': font_size, 'numRuns': numRuns, 'n_iter_search': n_iter_search,
              'verbose': verbose}
    
    return Params
