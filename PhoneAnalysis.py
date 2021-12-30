#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 10:22:05 2021

@author: ront
"""
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split
import copy
from PhoneAnalysis_func import *


createPhoneDataset = False
enableTrain = False
enableTest = True

phoneAnalysisFileNameTrainData = 's3'  #'train_nexus4'
phoneAnalysisFileNameTestData = 's3'  # {'s3', 'train_nexus4'}
phoneSavedModelFileName = 'trainedOnNexus4'


if createPhoneDataset:
    print(f'creating dataset')
    filePath = './Activity recognition exp/phonesData.pt'
    modelList = ['s3']  # ['nexus4']
    phoneCompleteDataset = PhoneDataset(filePath, modelList)
    pickle.dump(phoneCompleteDataset, open(phoneAnalysisFileNameTrainData + '_dataset.pt', 'wb'))
    
if enableTrain:
    enablePlots = True
    phoneCompleteDataset = pickle.load(open(phoneAnalysisFileNameTrainData + '_dataset.pt', 'rb'))
    statisticsDict = {'mu': phoneCompleteDataset.mu, 'Sigma_minus_half': phoneCompleteDataset.Sigma_minus_half, 'Sigma_half': phoneCompleteDataset.Sigma_half}
    print(f'dataset contains {len(phoneCompleteDataset)} time series')
    
    # training properties:    
    trainOnNormalizedData = True    
    nTrainsForCrossValidation = 1
    nTrainsOnSameSplit = 1
    batchSize = 8*10
    validation_fraction = 0.2
    nValidation = int(validation_fraction*len(phoneCompleteDataset))
    nTrain = len(phoneCompleteDataset) - nValidation
    
    allFeatures = phoneCompleteDataset.phonesDf.columns[3:-1].tolist()
    featuresIncludeInTrainIndices = [0, 1, 2]  #[0, 1, 21]
    
    # create the trained model
    print('creating model')
    hidden_dim = 20
    num_layers = 1
    nClasses = phoneCompleteDataset.phonesDf['gt'].unique().shape[0]
    useSelectedFeatures = False
    
    enableDataParallel = True
    
    modelDict = {'nClasses': nClasses, 'hidden_dim': hidden_dim, 'num_layers': num_layers, 'useSelectedFeatures': useSelectedFeatures, 'allFeatures': allFeatures, 'featuresIncludeInTrainIndices': featuresIncludeInTrainIndices, 'nFeatures': phoneCompleteDataset.nFeatures, 'trainOnNormalizedData': trainOnNormalizedData, 'statisticsDict': statisticsDict, 'fs': 1/0.005}
    pickle.dump(modelDict, open(phoneSavedModelFileName +  '_modelDict.pt', 'wb'))
    
    model_state_dict_list, validationLoss_list = list(), list()
    
    for trainIdx in range(nTrainsForCrossValidation):
        maxTrain_vs_validation_classDistributionDiff = np.Inf
        while maxTrain_vs_validation_classDistributionDiff > 25:  # a difference of 20% in one of the classes between train and validation
            trainData, validationData = random_split(phoneCompleteDataset,[nTrain, nValidation])
            
            # check if split is balanced:
            trainClasses = phoneCompleteDataset.phonesDf[phoneCompleteDataset.phonesDf['Id'].isin(trainData.indices)]['gt']
            values, counts = np.unique(trainClasses, return_counts=True)
            trainClassesFraction = np.round(100*counts/counts.sum())
            validationClasses = phoneCompleteDataset.phonesDf[phoneCompleteDataset.phonesDf['Id'].isin(validationData.indices)]['gt']
            values, counts = np.unique(validationClasses, return_counts=True)
            validationClassesFraction = np.round(100*counts/counts.sum())
            validationClassesFraction_vs_trainClassesFraction = np.round(100*np.divide(np.abs(validationClassesFraction-trainClassesFraction), trainClassesFraction))
            maxTrain_vs_validation_classDistributionDiff = validationClassesFraction_vs_trainClassesFraction.max()
            print(f'max diff classes = {maxTrain_vs_validation_classDistributionDiff}')
        
        
        trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True, num_workers=0)
        #trainTestLoader = DataLoader(trainData, batch_size=600, shuffle=False, num_workers=0)
        validationLoader = DataLoader(validationData, batch_size=batchSize, shuffle=False, num_workers=0)
        
        Filter_rnn = RNN_Filter(input_dim = len(modelDict['allFeatures']), hidden_dim=modelDict['hidden_dim'], output_dim=modelDict['nClasses'], num_layers=modelDict['num_layers'], modelDict=modelDict)
        
        # train:  
        validationLoss = np.inf
        for s in range(nTrainsOnSameSplit):
            print(f'starting training, attemp no. {s}')
            Filter_rnn.apply(init_weights)
            model_state_dict_s, validationLoss_s = trainModel(Filter_rnn, trainLoader, validationLoader, phoneCompleteDataset, enableDataParallel, modelDict, enablePlots, 'train')
            if validationLoss_s < validationLoss:
                validationLoss = validationLoss_s
                model_state_dict = model_state_dict_s
            
        model_state_dict_list.append(model_state_dict)
        validationLoss_list.append(validationLoss)                
    
    # save the net with median performance:
    validationLoss_array = np.asarray(validationLoss_list)
    Ann_idx = np.argsort(validationLoss_array)[0]    
    torch.save(model_state_dict_list[Ann_idx], phoneSavedModelFileName + '_model.pt')
    
if enableTest:
    print(f'creating test dataset, filename {phoneAnalysisFileNameTestData}')
    phoneCompleteDataset = pickle.load(open(phoneAnalysisFileNameTestData + '_dataset.pt', 'rb'))    
    testLoader = DataLoader(phoneCompleteDataset, batch_size=20, shuffle=False, num_workers=0)     
    
    modelDict = pickle.load(open(phoneSavedModelFileName + '_modelDict.pt', 'rb'))
    Filter_rnn = RNN_Filter(input_dim = len(modelDict['allFeatures']), hidden_dim=modelDict['hidden_dim'], output_dim=modelDict['nClasses'], num_layers=modelDict['num_layers'], modelDict=modelDict)
    Filter_rnn.load_state_dict(torch.load(phoneSavedModelFileName  + '_model.pt'))
    
    # test:
    print('starting test')
    #device = 'cpu'
    #Filter_rnn.eval().to(device)       
    
    _, _ = trainModel(Filter_rnn, testLoader, testLoader, phoneCompleteDataset, False, modelDict, True, 'test')
    
          
