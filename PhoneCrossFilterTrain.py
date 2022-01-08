#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 10:22:05 2021

@author: ront
"""
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split
import copy
from PhoneAnalysis_func import *


enableTrain = False
enableTest = True
enableOverwriteStatistics = False


phoneAnalysisFileNamesTrainData = ['nexus4', 's3', 's3mini', 'samsungold']  #, 'lgwatch']
phoneSavedSmootherFileNames = ['smoother_trainedOn_' + phoneAnalysisFileNameTrainData for phoneAnalysisFileNameTrainData in phoneAnalysisFileNamesTrainData]
phoneSavedImprovedFilterFileNames = ['improvedFilterFor_' + phoneAnalysisFileNameTrainData + '_trainedOnSmootherOf_' + SphoneAnalysisFileNameTrainData for phoneAnalysisFileNameTrainData in phoneAnalysisFileNamesTrainData for SphoneAnalysisFileNameTrainData in phoneAnalysisFileNamesTrainData]

phoneSavedFilterFileNames = ['trainedOn_' + phoneAnalysisFileNameTrainData for phoneAnalysisFileNameTrainData in phoneAnalysisFileNamesTrainData]
#phoneAnalysisFileNameTestDataFiles = ['nexus4', 's3', 's3mini', 'samsungold', 'lgwatch', 'gear']  #['nexus4', 's3', 's3mini', 'samsungold', 'lgwatch', 'gear']  # {'s3', 'train_nexus4', 'lgwatch', 'gear'}

fs = 1/0.005
    
if enableTrain:
    for smootherPhoneModel, phoneSavedSmootherFileName in zip(phoneAnalysisFileNamesTrainData, phoneSavedSmootherFileNames):
        for phoneAnalysisFileNameTrainData in phoneAnalysisFileNamesTrainData:  # the improved filter
            if phoneAnalysisFileNameTrainData == smootherPhoneModel: continue
        
            phoneSavedModelFileName = 'improvedFilterFor_' + phoneAnalysisFileNameTrainData + '_trainedOnSmootherOf_' + smootherPhoneModel
            print(f'starting {phoneSavedModelFileName}')
        
            enablePlots = True
            phoneCompleteDataset = pickle.load(open(phoneAnalysisFileNameTrainData + '_dataset.pt', 'rb'))
            
            model = phoneCompleteDataset.metaDataDf['Classification'].unique().tolist()[0]
            classes = phoneCompleteDataset.phonesDf['gt']
            values, counts = np.unique(classes, return_counts=True)
            counts = counts/counts.sum()
            values = ['noClass', 'stand', 'sit', 'walk', 'stairsup', 'stairsdown', 'bike']
            plt.bar(values, counts)
            plt.title(model)
            plt.show()
            
            statisticsDict = {'classDistribution': (values, counts), 'mu': phoneCompleteDataset.mu, 'Sigma_minus_half': phoneCompleteDataset.Sigma_minus_half, 'Sigma_half': phoneCompleteDataset.Sigma_half}
            print(f'dataset contains {len(phoneCompleteDataset)} time series')
            
            # training properties:    
            trainOnNormalizedData = True    
            nTrainsForCrossValidation = 1
            nTrainsOnSameSplit = 1 # not more than 1 because train indices will not match
            batchSize = 8*10
            validation_fraction = 0.3
            nValidation = int(validation_fraction*len(phoneCompleteDataset))
            nTrain = len(phoneCompleteDataset) - nValidation
            
            allFeatures = phoneCompleteDataset.phonesDf.columns[3:-1].tolist()
            featuresIncludeInTrainIndices = [0, 1, 2]  
            
            # create the trained model
            print('creating model')
            hidden_dim = 30
            num_layers = 1
            nClasses = phoneCompleteDataset.phonesDf['gt'].unique().shape[0]
            useSelectedFeatures = True
            
            enableDataParallel = True
            
            modelDict = {'trainedOn': smootherPhoneModel, 'smoother': False, 'nClasses': nClasses, 'hidden_dim': hidden_dim, 'num_layers': num_layers, 'useSelectedFeatures': useSelectedFeatures, 'allFeatures': allFeatures, 'featuresIncludeInTrainIndices': featuresIncludeInTrainIndices, 'nFeatures': phoneCompleteDataset.nFeatures, 'trainOnNormalizedData': trainOnNormalizedData, 'statisticsDict': statisticsDict, 'fs': 1/0.005}
            
            
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
                    if not(validationClassesFraction.shape[0] == trainClassesFraction.shape[0]): continue
                    validationClassesFraction_vs_trainClassesFraction = np.round(100*np.divide(np.abs(validationClassesFraction-trainClassesFraction), trainClassesFraction))
                    maxTrain_vs_validation_classDistributionDiff = validationClassesFraction_vs_trainClassesFraction.max()
                    print(f'max diff classes = {maxTrain_vs_validation_classDistributionDiff}')
                
                print(f'validation classes fraction {validationClassesFraction}')
                print(f'train classes fraction {trainClassesFraction}')
                trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True, num_workers=0)        
                validationLoader = DataLoader(validationData, batch_size=batchSize, shuffle=True, num_workers=0)
                
                modelDict['datasetFile'] = phoneAnalysisFileNameTrainData + '_dataset.pt'
                modelDict['trainIndices'], modelDict['validationIndices'] = trainLoader.dataset.indices, validationLoader.dataset.indices
                pickle.dump(modelDict, open(phoneSavedModelFileName +  '_modelDict.pt', 'wb'))
                
                smootherModelDict = pickle.load(open(phoneSavedSmootherFileName + '_modelDict.pt', 'rb'))
                savedSmoother = RNN_Filter(input_dim = len(smootherModelDict['allFeatures']), hidden_dim=smootherModelDict['hidden_dim'], output_dim=smootherModelDict['nClasses'], num_layers=smootherModelDict['num_layers'], modelDict=smootherModelDict)
                savedSmoother.load_state_dict(torch.load(phoneSavedSmootherFileName  + '_model.pt'))                
                savedSmoother.eval()
                
                Filter_rnn = RNN_Filter(input_dim = len(modelDict['allFeatures']), hidden_dim=modelDict['hidden_dim'], output_dim=modelDict['nClasses'], num_layers=modelDict['num_layers'], modelDict=modelDict)
                
                # train:  
                validationLoss = np.inf
                for s in range(nTrainsOnSameSplit):
                    print(f'starting training, attemp no. {s}')
                    Filter_rnn.apply(init_weights)
                    model_state_dict_s, validationLoss_s, _ = trainModel(Filter_rnn, trainLoader, validationLoader, phoneCompleteDataset, enableDataParallel, modelDict, enablePlots, 'train', True, savedSmoother)
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
    enableDataParallel = False
    resultList = list()
    for phoneAnalysisFileNameTestData, phoneSavedFilterFileName, phoneSavedSmootherFileName in zip(phoneAnalysisFileNamesTrainData, phoneSavedFilterFileNames, phoneSavedSmootherFileNames):
        #if not(phoneAnalysisFileNameTestData == 's3'): continue
        # load phone's dataset and dedicated filter for reference performance
        print(f'creating test dataset, filename {phoneAnalysisFileNameTestData}')
        phoneCompleteDataset = pickle.load(open(phoneAnalysisFileNameTestData + '_dataset.pt', 'rb'))    
        print(f'total of {np.round((phoneCompleteDataset.phonesDf.shape[0]-1)/fs)} sec')
        
        dedicatedFilterModelDict = pickle.load(open(phoneSavedFilterFileName + '_modelDict.pt', 'rb'))
        if dedicatedFilterModelDict['datasetFile'] == phoneAnalysisFileNameTestData + '_dataset.pt':
            dedicatedFilter_trainData = Subset(phoneCompleteDataset, dedicatedFilterModelDict['trainIndices'])        
            dedicatedFilter_validationData = Subset(phoneCompleteDataset, dedicatedFilterModelDict['validationIndices'])        
        else:
            dedicatedFilter_trainData, dedicatedFilter_validationData = phoneCompleteDataset, phoneCompleteDataset
            assert False,'this should not happen'
        
        dedicatedFilter_trainLoader = DataLoader(dedicatedFilter_trainData, batch_size=20, shuffle=True, num_workers=0)     
        dedicatedFilter_validationLoader = DataLoader(dedicatedFilter_validationData, batch_size=20*8, shuffle=True, num_workers=0)                 
        
        dedicatedFilter_rnn = RNN_Filter(input_dim = len(dedicatedFilterModelDict['allFeatures']), hidden_dim=dedicatedFilterModelDict['hidden_dim'], output_dim=dedicatedFilterModelDict['nClasses'], num_layers=dedicatedFilterModelDict['num_layers'], modelDict=dedicatedFilterModelDict)
        dedicatedFilter_rnn.load_state_dict(torch.load(phoneSavedFilterFileName  + '_model.pt'))
        
        print(f'starting test of {phoneAnalysisFileNameTestData} on model {phoneSavedFilterFileName}')
        _, _, dedicatedFilter_meanLikelihood_vsTime_tuple = trainModel(dedicatedFilter_rnn, dedicatedFilter_trainLoader, dedicatedFilter_validationLoader, phoneCompleteDataset, enableDataParallel, dedicatedFilterModelDict, True, 'test')
        
        
        dedicatedSmootherModelDict = pickle.load(open(phoneSavedSmootherFileName + '_modelDict.pt', 'rb'))
        if dedicatedSmootherModelDict['datasetFile'] == phoneAnalysisFileNameTestData + '_dataset.pt':
            dedicatedSmoother_trainData = Subset(phoneCompleteDataset, dedicatedSmootherModelDict['trainIndices'])        
            dedicatedSmoother_validationData = Subset(phoneCompleteDataset, dedicatedSmootherModelDict['validationIndices'])        
        else:
            dedicatedSmoother_trainData, dedicatedSmoother_validationData = phoneCompleteDataset, phoneCompleteDataset
            assert False,'this should not happen'
        
        dedicatedSmoother_trainLoader = DataLoader(dedicatedSmoother_trainData, batch_size=20, shuffle=True, num_workers=0)     
        dedicatedSmoother_validationLoader = DataLoader(dedicatedSmoother_validationData, batch_size=20*8, shuffle=True, num_workers=0)                 
        
        dedicatedSmoother_rnn = RNN_Filter(input_dim = len(dedicatedSmootherModelDict['allFeatures']), hidden_dim=dedicatedSmootherModelDict['hidden_dim'], output_dim=dedicatedSmootherModelDict['nClasses'], num_layers=dedicatedSmootherModelDict['num_layers'], modelDict=dedicatedSmootherModelDict)
        dedicatedSmoother_rnn.load_state_dict(torch.load(phoneSavedSmootherFileName  + '_model.pt'))
        
        print(f'starting test of {phoneAnalysisFileNameTestData} on model {phoneSavedSmootherFileName}')
        _, _, dedicatedSmoother_meanLikelihood_vsTime_tuple = trainModel(dedicatedSmoother_rnn, dedicatedSmoother_trainLoader, dedicatedSmoother_validationLoader, phoneCompleteDataset, enableDataParallel, dedicatedSmootherModelDict, True, 'test')
        
        for smootherPhoneModel, nonDedicatedFilterFileName, nonDedicatedSmootherFileName in zip(phoneAnalysisFileNamesTrainData, phoneSavedFilterFileNames, phoneSavedSmootherFileNames):
            #if not(smootherPhoneModel == 'nexus4'): continue
            if smootherPhoneModel == phoneAnalysisFileNameTestData: continue
            improvedFilterFileName = 'improvedFilterFor_' + phoneAnalysisFileNameTestData + '_trainedOnSmootherOf_' + smootherPhoneModel
            
            # non dedicated filter:
            nonDedicatedFilterModelDict = pickle.load(open(nonDedicatedFilterFileName + '_modelDict.pt', 'rb'))
            if nonDedicatedFilterModelDict['datasetFile'] == phoneAnalysisFileNameTestData + '_dataset.pt':
                assert False,'this should not happen'
                nonDedicatedFilter_trainData = Subset(phoneCompleteDataset, nonDedicatedFilterModelDict['trainIndices'])        
                nonDedicatedFilter_validationData = Subset(phoneCompleteDataset, nonDedicatedFilterModelDict['validationIndices'])        
            else:
                nonDedicatedFilter_trainData, nonDedicatedFilter_validationData = phoneCompleteDataset, phoneCompleteDataset
                
            
            nonDedicatedFilter_trainLoader = DataLoader(nonDedicatedFilter_trainData, batch_size=20, shuffle=True, num_workers=0)     
            nonDedicatedFilter_validationLoader = DataLoader(nonDedicatedFilter_validationData, batch_size=20*8, shuffle=True, num_workers=0)                 
            
            nonDedicatedFilter_rnn = RNN_Filter(input_dim = len(nonDedicatedFilterModelDict['allFeatures']), hidden_dim=nonDedicatedFilterModelDict['hidden_dim'], output_dim=nonDedicatedFilterModelDict['nClasses'], num_layers=nonDedicatedFilterModelDict['num_layers'], modelDict=nonDedicatedFilterModelDict)
            nonDedicatedFilter_rnn.load_state_dict(torch.load(nonDedicatedFilterFileName  + '_model.pt'))
            
            print(f'starting test of {phoneAnalysisFileNameTestData} on model {nonDedicatedFilterFileName}')
            _, _, nonDedicatedFilter_meanLikelihood_vsTime_tuple = trainModel(nonDedicatedFilter_rnn, nonDedicatedFilter_trainLoader, nonDedicatedFilter_validationLoader, phoneCompleteDataset, enableDataParallel, nonDedicatedFilterModelDict, True, 'test')
            
            # non dedicated smoother:
            nonDedicatedSmootherModelDict = pickle.load(open(nonDedicatedSmootherFileName + '_modelDict.pt', 'rb'))
            if nonDedicatedSmootherModelDict['datasetFile'] == phoneAnalysisFileNameTestData + '_dataset.pt':
                assert False,'this should not happen'
                nonDedicatedSmoother_trainData = Subset(phoneCompleteDataset, nonDedicatedSmootherModelDict['trainIndices'])        
                nonDedicatedSmoother_validationData = Subset(phoneCompleteDataset, nonDedicatedSmootherModelDict['validationIndices'])        
            else:
                nonDedicatedSmoother_trainData, nonDedicatedSmoother_validationData = phoneCompleteDataset, phoneCompleteDataset
                
            
            nonDedicatedSmoother_trainLoader = DataLoader(nonDedicatedSmoother_trainData, batch_size=20, shuffle=True, num_workers=0)     
            nonDedicatedSmoother_validationLoader = DataLoader(nonDedicatedSmoother_validationData, batch_size=20*8, shuffle=True, num_workers=0)                 
            
            nonDedicatedSmoother_rnn = RNN_Filter(input_dim = len(nonDedicatedSmootherModelDict['allFeatures']), hidden_dim=nonDedicatedSmootherModelDict['hidden_dim'], output_dim=nonDedicatedSmootherModelDict['nClasses'], num_layers=nonDedicatedSmootherModelDict['num_layers'], modelDict=nonDedicatedSmootherModelDict)
            nonDedicatedSmoother_rnn.load_state_dict(torch.load(nonDedicatedSmootherFileName  + '_model.pt'))
            
            print(f'starting test of {phoneAnalysisFileNameTestData} on model {nonDedicatedSmootherFileName}')
            _, _, nonDedicatedSmoother_meanLikelihood_vsTime_tuple = trainModel(nonDedicatedSmoother_rnn, nonDedicatedSmoother_trainLoader, nonDedicatedSmoother_validationLoader, phoneCompleteDataset, enableDataParallel, nonDedicatedSmootherModelDict, True, 'test')
            
            # improved filter:
            improvedFilterModelDict = pickle.load(open(improvedFilterFileName + '_modelDict.pt', 'rb'))
            
            if improvedFilterModelDict['datasetFile'] == phoneAnalysisFileNameTestData + '_dataset.pt':                
                improvedFilter_trainData = Subset(phoneCompleteDataset, improvedFilterModelDict['trainIndices'])        
                improvedFilter_validationData = Subset(phoneCompleteDataset, improvedFilterModelDict['validationIndices'])        
            else:
                assert False,'this should not happen'
                improvedFilter_trainData, improvedFilter_validationData = phoneCompleteDataset, phoneCompleteDataset
            
            improvedFilter_trainLoader = DataLoader(improvedFilter_trainData, batch_size=20, shuffle=True, num_workers=0)     
            improvedFilter_validationLoader = DataLoader(improvedFilter_validationData, batch_size=20*8, shuffle=True, num_workers=0)                 
            
            improvedFilter_rnn = RNN_Filter(input_dim = len(improvedFilterModelDict['allFeatures']), hidden_dim=improvedFilterModelDict['hidden_dim'], output_dim=improvedFilterModelDict['nClasses'], num_layers=improvedFilterModelDict['num_layers'], modelDict=improvedFilterModelDict)
            improvedFilter_rnn.load_state_dict(torch.load(improvedFilterFileName  + '_model.pt'))
            
            print(f'starting test of {phoneAnalysisFileNameTestData} on model {improvedFilterFileName}')
            _, _, improvedFilter_meanLikelihood_vsTime_tuple = trainModel(improvedFilter_rnn, improvedFilter_trainLoader, improvedFilter_validationLoader, phoneCompleteDataset, enableDataParallel, improvedFilterModelDict, True, 'test')
            
            # plots:
            resTuple=dedicatedFilter_meanLikelihood_vsTime_tuple
            plt.plot(resTuple[1], resTuple[0], label=phoneAnalysisFileNameTestData + ' on ' + phoneAnalysisFileNameTestData + ' filter')
            
            resTuple=dedicatedSmoother_meanLikelihood_vsTime_tuple            
            plt.plot(resTuple[1], resTuple[0], label=phoneAnalysisFileNameTestData + ' on ' + phoneAnalysisFileNameTestData + ' smoother')
            
            resTuple=nonDedicatedFilter_meanLikelihood_vsTime_tuple
            plt.plot(resTuple[1], resTuple[0], label=phoneAnalysisFileNameTestData + ' on ' + smootherPhoneModel + ' filter')
            
            resTuple=nonDedicatedSmoother_meanLikelihood_vsTime_tuple
            plt.plot(resTuple[1], resTuple[0], label=phoneAnalysisFileNameTestData + ' on ' + smootherPhoneModel + ' smoother')
            
            resTuple=improvedFilter_meanLikelihood_vsTime_tuple
            plt.plot(resTuple[1], resTuple[0], label=phoneAnalysisFileNameTestData + ' on ' + 'improved filter')                        
            
            plt.xlabel('sec from change of state')
            plt.ylabel('likelihood')
            plt.grid()
            plt.legend()
            plt.show()
            
            resultDict = {'testedPhone': phoneAnalysisFileNameTestData, 'estimatorPhone': smootherPhoneModel, 'dedicatedFilter': dedicatedFilter_meanLikelihood_vsTime_tuple, 'dedicatedSmoother': dedicatedSmoother_meanLikelihood_vsTime_tuple, 'nonDedicatedFilter': nonDedicatedFilter_meanLikelihood_vsTime_tuple, 'nonDedicatedSmoother': nonDedicatedSmoother_meanLikelihood_vsTime_tuple, 'learnedFilter': improvedFilter_meanLikelihood_vsTime_tuple}
            resultList.append(resultDict)

    pickle.dump(resultDict, open('allResults.pt', 'wb'))
            
            
            
            
            
        
        
        
    
