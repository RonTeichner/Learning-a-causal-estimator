#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 10:22:05 2021

@author: 
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


enableTrain = True 
enableSparse = True

enableTest = True
onlyDedicated = False
noImprove = False
jointEstimators = True
enableOverwriteStatistics = False

enablePlotTest = True

if enableSparse:
    sparseStr = 'sparse_'
else:
    sparseStr = ''
    
phoneAnalysisFileNamesTrainData = ['nexus4', 's3', 's3mini']  #, 'samsungold', 'lgwatch']  #, 'lgwatch']
phoneSavedSmootherFileNames = [sparseStr + 'smoother_trainedOn_' + phoneAnalysisFileNameTrainData for phoneAnalysisFileNameTrainData in phoneAnalysisFileNamesTrainData]
phoneSavedFilterFileNames = [sparseStr + 'trainedOn_' + phoneAnalysisFileNameTrainData for phoneAnalysisFileNameTrainData in phoneAnalysisFileNamesTrainData]
#phoneAnalysisFileNameTestDataFiles = ['nexus4', 's3', 's3mini', 'samsungold', 'lgwatch', 'gear']  #['nexus4', 's3', 's3mini', 'samsungold', 'lgwatch', 'gear']  # {'s3', 'train_nexus4', 'lgwatch', 'gear'}

fs = 1/0.005
    
if enableTrain:
    for smootherPhoneModel, phoneSavedSmootherFileName in zip(phoneAnalysisFileNamesTrainData, phoneSavedSmootherFileNames):
        if not(smootherPhoneModel == 's3mini'): continue
        for phoneAnalysisFileNameTrainData in phoneAnalysisFileNamesTrainData:  # the improved filter
            if phoneAnalysisFileNameTrainData == smootherPhoneModel: continue
            if not(phoneAnalysisFileNameTrainData == 's3mini'): continue
        
            phoneSavedModelFileName = sparseStr + 'improvedFilterFor_' + phoneAnalysisFileNameTrainData + '_trainedOnSmootherOf_' + smootherPhoneModel
            print(f'starting {phoneSavedModelFileName}')
        
            enablePlots = False
            phoneCompleteDataset = pickle.load(open(phoneAnalysisFileNameTrainData + '_dataset.pt', 'rb'))
            
            model = phoneCompleteDataset.metaDataDf['Classification'].unique().tolist()[0]
            classes = phoneCompleteDataset.phonesDf['gt']
            values, counts = np.unique(classes, return_counts=True)
            counts = counts/counts.sum()
            values = ['noClass', 'stand', 'sit', 'walk', 'stairsup', 'stairsdown', 'bike']
            #plt.bar(values, counts)
            #plt.title(model)
            #plt.show()
            
            statisticsDict = {'classDistribution': (values, counts), 'mu': phoneCompleteDataset.mu, 'Sigma_minus_half': phoneCompleteDataset.Sigma_minus_half, 'Sigma_half': phoneCompleteDataset.Sigma_half}
            print(f'dataset contains {len(phoneCompleteDataset)} time series')
            
            # training properties:    
            trainOnNormalizedData = True    
            nTrainsForCrossValidation = 1
            nTrainsOnSameSplit = 3 # not more than 1 because train indices will not match
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
            
            modelDict = {'enableSparse': enableSparse, 'trainedOn': smootherPhoneModel, 'smoother': True, 'nClasses': nClasses, 'hidden_dim': hidden_dim, 'num_layers': num_layers, 'useSelectedFeatures': useSelectedFeatures, 'allFeatures': allFeatures, 'featuresIncludeInTrainIndices': featuresIncludeInTrainIndices, 'nFeatures': phoneCompleteDataset.nFeatures, 'trainOnNormalizedData': trainOnNormalizedData, 'statisticsDict': statisticsDict, 'fs': 1/0.005}
            
            
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
                
                if enableOverwriteStatistics:
                    statisticsDict = {'classDistribution': (values, counts), 'mu': phoneCompleteDataset.mu, 'Sigma_minus_half': phoneCompleteDataset.Sigma_minus_half, 'Sigma_half': phoneCompleteDataset.Sigma_half}
                    smootherModelDict['statisticsDict'] = statisticsDict
                    if smootherModelDict['useSelectedFeatures']:
                        # modelDict['statisticsDict']['mu'] includes the statistics of the time-axis at the last coordinate
                        idx = smootherModelDict['featuresIncludeInTrainIndices']
                        savedSmoother.means = nn.parameter.Parameter(torch.tensor(smootherModelDict['statisticsDict']['mu'][idx], dtype=torch.float), requires_grad=False)
                        Sigma_minus_half = np.diag(np.diag(smootherModelDict['statisticsDict']['Sigma_minus_half'])[idx])
                        savedSmoother.Sigma_minus_half = nn.parameter.Parameter(torch.tensor(Sigma_minus_half, dtype=torch.float), requires_grad=False)            
                    else:                
                        savedSmoother.means = nn.parameter.Parameter(torch.tensor(smootherModelDict['statisticsDict']['mu'], dtype=torch.float), requires_grad=False)
                        savedSmoother.Sigma_minus_half = nn.parameter.Parameter(torch.tensor(smootherModelDict['statisticsDict']['Sigma_minus_half'], dtype=torch.float), requires_grad=False)
                    
                    
                savedSmoother.eval()
                
                Filter_rnn = RNN_Filter(input_dim = len(modelDict['allFeatures']), hidden_dim=modelDict['hidden_dim'], output_dim=modelDict['nClasses'], num_layers=modelDict['num_layers'], modelDict=modelDict)
                #Filter_rnn.load_state_dict(torch.load(phoneSavedSmootherFileName  + '_model.pt')) 
                
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
        if not(phoneAnalysisFileNameTestData == 's3mini'): continue
            
        # load phone's dataset and dedicated filter for reference performance
        print(f'creating test dataset, filename {phoneAnalysisFileNameTestData}')
        phoneCompleteDataset = pickle.load(open(phoneAnalysisFileNameTestData + '_dataset.pt', 'rb'))    
        print(f'total of {np.round((phoneCompleteDataset.phonesDf.shape[0]-1)/fs)} sec')
        
        if not(jointEstimators):
            dedicatedFilterModelDict = pickle.load(open(phoneSavedFilterFileName + '_modelDict.pt', 'rb'))
            if dedicatedFilterModelDict['datasetFile'] == phoneAnalysisFileNameTestData + '_dataset.pt':
                dedicatedFilter_trainData = Subset(phoneCompleteDataset, dedicatedFilterModelDict['trainIndices'])        
                dedicatedFilter_validationData = Subset(phoneCompleteDataset, dedicatedFilterModelDict['validationIndices'])        
            else:
                dedicatedFilter_trainData, dedicatedFilter_validationData = phoneCompleteDataset, phoneCompleteDataset
                assert False,'this should not happen'
            
            dedicatedFilter_trainLoader = DataLoader(dedicatedFilter_trainData, batch_size=20, shuffle=True, num_workers=0)     
            dedicatedFilter_validationLoader = DataLoader(dedicatedFilter_validationData, batch_size=20*8, shuffle=True, num_workers=0)                 
            
            estimatorRnn = RNN_Filter(input_dim = len(dedicatedFilterModelDict['allFeatures']), hidden_dim=dedicatedFilterModelDict['hidden_dim'], output_dim=dedicatedFilterModelDict['nClasses'], num_layers=dedicatedFilterModelDict['num_layers'], modelDict=dedicatedFilterModelDict)
            estimatorRnn.load_state_dict(torch.load(phoneSavedFilterFileName  + '_model.pt'))
            
            print(f'starting test of {phoneAnalysisFileNameTestData} on model {phoneSavedFilterFileName}')
            _, _, dedicatedFilter_meanLikelihood_vsTime_tuple = trainModel(estimatorRnn, dedicatedFilter_trainLoader, dedicatedFilter_validationLoader, phoneCompleteDataset, enableDataParallel, dedicatedFilterModelDict, True, 'test')
        
        
        dedicatedSmootherModelDict = pickle.load(open(phoneSavedSmootherFileName + '_modelDict.pt', 'rb'))
        if dedicatedSmootherModelDict['datasetFile'] == phoneAnalysisFileNameTestData + '_dataset.pt':
            dedicatedSmoother_trainData = Subset(phoneCompleteDataset, dedicatedSmootherModelDict['trainIndices'])        
            dedicatedSmoother_validationData = Subset(phoneCompleteDataset, dedicatedSmootherModelDict['validationIndices'])        
        else:
            dedicatedSmoother_trainData, dedicatedSmoother_validationData = phoneCompleteDataset, phoneCompleteDataset
            assert False,'this should not happen'
        
        dedicatedSmoother_trainLoader = DataLoader(dedicatedSmoother_trainData, batch_size=20, shuffle=True, num_workers=0)     
        dedicatedSmoother_validationLoader = DataLoader(dedicatedSmoother_validationData, batch_size=20*8, shuffle=True, num_workers=0)                 
        
        estimatorRnn = RNN_Filter(input_dim = len(dedicatedSmootherModelDict['allFeatures']), hidden_dim=dedicatedSmootherModelDict['hidden_dim'], output_dim=dedicatedSmootherModelDict['nClasses'], num_layers=dedicatedSmootherModelDict['num_layers'], modelDict=dedicatedSmootherModelDict)
        estimatorRnn.load_state_dict(torch.load(phoneSavedSmootherFileName  + '_model.pt'))
        
        print(f'starting test of {phoneAnalysisFileNameTestData} on model {phoneSavedSmootherFileName}')
        _, _, dedicatedSmoother_meanLikelihood_vsTime_tuple = trainModel(estimatorRnn, dedicatedSmoother_trainLoader, dedicatedSmoother_validationLoader, phoneCompleteDataset, enableDataParallel, dedicatedSmootherModelDict, True, 'test')
        
        if jointEstimators:
            dedicatedFilter_meanLikelihood_vsTime_tuple = (dedicatedSmoother_meanLikelihood_vsTime_tuple[0], dedicatedSmoother_meanLikelihood_vsTime_tuple[1])
            dedicatedSmoother_meanLikelihood_vsTime_tuple = (dedicatedSmoother_meanLikelihood_vsTime_tuple[3], dedicatedSmoother_meanLikelihood_vsTime_tuple[4])
        
        if onlyDedicated:
            # plots:
            resTuple=dedicatedFilter_meanLikelihood_vsTime_tuple
            plt.plot(resTuple[1], resTuple[0], label=phoneAnalysisFileNameTestData + ' on ' + phoneAnalysisFileNameTestData + ' filter')
            
            resTuple=dedicatedSmoother_meanLikelihood_vsTime_tuple            
            plt.plot(resTuple[1], resTuple[0], label=phoneAnalysisFileNameTestData + ' on ' + phoneAnalysisFileNameTestData + ' smoother')
            
            plt.xlabel('sec from change of state')
            plt.ylabel('likelihood')
            plt.grid()
            #plt.ylim([0.4, 0.85])
            #plt.xlim([0, 4])
            plt.legend()
            plt.show()
            
            continue
        
        for testedPhoneModel in phoneAnalysisFileNamesTrainData:
            #if not(testedPhoneModel == 's3'): continue
            if testedPhoneModel == phoneAnalysisFileNameTestData: continue
            nonDedicatedFilterFileName, nonDedicatedSmootherFileName = phoneSavedFilterFileName, phoneSavedSmootherFileName
            improvedFilterFileName = sparseStr + 'improvedFilterFor_' + testedPhoneModel + '_trainedOnSmootherOf_' + phoneAnalysisFileNameTestData
            
            phoneCompleteDataset = pickle.load(open(testedPhoneModel + '_dataset.pt', 'rb'))    
            
            # non dedicated filter:
            if not(jointEstimators):
                nonDedicatedFilterModelDict = pickle.load(open(nonDedicatedFilterFileName + '_modelDict.pt', 'rb'))
                if nonDedicatedFilterModelDict['datasetFile'] == testedPhoneModel + '_dataset.pt':
                    assert False,'this should not happen'
                    nonDedicatedFilter_trainData = Subset(phoneCompleteDataset, nonDedicatedFilterModelDict['trainIndices'])        
                    nonDedicatedFilter_validationData = Subset(phoneCompleteDataset, nonDedicatedFilterModelDict['validationIndices'])        
                else:
                    nonDedicatedFilter_trainData, nonDedicatedFilter_validationData = phoneCompleteDataset, phoneCompleteDataset
                    
                
                nonDedicatedFilter_trainLoader = DataLoader(nonDedicatedFilter_trainData, batch_size=20, shuffle=True, num_workers=0)     
                nonDedicatedFilter_validationLoader = DataLoader(nonDedicatedFilter_validationData, batch_size=20*8, shuffle=True, num_workers=0)                 
                
                estimatorRnn = RNN_Filter(input_dim = len(nonDedicatedFilterModelDict['allFeatures']), hidden_dim=nonDedicatedFilterModelDict['hidden_dim'], output_dim=nonDedicatedFilterModelDict['nClasses'], num_layers=nonDedicatedFilterModelDict['num_layers'], modelDict=nonDedicatedFilterModelDict)
                estimatorRnn.load_state_dict(torch.load(nonDedicatedFilterFileName  + '_model.pt'))
                
                if enableOverwriteStatistics:
                    statisticsDict = {'classDistribution': nonDedicatedFilterModelDict['statisticsDict']['classDistribution'], 'mu': phoneCompleteDataset.mu, 'Sigma_minus_half': phoneCompleteDataset.Sigma_minus_half, 'Sigma_half': phoneCompleteDataset.Sigma_half}
                    nonDedicatedFilterModelDict['statisticsDict'] = statisticsDict
                    if nonDedicatedFilterModelDict['useSelectedFeatures']:
                        # modelDict['statisticsDict']['mu'] includes the statistics of the time-axis at the last coordinate
                        idx = nonDedicatedFilterModelDict['featuresIncludeInTrainIndices']
                        estimatorRnn.means = nn.parameter.Parameter(torch.tensor(nonDedicatedFilterModelDict['statisticsDict']['mu'][idx], dtype=torch.float), requires_grad=False)
                        Sigma_minus_half = np.diag(np.diag(nonDedicatedFilterModelDict['statisticsDict']['Sigma_minus_half'])[idx])
                        estimatorRnn.Sigma_minus_half = nn.parameter.Parameter(torch.tensor(Sigma_minus_half, dtype=torch.float), requires_grad=False)            
                    else:                
                        estimatorRnn.means = nn.parameter.Parameter(torch.tensor(nonDedicatedFilterModelDict['statisticsDict']['mu'], dtype=torch.float), requires_grad=False)
                        estimatorRnn.Sigma_minus_half = nn.parameter.Parameter(torch.tensor(nonDedicatedFilterModelDict['statisticsDict']['Sigma_minus_half'], dtype=torch.float), requires_grad=False)
                
                print(f'starting test of {testedPhoneModel} on model {nonDedicatedFilterFileName}')
                _, _, nonDedicatedFilter_meanLikelihood_vsTime_tuple = trainModel(estimatorRnn, nonDedicatedFilter_trainLoader, nonDedicatedFilter_validationLoader, phoneCompleteDataset, enableDataParallel, nonDedicatedFilterModelDict, True, 'test')
            
            # non dedicated smoother:
            nonDedicatedSmootherModelDict = pickle.load(open(nonDedicatedSmootherFileName + '_modelDict.pt', 'rb'))
            if nonDedicatedSmootherModelDict['datasetFile'] == testedPhoneModel + '_dataset.pt':
                assert False,'this should not happen'
                nonDedicatedSmoother_trainData = Subset(phoneCompleteDataset, nonDedicatedSmootherModelDict['trainIndices'])        
                nonDedicatedSmoother_validationData = Subset(phoneCompleteDataset, nonDedicatedSmootherModelDict['validationIndices'])        
            else:
                nonDedicatedSmoother_trainData, nonDedicatedSmoother_validationData = phoneCompleteDataset, phoneCompleteDataset
                
            
            nonDedicatedSmoother_trainLoader = DataLoader(nonDedicatedSmoother_trainData, batch_size=20, shuffle=True, num_workers=0)     
            nonDedicatedSmoother_validationLoader = DataLoader(nonDedicatedSmoother_validationData, batch_size=20*8, shuffle=True, num_workers=0)                 
            
            estimatorRnn = RNN_Filter(input_dim = len(nonDedicatedSmootherModelDict['allFeatures']), hidden_dim=nonDedicatedSmootherModelDict['hidden_dim'], output_dim=nonDedicatedSmootherModelDict['nClasses'], num_layers=nonDedicatedSmootherModelDict['num_layers'], modelDict=nonDedicatedSmootherModelDict)
            estimatorRnn.load_state_dict(torch.load(nonDedicatedSmootherFileName  + '_model.pt'))
            
            if enableOverwriteStatistics:
                statisticsDict = {'classDistribution': nonDedicatedSmootherModelDict['statisticsDict']['classDistribution'], 'mu': phoneCompleteDataset.mu, 'Sigma_minus_half': phoneCompleteDataset.Sigma_minus_half, 'Sigma_half': phoneCompleteDataset.Sigma_half}
                nonDedicatedSmootherModelDict['statisticsDict'] = statisticsDict
                if nonDedicatedSmootherModelDict['useSelectedFeatures']:
                    # modelDict['statisticsDict']['mu'] includes the statistics of the time-axis at the last coordinate
                    idx = nonDedicatedSmootherModelDict['featuresIncludeInTrainIndices']
                    estimatorRnn.means = nn.parameter.Parameter(torch.tensor(nonDedicatedSmootherModelDict['statisticsDict']['mu'][idx], dtype=torch.float), requires_grad=False)
                    Sigma_minus_half = np.diag(np.diag(nonDedicatedSmootherModelDict['statisticsDict']['Sigma_minus_half'])[idx])
                    estimatorRnn.Sigma_minus_half = nn.parameter.Parameter(torch.tensor(Sigma_minus_half, dtype=torch.float), requires_grad=False)            
                else:                
                    estimatorRnn.means = nn.parameter.Parameter(torch.tensor(nonDedicatedSmootherModelDict['statisticsDict']['mu'], dtype=torch.float), requires_grad=False)
                    estimatorRnn.Sigma_minus_half = nn.parameter.Parameter(torch.tensor(nonDedicatedSmootherModelDict['statisticsDict']['Sigma_minus_half'], dtype=torch.float), requires_grad=False)
            
            
            print(f'starting test of {testedPhoneModel} on model {nonDedicatedSmootherFileName}')
            _, _, nonDedicatedSmoother_meanLikelihood_vsTime_tuple = trainModel(estimatorRnn, nonDedicatedSmoother_trainLoader, nonDedicatedSmoother_validationLoader, phoneCompleteDataset, enableDataParallel, nonDedicatedSmootherModelDict, True, 'test')
            
            if jointEstimators:
                nonDedicatedFilter_meanLikelihood_vsTime_tuple = (nonDedicatedSmoother_meanLikelihood_vsTime_tuple[0], nonDedicatedSmoother_meanLikelihood_vsTime_tuple[1])
                nonDedicatedSmoother_meanLikelihood_vsTime_tuple = (nonDedicatedSmoother_meanLikelihood_vsTime_tuple[3], nonDedicatedSmoother_meanLikelihood_vsTime_tuple[4])
            
            if not(noImprove):
                # improved filter:
                improvedFilterModelDict = pickle.load(open(improvedFilterFileName + '_modelDict.pt', 'rb'))
                
                if improvedFilterModelDict['datasetFile'] == testedPhoneModel + '_dataset.pt':                
                    improvedFilter_trainData = Subset(phoneCompleteDataset, improvedFilterModelDict['trainIndices'])        
                    improvedFilter_validationData = Subset(phoneCompleteDataset, improvedFilterModelDict['validationIndices'])        
                else:
                    assert False,'this should not happen'
                    improvedFilter_trainData, improvedFilter_validationData = phoneCompleteDataset, phoneCompleteDataset
                
                improvedFilter_trainLoader = DataLoader(improvedFilter_trainData, batch_size=20, shuffle=True, num_workers=0)     
                improvedFilter_validationLoader = DataLoader(improvedFilter_validationData, batch_size=20*8, shuffle=True, num_workers=0)                 
                
                estimatorRnn = RNN_Filter(input_dim = len(improvedFilterModelDict['allFeatures']), hidden_dim=improvedFilterModelDict['hidden_dim'], output_dim=improvedFilterModelDict['nClasses'], num_layers=improvedFilterModelDict['num_layers'], modelDict=improvedFilterModelDict)
                estimatorRnn.load_state_dict(torch.load(improvedFilterFileName  + '_model.pt'))
                
                if enableOverwriteStatistics:
                    statisticsDict = {'classDistribution': improvedFilterModelDict['statisticsDict']['classDistribution'], 'mu': phoneCompleteDataset.mu, 'Sigma_minus_half': phoneCompleteDataset.Sigma_minus_half, 'Sigma_half': phoneCompleteDataset.Sigma_half}
                    improvedFilterModelDict['statisticsDict'] = statisticsDict
                    if improvedFilterModelDict['useSelectedFeatures']:
                        # modelDict['statisticsDict']['mu'] includes the statistics of the time-axis at the last coordinate
                        idx = improvedFilterModelDict['featuresIncludeInTrainIndices']
                        estimatorRnn.means = nn.parameter.Parameter(torch.tensor(improvedFilterModelDict['statisticsDict']['mu'][idx], dtype=torch.float), requires_grad=False)
                        Sigma_minus_half = np.diag(np.diag(improvedFilterModelDict['statisticsDict']['Sigma_minus_half'])[idx])
                        estimatorRnn.Sigma_minus_half = nn.parameter.Parameter(torch.tensor(Sigma_minus_half, dtype=torch.float), requires_grad=False)            
                    else:                
                        estimatorRnn.means = nn.parameter.Parameter(torch.tensor(improvedFilterModelDict['statisticsDict']['mu'], dtype=torch.float), requires_grad=False)
                        estimatorRnn.Sigma_minus_half = nn.parameter.Parameter(torch.tensor(improvedFilterModelDict['statisticsDict']['Sigma_minus_half'], dtype=torch.float), requires_grad=False)
                
                print(f'starting test of {testedPhoneModel} on model {improvedFilterFileName}')
                _, _, improvedFilter_meanLikelihood_vsTime_tuple = trainModel(estimatorRnn, improvedFilter_trainLoader, improvedFilter_validationLoader, phoneCompleteDataset, enableDataParallel, improvedFilterModelDict, True, 'test')
                improvedFilter_meanLikelihood_vsTime_tuple = (improvedFilter_meanLikelihood_vsTime_tuple[0], improvedFilter_meanLikelihood_vsTime_tuple[1])
            else:                
                improvedFilter_meanLikelihood_vsTime_tuple = (np.zeros_like(dedicatedSmoother_meanLikelihood_vsTime_tuple[0]), dedicatedSmoother_meanLikelihood_vsTime_tuple[1])
                
            
            # plots:
            resTuple=dedicatedFilter_meanLikelihood_vsTime_tuple
            plt.plot(resTuple[1], resTuple[0], label=phoneAnalysisFileNameTestData + ' on ' + phoneAnalysisFileNameTestData + ' filter')
            
            resTuple=dedicatedSmoother_meanLikelihood_vsTime_tuple            
            plt.plot(resTuple[1], resTuple[0], label=phoneAnalysisFileNameTestData + ' on ' + phoneAnalysisFileNameTestData + ' smoother')
            
            resTuple=nonDedicatedFilter_meanLikelihood_vsTime_tuple
            plt.plot(resTuple[1], resTuple[0], label=testedPhoneModel + ' on ' + phoneAnalysisFileNameTestData + ' filter')
            
            resTuple=nonDedicatedSmoother_meanLikelihood_vsTime_tuple
            plt.plot(resTuple[1], resTuple[0], label=testedPhoneModel + ' on ' + phoneAnalysisFileNameTestData + ' smoother')
            
            resTuple=improvedFilter_meanLikelihood_vsTime_tuple
            plt.plot(resTuple[1], resTuple[0], label=testedPhoneModel + ' on ' + 'improved filter')                        
            
            plt.xlabel('sec from change of state')
            plt.ylabel('likelihood')
            plt.grid()
            #plt.ylim([0.4, 0.85])
            #plt.xlim([0, 4])
            plt.legend()
            plt.show()
            
            resultDict = {'testedPhone': testedPhoneModel, 'estimatorPhone':  phoneAnalysisFileNameTestData, 'dedicatedFilter': dedicatedFilter_meanLikelihood_vsTime_tuple, 'dedicatedSmoother': dedicatedSmoother_meanLikelihood_vsTime_tuple, 'nonDedicatedFilter': nonDedicatedFilter_meanLikelihood_vsTime_tuple, 'nonDedicatedSmoother': nonDedicatedSmoother_meanLikelihood_vsTime_tuple, 'learnedFilter': improvedFilter_meanLikelihood_vsTime_tuple}
            resultList.append(resultDict)

    pickle.dump(resultList, open('allResults.pt', 'wb'))
    
if enablePlotTest:
    resultList = pickle.load(open('allResults.pt', 'rb'))
        
    resultsDict = resultList[-2]
    plt.close()
    
    plt.figure(figsize=(8,6))
    plt.subplot(1, 2, 1)
    # plots:
    resTuple=resultsDict['dedicatedFilter']
    plt.plot(resTuple[1], resTuple[0], linestyle=(0, (5, 10)), label=resultsDict['estimatorPhone'] + ' on ' + resultsDict['estimatorPhone'] + ' filter')
    
    resTuple=resultsDict['dedicatedSmoother']            
    plt.plot(resTuple[1], resTuple[0], linestyle='dotted', label=resultsDict['estimatorPhone'] + ' on ' + resultsDict['estimatorPhone'] + ' smoother')
    
    resTuple=resultsDict['nonDedicatedFilter']            
    plt.plot(resTuple[1], resTuple[0], linestyle='dashed', label=resultsDict['testedPhone'] + ' on ' + resultsDict['estimatorPhone'] + ' filter')
    
    resTuple=resultsDict['nonDedicatedSmoother']            
    plt.plot(resTuple[1], resTuple[0], linestyle='dashdot', label=resultsDict['testedPhone'] + ' on ' + resultsDict['estimatorPhone'] + ' smoother')
    
    resTuple=resultsDict['learnedFilter']            
    plt.plot(resTuple[1], resTuple[0], linestyle='solid', label=resultsDict['testedPhone'] + ' on ' + 'improved filter')                        
    
    plt.xlabel('sec from change of state')
    plt.ylabel('Posterio probability of correct category')
    plt.grid()
    plt.ylim([0.2, 0.75])
    plt.xlim([0, 6])
    plt.legend()
    
    resultsDict = resultList[-1]
    plt.subplot(1, 2, 2)
    # plots:
    resTuple=resultsDict['dedicatedFilter']
    plt.plot(resTuple[1], resTuple[0], linestyle=(0, (5, 10)), label=resultsDict['estimatorPhone'] + ' on ' + resultsDict['estimatorPhone'] + ' filter')
    
    resTuple=resultsDict['dedicatedSmoother']            
    plt.plot(resTuple[1], resTuple[0], linestyle='dotted', label=resultsDict['estimatorPhone'] + ' on ' + resultsDict['estimatorPhone'] + ' smoother')
    
    resTuple=resultsDict['nonDedicatedFilter']            
    plt.plot(resTuple[1], resTuple[0], linestyle='dashed', label=resultsDict['testedPhone'] + ' on ' + resultsDict['estimatorPhone'] + ' filter')
    
    resTuple=resultsDict['nonDedicatedSmoother']            
    plt.plot(resTuple[1], resTuple[0], linestyle='dashdot', label=resultsDict['testedPhone'] + ' on ' + resultsDict['estimatorPhone'] + ' smoother')
    
    resTuple=resultsDict['learnedFilter']            
    plt.plot(resTuple[1], resTuple[0], linestyle='solid', label=resultsDict['testedPhone'] + ' on ' + 'improved filter')                        
    
    plt.xlabel('sec from change of state')
    #plt.ylabel('Posterio probability of correct category')
    plt.grid()
    plt.ylim([0.2, 0.75])
    plt.xlim([0, 6])
    plt.legend()
    plt.show()
            
            
            
            
        
        
        
    
