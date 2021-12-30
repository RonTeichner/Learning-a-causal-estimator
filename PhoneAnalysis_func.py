#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 10:08:08 2021

@author: ront
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import pickle
from scipy.signal import find_peaks
from scipy import interpolate
import time

class PhoneDataset(Dataset):

    def __init__(self, filePath, modelList, transform=None):
        self.filePath = filePath
        self.modelList = modelList
        self.transform = transform
        
        self.metaDataDf, self.phonesDf, self.SigMatFeatureUnits = self.getDataset()
        
        self.nFeatures = len(self.phonesDf.columns[3:].tolist())
        self.measFillValue = -1e4
        
        self.measurements, self.measurements_tvec = [torch.from_numpy(i) for i in self.df2SigMat(self.phonesDf)]
        # note that the last feature is the label (the target)
        self.mu = self.phonesDf.mean(axis=0)[3:-1].to_numpy()[:, None]
        stdsVec = self.phonesDf.std(axis=0)[3:-1].to_numpy()
        self.Sigma_minus_half = np.diag(1/stdsVec)
        self.Sigma_half = np.diag(stdsVec)
        
    def getDataset(self):
        dataDict = pickle.load(open(self.filePath, 'rb'))
        phonesDf, metaDataDf = dataDict['phonesDf'], dataDict['metaDataDf']        
        SigMatFeatureUnits = None
        
        metaDataDf = metaDataDf[metaDataDf['Classification'].isin(self.modelList)]
        phonesDf = phonesDf[phonesDf['Id'].isin(metaDataDf['Id'])]
        
        metaDataDf, phonesDf, SigMatFeatureUnits = self.convertBatch2Id(metaDataDf, phonesDf, SigMatFeatureUnits)
        
        return metaDataDf, phonesDf, SigMatFeatureUnits
    
    def convertBatch2Id(self, metaDataDf, phonesDf, SigMatFeatureUnits):
        
        phonesDf_noBatches = pd.DataFrame(columns=phonesDf.columns.tolist())
        
        lengthIdx = 2        
        metaDataColumns = metaDataDf.columns.tolist()
        metaDataDf_noBatches = pd.DataFrame(columns = metaDataColumns[:lengthIdx] + ['lengthOfSeries'] + metaDataColumns[lengthIdx:])                
                
        Ids_included = phonesDf['Id'].unique().tolist()
        running_Id = -1
        for Id in Ids_included:
            print(f'convertBatch2Id: starting Id {Id} out of {len(Ids_included)}')
            singleIdDf = phonesDf[phonesDf['Id'] == Id].copy()
            batches_included = singleIdDf['batch'].unique().tolist()
            
            singleId_metaDataDf_IdOrig = metaDataDf[metaDataDf['Id'] == Id].copy()            
            singleId_metaDataDf_IdOrig.insert(lengthIdx, 'lengthOfSeries', 0)
            singleId_metaDataDf_IdOrig = singleId_metaDataDf_IdOrig.reset_index(drop=True)
            
            for b, batch in enumerate(batches_included):
                singleIdSingleBatchDf = singleIdDf[singleIdDf['batch'] == batch].copy()
                
                singleIdSingleBatchDf.loc[:, 'time'] = singleIdSingleBatchDf['time'].to_numpy() - singleIdSingleBatchDf['time'].to_numpy()[0]
                
                maxGap = singleIdSingleBatchDf['time'].diff().abs().max()                
                assert singleIdSingleBatchDf['time'].diff().abs().max() < 0.0051, f'max gap is {maxGap}, Id={Id}, batch={batch}'
                
                singleId_metaDataDf = singleId_metaDataDf_IdOrig.copy()
                
                lengthOfSeries = singleIdSingleBatchDf.shape[0]
                singleId_metaDataDf.loc[0, 'lengthOfSeries'] = lengthOfSeries
                running_Id += 1
                singleId_metaDataDf.loc[0, 'Id'] = running_Id
                singleIdSingleBatchDf.loc[:, 'Id'] = running_Id
                singleIdSingleBatchDf.loc[:, 'batch'] = 0
                phonesDf_noBatches = pd.concat([phonesDf_noBatches, singleIdSingleBatchDf], ignore_index=True)
                metaDataDf_noBatches = pd.concat([metaDataDf_noBatches, singleId_metaDataDf.copy()], ignore_index=True)
        
        
        return metaDataDf_noBatches, phonesDf_noBatches, SigMatFeatureUnits
    
    def df2SigMat(self, df):
        # np.abs(self.patientsDf[self.patientsDf['Id'] == 3]['feature_1'].to_numpy() - measurementsMatrix[3, :, 1]).max()
        Ids_included = df['Id'].unique().tolist()
        maxLineage = -np.inf
        for Id in Ids_included:
            singleLineageDF = df[df['Id'] == Id]
            tVecLineage = singleLineageDF['time']  # the same for all patients
            if tVecLineage.shape[0] > maxLineage:
                maxLineage = tVecLineage.shape[0]
                tVec = tVecLineage
        nPatients = len(Ids_included)
        nTime = tVec.shape[0]
        measuremnts = np.zeros((nPatients, nTime, self.nFeatures), dtype='float32')
        for i, Id in enumerate(Ids_included):
            singleLineageDF = df[df['Id'] == Id]
            measuremnts[i] = np.concatenate((singleLineageDF.iloc[:, range(3, 3 + self.nFeatures)].to_numpy(dtype='float32'), self.measFillValue*np.ones((nTime - singleLineageDF.shape[0], self.nFeatures), dtype='float32')), axis=0)
        return [measuremnts, tVec.to_numpy(dtype='float32')[None, :].repeat(nPatients, axis=0)[:, :, None]]

        
    def __len__(self):
        return self.metaDataDf.shape[0]
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #sample = {'measurements': self.patientsDf.iloc[range(idx*self.nTime, (idx+1)*self.nTime), range(3, 3+self.nFeatures)].to_numpy(dtype='float32')}
        sample = {'measurements': self.measurements[idx], 'indices': idx, 'tVec': self.measurements_tvec[idx], 'lengthOfSeries': self.metaDataDf.iloc[idx, 2]}

        if self.transform:
            sample = self.transform(sample)

        return sample

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
# class definition
class RNN_Filter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, modelDict):
        super(RNN_Filter, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.fs = modelDict['fs']  # [hz]
        self.windowSize = 1  # [sec]
        self.windowSize_samples = int(np.round(self.windowSize*self.fs))
        self.windowSize_samples = self.windowSize_samples + (np.mod(self.windowSize_samples, 2) == 0)
        self.nConv_out_channels = 36
        
        self.trainOnNormalizedData = modelDict['trainOnNormalizedData']
        
        if modelDict['useSelectedFeatures']:
            # modelDict['statisticsDict']['mu'] includes the statistics of the time-axis at the last coordinate
            idx = modelDict['featuresIncludeInTrainIndices']
            self.means = nn.parameter.Parameter(torch.tensor(modelDict['statisticsDict']['mu'][idx], dtype=torch.float), requires_grad=False)
            Sigma_minus_half = np.diag(np.diag(modelDict['statisticsDict']['Sigma_minus_half'])[idx])
            self.Sigma_minus_half = nn.parameter.Parameter(torch.tensor(Sigma_minus_half, dtype=torch.float), requires_grad=False)            
        else:                
            self.means = nn.parameter.Parameter(torch.tensor(modelDict['statisticsDict']['mu'], dtype=torch.float), requires_grad=False)
            self.Sigma_minus_half = nn.parameter.Parameter(torch.tensor(modelDict['statisticsDict']['Sigma_minus_half'], dtype=torch.float), requires_grad=False)
            
        self.zeroPadd = nn.parameter.Parameter(torch.zeros(1, self.windowSize_samples-1, self.input_dim), requires_grad=False)
        self.zeroPadd2 = nn.parameter.Parameter(torch.zeros(1, 1, self.nConv_out_channels, self.windowSize_samples-1), requires_grad=False)
        
        self.conv2d_0 = nn.Conv2d(in_channels=1, out_channels=self.nConv_out_channels, kernel_size=(self.input_dim, self.windowSize_samples), padding='valid')
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=self.nConv_out_channels, kernel_size=(self.nConv_out_channels, self.windowSize_samples), padding='valid')
        self.conv2d_2 = nn.Conv2d(in_channels=1, out_channels=self.nConv_out_channels, kernel_size=(self.nConv_out_channels, self.windowSize_samples), padding='valid')
        self.conv2d_3 = nn.Conv2d(in_channels=1, out_channels=self.nConv_out_channels, kernel_size=(self.nConv_out_channels, self.windowSize_samples), padding='valid')
        self.activation = nn.ReLU()

        # setup RNN layer        
        self.Filter_rnn = nn.LSTM(self.nConv_out_channels, self.hidden_dim, self.num_layers, batch_first=True)

        # setup output layer
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, z):
        #
        nPatients, nTime, nFeatures = z.shape
        if self.trainOnNormalizedData:
            means = self.means[None, None, :, :].expand(nPatients, nTime, -1, -1)
            Sigma_minus_half = self.Sigma_minus_half[None, None, :, :].expand(nPatients, nTime, -1, -1)
            tilde_z = z[:, :, :, None]            
            normalized_tilde_z = torch.matmul(Sigma_minus_half, tilde_z - means)[:, :, :, 0]
        
        # the zero padding ensures causal output
        zeroPadd = self.zeroPadd.expand(nPatients, -1, -1)
        normalized_tilde_z_padded = torch.cat((zeroPadd, normalized_tilde_z), axis=1)
        
        zeroPadd = self.zeroPadd2.expand(nPatients, -1, -1, -1)
        conv_0 = self.activation(self.conv2d_0(normalized_tilde_z_padded.transpose(2, 1).unsqueeze(1)))
        conv_1 = self.activation(self.conv2d_1(torch.cat((zeroPadd, conv_0.transpose(1,2)), axis=3)))
        conv_2 = self.activation(self.conv2d_2(torch.cat((zeroPadd, conv_1.transpose(1,2)), axis=3)))
        conv_3 = self.activation(self.conv2d_3(torch.cat((zeroPadd, conv_2.transpose(1,2)), axis=3)))
        
        conv_out = conv_1.squeeze(2).transpose(2, 1)
        
        self.Filter_rnn.flatten_parameters() 
        controlHiddenDim, hidden = self.Filter_rnn(conv_out)
        # controlHiddenDim of shape like normalized_tilde_z
        hat_x_k_plus_1_given_k = self.logSoftmax(self.linear(controlHiddenDim))

        return hat_x_k_plus_1_given_k

def augCrop(measurements, measurements_tvec, lengthOfSeries, nMinimalSamples, nMaximalSamples):    
    nSeries, nTime, nFeature = measurements.shape
    for s in range(nSeries):
        if lengthOfSeries[s] < nTime:
            fillValue = measurements[s, -1, 0].item()
            break
    
    nMaximalSamples = int(np.min([nMaximalSamples,lengthOfSeries.max().item()]))
    
    measurementsAug, measurements_tvecAug, lengthOfSeriesAug = fillValue*torch.ones_like(measurements[:, :nMaximalSamples]), torch.ones_like(measurements_tvec[:, :nMaximalSamples]), torch.zeros_like(lengthOfSeries)
    for s in range(nSeries):
        if lengthOfSeries[s] > nMinimalSamples:
            startIdx = torch.randint(int((lengthOfSeries[s]-nMinimalSamples).item()), (1,))[0]    
            stopIdx = torch.min(lengthOfSeries[s] - 1, startIdx+nMaximalSamples-1)
            lengthOfSeriesAug[s] = stopIdx - startIdx + 1
            measurementsAug[s, :lengthOfSeriesAug[s]] = measurements[s, startIdx:stopIdx+1]
            measurements_tvecAug[s, :lengthOfSeriesAug[s]] = measurements_tvec[s, :lengthOfSeriesAug[s]]
        else:
            measurementsAug[s], measurements_tvecAug[s], lengthOfSeriesAug[s] = measurements[s, :nMaximalSamples], measurements_tvec[s, :nMaximalSamples], torch.clamp(lengthOfSeries[s], max=nMaximalSamples)
            
    
    return measurementsAug, measurements_tvecAug, lengthOfSeriesAug
    
def trainModel(stateEst_ANN, trainLoader, validationLoader, patientsDataset, enableDataParallel, modelDict, enablePlots, mode):    
    lowThrLr = 1e-5
    nClasses = stateEst_ANN.output_dim
    if mode == 'train':
        nValidationEpochs = 10  # for averaging the results over many augmentations
        nMinimalSamples = np.floor(10*modelDict['fs'])  # equal to 10 seconds at the sample rate 
        nMaximalSamples = np.floor(30*modelDict['fs'])
    elif mode == 'test':
        nValidationEpochs = 5#30  # for averaging the results over many augmentations
        nMinimalSamples = np.floor(10*modelDict['fs'])  # equal to 10 seconds at the sample rate 
        nMaximalSamples = np.floor(30*modelDict['fs'])
    
    
    # training method:    
    if mode == 'train':
        lr = 0.001
        optimizer = optim.Adam(stateEst_ANN.parameters(), lr=lr, weight_decay=0)
        patience = 20
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.1*2, patience=patience, threshold=1e-6)
        
    criterion = nn.NLLLoss(ignore_index=0)  # ignoring the noClass category
    
    # moving model to cuda:    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    if enableDataParallel and torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        stateEst_ANN = nn.DataParallel(stateEst_ANN)    

    stateEst_ANN.to(device)
    criterion.to(device)    
    #model.apply(init_weights)

    # training and saving the model when validation is best:
    if mode == 'train': print('start training')
    min_valid_loss, previousValidationMeanStd, previousValidationMeanCorrectFraction, previousValidationLikelihoodFraction, previousValidationLoss = np.inf, np.inf, np.inf, np.inf, np.inf
    epoch = -1
    minimalSeriesLength = 0
        
    trainLossList, validationLossList, trainCorrectFractionist, trainLikelihoodList, validationCorrectFractionist, validationLikelihoodList, trainMeanStdList, validationMeanStdList, validationPearsonList, trainPearsonList = list(), list(), list(), list(), list(), list(), list(), list(), list(), list()
    
    while True:
        epoch += 1
        train_loss, train_correct, train_correct_nSamples, train_likelihood = 0.0, np.zeros(nClasses+1), np.zeros(nClasses+1), np.zeros(nClasses+1)
        if mode == 'train' and epoch > patience and np.mod(epoch, 10) == 0: print(f'training: starting epoch {epoch}; lr = {scheduler._last_lr[-1]}')
        
        if mode == 'train':
            stateEst_ANN.train()
        elif mode == 'test':
            stateEst_ANN.eval()

        for i_batch, sample_batched in enumerate(trainLoader):
            # print(f'starting epoch {epoch}, batch {i_batch}')
            if mode == 'train': optimizer.zero_grad()

            lengthOfSeries = sample_batched["lengthOfSeries"]
            validSeries = lengthOfSeries >= minimalSeriesLength
            if not(validSeries.any()): continue
            lengthOfSeries = lengthOfSeries[validSeries]
            maxLengthOfSeries = lengthOfSeries.max()
            measurements = sample_batched["measurements"][:, :maxLengthOfSeries][validSeries]  # [nBatch, nTime, nFeature]            
            measurements_tvec = sample_batched["tVec"][:, :maxLengthOfSeries][validSeries]
            
            #  crop measurements for augmentations purpose:
            if mode == 'train':
                measurements, measurements_tvec, lengthOfSeries = augCrop(measurements, measurements_tvec, lengthOfSeries, nMinimalSamples, nMaximalSamples)
            
            labels = measurements[:, :, -1][:, :, None].type(torch.int64)
            filteringLabels = labels[:, 1:]
            measurements = measurements[:, :, :-1]
            currentBatchSize = measurements.shape[0]
            
            if modelDict['useSelectedFeatures']: measurements = measurements[:, :, modelDict['featuresIncludeInTrainIndices']]                                                

            measurements = measurements.to(device)            
            lengthOfSeries = lengthOfSeries.to(device)
            filteringLabels = filteringLabels.to(device)
            # measurements, funcOfMeas of shape [batchSize, nTime, nFeatures]

            hat_x_k_plus_1_given_k = stateEst_ANN(measurements)
            hat_x_k_plus_1_given_k = hat_x_k_plus_1_given_k[:, :-1] # filtering
            nPatients, nTime, _ = hat_x_k_plus_1_given_k.shape
            # now hat_x_k_plus_1_given_k at [:,k] has the estimation of the state at time [k+1] given measurements up to and including time k
            # filteringLabels at [:k] has the label of time [k+1]
            
            hat_x_k_plus_1_given_k_flatten = torch.flatten(hat_x_k_plus_1_given_k, start_dim=0, end_dim=-2)
            filteringLabels_flatten = torch.flatten(filteringLabels, start_dim=0, end_dim=-2)
            # torch.flatten(hat_x_k_plus_1_given_k, start_dim=0, end_dim=-2)[0] - hat_x_k_plus_1_given_k[0,0]
            # torch.flatten(hat_x_k_plus_1_given_k, start_dim=0, end_dim=-2)[1] - hat_x_k_plus_1_given_k[0,1]
            # torch.flatten(hat_x_k_plus_1_given_k, start_dim=0, end_dim=-2)[hat_x_k_plus_1_given_k.shape[1]] - hat_x_k_plus_1_given_k[1,0]
            validIndices = torch.zeros_like(hat_x_k_plus_1_given_k_flatten[:, 0]).bool()
            for s in range(lengthOfSeries.shape[0]):
                seriesLength = lengthOfSeries[s] - 1
                seriesStartIdx = nTime*s
                seriesStopIdx = seriesStartIdx + seriesLength
                validIndices[seriesStartIdx : seriesStopIdx] = True
                                
            loss = criterion(hat_x_k_plus_1_given_k_flatten[validIndices], filteringLabels_flatten[validIndices, 0])            
     
            loss = measurements.shape[0]/len(trainLoader.dataset) * loss
            
            if mode == 'train':                        
                loss.backward()
                optimizer.step()  # parameter update

            train_loss += loss.item()  
            
            correctClassification, likelihoods, correctClassification_nSamples = np.zeros(nClasses+1), np.zeros(nClasses+1), np.zeros(nClasses+1)
            estimations, labels = hat_x_k_plus_1_given_k_flatten[validIndices], filteringLabels_flatten[validIndices, 0]
            for c in range(nClasses):
                labelIndices = labels == c
                correctClassification_nSamples[c] = labelIndices.sum().item()
                if labelIndices.sum() > 0:
                    correctClassification[c] = ((torch.argmax(estimations[labelIndices], dim=1) == labels[labelIndices]).sum()).item()                    
                    likelihoods[c] = np.exp(estimations[labelIndices, c].detach().cpu().numpy()).sum()
                else:
                    correctClassification[c], likelihoods[c] = 0, 0
            classesIndices = labels > 0    # 0 is the no class that the filter is not trained on
            correctClassification[-1] = ((torch.argmax(estimations[classesIndices], dim=1) == labels[classesIndices]).sum()).item()   
            allEstimations, allLabels = estimations[classesIndices].detach().cpu().numpy(), labels[classesIndices].detach().cpu().numpy()
            for i in range(allEstimations.shape[0]):
                likelihoods[-1] = likelihoods[-1] + np.exp(allEstimations[i, allLabels[i]])
            correctClassification_nSamples[-1] = classesIndices.sum().item()
            train_correct = train_correct + correctClassification
            train_likelihood = train_likelihood + likelihoods
            train_correct_nSamples = train_correct_nSamples + correctClassification_nSamples

        #scheduler.step(train_loss)
        trainLossList.append(train_loss)
        trainCorrectFraction = np.divide(train_correct, train_correct_nSamples)
        trainLikelihoodFractions = np.divide(train_likelihood, train_correct_nSamples)
        trainCorrectFractionist.append(trainCorrectFraction)
        trainLikelihoodList.append(trainLikelihoodFractions)
        # print("Outside: measurements size", measurements.size(), "combination size", combination.size())

        validation_loss, validation_correctFraction, validation_likelihoodFraction = 0.0, 0.0, 0.0
        stateEst_ANN.eval()
        for validationEpoch in range(nValidationEpochs):
            if mode == 'test': print(f'startin validation epoch {validationEpoch} out of {nValidationEpochs}')
            validation_loss_singleEpoch, validation_correctFraction_singleEpoch, validation_likelihood_singleEpoch, validation_correctFraction_singleEpoch_nSamples = 0.0, 0.0, 0.0, 0.0
            for i_batch, sample_batched in enumerate(validationLoader):
                lengthOfSeries = sample_batched["lengthOfSeries"]
                validSeries = lengthOfSeries >= minimalSeriesLength
                if not(validSeries.any()): continue
                lengthOfSeries = lengthOfSeries[validSeries]
                maxLengthOfSeries = lengthOfSeries.max()
                measurements = sample_batched["measurements"][:, :maxLengthOfSeries][validSeries]  # [nBatch, nTime, nFeature]
                measurements_tvec = sample_batched["tVec"][:, :maxLengthOfSeries][validSeries]
                
                #  crop measurements for augmentations purpose:                
                measurements, measurements_tvec, lengthOfSeries = augCrop(measurements, measurements_tvec, lengthOfSeries, nMinimalSamples, nMaximalSamples)
                    
                labels = measurements[:, :, -1][:, :, None].type(torch.int64)
                filteringLabels = labels[:, 1:]
                measurements = measurements[:, :, :-1]
                currentBatchSize = measurements.shape[0]
                
                if modelDict['useSelectedFeatures']: measurements = measurements[:, :, modelDict['featuresIncludeInTrainIndices']]                                                

                measurements = measurements.to(device)            
                lengthOfSeries = lengthOfSeries.to(device)
                filteringLabels = filteringLabels.to(device)
                # measurements, funcOfMeas of shape [batchSize, nTime, nFeatures]

                hat_x_k_plus_1_given_k = stateEst_ANN(measurements)
                hat_x_k_plus_1_given_k = hat_x_k_plus_1_given_k[:, :-1] # filtering
                nPatients, nTime, _ = hat_x_k_plus_1_given_k.shape
                # now hat_x_k_plus_1_given_k at [:,k] has the estimation of the state at time [k+1] given measurements up to and including time k
                # filteringLabels at [:k] has the label of time [k+1]
                
                hat_x_k_plus_1_given_k_flatten = torch.flatten(hat_x_k_plus_1_given_k, start_dim=0, end_dim=-2)
                filteringLabels_flatten = torch.flatten(filteringLabels, start_dim=0, end_dim=-2)
                # torch.flatten(hat_x_k_plus_1_given_k, start_dim=0, end_dim=-2)[0] - hat_x_k_plus_1_given_k[0,0]
                # torch.flatten(hat_x_k_plus_1_given_k, start_dim=0, end_dim=-2)[1] - hat_x_k_plus_1_given_k[0,1]
                # torch.flatten(hat_x_k_plus_1_given_k, start_dim=0, end_dim=-2)[hat_x_k_plus_1_given_k.shape[1]] - hat_x_k_plus_1_given_k[1,0]
                validIndices = torch.zeros_like(hat_x_k_plus_1_given_k_flatten[:, 0]).bool()
                for s in range(lengthOfSeries.shape[0]):
                    seriesLength = lengthOfSeries[s] - 1
                    seriesStartIdx = nTime*s
                    seriesStopIdx = seriesStartIdx + seriesLength
                    validIndices[seriesStartIdx : seriesStopIdx] = True
                                    
                loss = criterion(hat_x_k_plus_1_given_k_flatten[validIndices], filteringLabels_flatten[validIndices, 0])            
         
                loss = measurements.shape[0]/len(trainLoader.dataset) * loss
                
                validation_loss_singleEpoch += loss.item()
                
                correctClassification, correctClassification_nSamples, likelihoods = np.zeros(nClasses+1), np.zeros(nClasses+1), np.zeros(nClasses+1)
                estimations, labels = hat_x_k_plus_1_given_k_flatten[validIndices], filteringLabels_flatten[validIndices, 0]
                for c in range(nClasses):
                    labelIndices = labels == c
                    correctClassification_nSamples[c] = labelIndices.sum().item()
                    if labelIndices.sum() > 0:
                        correctClassification[c] = ((torch.argmax(estimations[labelIndices], dim=1) == labels[labelIndices]).sum()).item()                    
                        likelihoods[c] = np.exp(estimations[labelIndices, c].detach().cpu().numpy()).sum()
                    else:
                        correctClassification[c] = 0
                classesIndices = labels > 0    # 0 is the no class that the filter is not trained on
                correctClassification[-1] = ((torch.argmax(estimations[classesIndices], dim=1) == labels[classesIndices]).sum()).item()   
                allEstimations, allLabels = estimations[classesIndices].detach().cpu().numpy(), labels[classesIndices].detach().cpu().numpy()
                for i in range(allEstimations.shape[0]):
                    likelihoods[-1] = likelihoods[-1] + np.exp(allEstimations[i, allLabels[i]])
                correctClassification_nSamples[-1] = classesIndices.sum().item()
                validation_correctFraction_singleEpoch = validation_correctFraction_singleEpoch + correctClassification
                validation_likelihood_singleEpoch = validation_likelihood_singleEpoch + likelihoods
                validation_correctFraction_singleEpoch_nSamples = validation_correctFraction_singleEpoch_nSamples + correctClassification_nSamples                                                                        
            
            validation_loss += validation_loss_singleEpoch/nValidationEpochs
            validation_correctFraction = validation_correctFraction + np.divide(validation_correctFraction_singleEpoch, validation_correctFraction_singleEpoch_nSamples)/nValidationEpochs
            validation_likelihoodFraction = validation_likelihoodFraction + np.divide(validation_likelihood_singleEpoch, validation_correctFraction_singleEpoch_nSamples)/nValidationEpochs
        
        if mode == 'train' and epoch > patience: scheduler.step(train_loss)
        # print(f'epoch: {epoch}; Validation Mean error w.r.t mean(abs(labels)) = {valid_loss / len(validationLoader) / combination_mean}; mean error in theta w.r.t mean(theta)= {meanErrorIn_theta}')
        validationLossList.append(validation_loss)
        validationCorrectFractionist.append(validation_correctFraction)
        validationLikelihoodList.append(validation_likelihoodFraction)
        #currentValidationMeanStd = patientsDataset.calcMeanStd(validationData.indices, combination_ANN)
        #currentTrainMeanStd = patientsDataset.calcMeanStd(trainData.indices, combination_ANN)
        #validationMeanStdList.append(currentValidationMeanStd)
        #trainMeanStdList.append(currentTrainMeanStd)

        if previousValidationLoss > validation_loss:
            #print(f'reference Std value = {referenceStdValue}')
            #print(f'epoch {epoch}, Validation Mean Std Decreased({previousValidationMeanStd:.6f}--->{currentValidationMeanStd:.6f})')
            print(f'epoch {epoch}, Validation loss Decreased({previousValidationLoss:.6f}--->{validation_loss:.6f})')
            print(f'epoch {epoch}, Validation correct fraction ({np.round(100*previousValidationMeanCorrectFraction)}--->{np.round(100*validation_correctFraction)})')
            print(f'epoch {epoch}, Validation likelihood ({np.round(100*previousValidationLikelihoodFraction)}--->{np.round(100*validation_likelihoodFraction)})')
            #print(f'epoch {epoch}, Train Mean Std {currentTrainMeanStd:.6f}')
            print(f'epoch {epoch}, Train loss {train_loss:.6f}')
            print(f'epoch {epoch}, Train correct fraction {np.round(100*trainCorrectFraction)}')
            print(f'epoch {epoch}, Train likelihood {np.round(100*trainLikelihoodFractions)}')
            
            #previousValidationMeanStd = currentValidationMeanStd
        if mode == 'train' and previousValidationLoss > validation_loss and epoch > patience:
            previousValidationLoss = validation_loss
            previousValidationMeanCorrectFraction = validation_correctFraction
            previousValidationLikelihoodFraction = validation_likelihoodFraction
            if enableDataParallel:
                model_state_dict = stateEst_ANN.module.state_dict().copy()
                #torch.save(combination_ANN.module.state_dict(), fileName + '_model.pt')
            else:
                model_state_dict = stateEst_ANN.state_dict().copy()
                #torch.save(combination_ANN.state_dict(), fileName + '_model.pt')
        '''    
        else:
            print(f'reference Std value = {referenceStdValue}')
            print(f'epoch {epoch}, Validation Mean Std ({previousValidationMeanStd:.6f}--->{currentValidationMeanStd:.6f})')
            print(f'epoch {epoch}, Validation loss Increased({min_valid_loss:.6f}--->{validation_loss:.6f})')
            print(f'epoch {epoch}, Train Mean Std {currentTrainMeanStd:.6f}')
            print(f'epoch {epoch}, Train loss {train_loss:.6f}')
        '''

        
        if mode == 'train' and epoch > patience and scheduler._last_lr[-1] < lowThrLr:
            print(f'Stoping optimization due to learning rate of {scheduler._last_lr[-1]}')
            break  
        if mode == 'test': 
            if enableDataParallel:
                model_state_dict = stateEst_ANN.module.state_dict().copy()
                #torch.save(combination_ANN.module.state_dict(), fileName + '_model.pt')
            else:
                model_state_dict = stateEst_ANN.state_dict().copy()
            previousValidationLoss = np.inf
            break
        #if epoch > 9: break

    epochVec = np.arange(0, len(validationCorrectFractionist))    
    if epochVec.shape[0] > 1:
        if enablePlots:
            plt.subplot(1,2,1)
            for c in range(nClasses):
                tList = [trainCorrectFractionist[i][c] for i in range(1,len(trainCorrectFractionist))]
                #vList = [validationCorrectFractionist[i][c] for i in range(1,len(validationCorrectFractionist))]
                plt.plot(epochVec[1:], tList, '--', label=f'class {c}')
                #plt.plot(epochVec[1:], vList, '--', label=f'validation class {c}')
            
            tList = [trainCorrectFractionist[i][-1] for i in range(1,len(trainCorrectFractionist))]
            #vList = [validationCorrectFractionist[i][-1] for i in range(1,len(validationCorrectFractionist))]
            plt.plot(epochVec[1:], tList, label=f'all')
            #plt.plot(epochVec[1:], vList, '--', label=f'all')
            
            plt.xlabel('epoch')
            #plt.yscale('log')
            plt.legend()
            plt.title('train correct fraction')
            plt.grid()
            
            plt.subplot(1,2,2)
            for c in range(nClasses):
                #tList = [trainCorrectFractionist[i][c] for i in range(1,len(trainCorrectFractionist))]
                vList = [validationCorrectFractionist[i][c] for i in range(1,len(validationCorrectFractionist))]
                #plt.plot(epochVec[1:], tList, '--', label=f'train class {c}')
                plt.plot(epochVec[1:], vList, '--', label=f'class {c}')
            
            #tList = [trainCorrectFractionist[i][-1] for i in range(1,len(trainCorrectFractionist))]
            vList = [validationCorrectFractionist[i][-1] for i in range(1,len(validationCorrectFractionist))]
            #plt.plot(epochVec[1:], tList, '--', label=f'all')
            plt.plot(epochVec[1:], vList, label=f'all')
            
            plt.xlabel('epoch')
            #plt.yscale('log')
            plt.legend()
            plt.title('validation correct fraction')
            plt.grid()
            plt.show()
        
    return model_state_dict, previousValidationLoss