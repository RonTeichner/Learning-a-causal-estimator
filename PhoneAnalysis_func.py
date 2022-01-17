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
import copy

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
        self.mu = self.phonesDf[self.phonesDf['gt'] > 0].mean(axis=0)[3:-1].to_numpy()[:, None]
        stdsVec = self.phonesDf[self.phonesDf['gt'] > 0].std(axis=0)[3:-1].to_numpy()
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

        
    def plotTimesSeries(self):
        model = self.metaDataDf['Classification'].unique()[0]
        devices = self.metaDataDf['Device'].unique().tolist()
        minClass, maxClass = self.phonesDf['gt'].min()-1, self.phonesDf['gt'].max()+1
        for device in devices:
            singleDeviceMetaData = self.metaDataDf[self.metaDataDf['Device'] == device]
            Ids_included = singleDeviceMetaData['Id'].unique().tolist()
            for Id in Ids_included:
                singleLineageDF = self.phonesDf[self.phonesDf['Id'] == Id]
                singleLineageMetaDataDf = self.metaDataDf[self.metaDataDf['Id'] == Id]
                user, device = singleLineageMetaDataDf['User'].to_numpy()[0], singleLineageMetaDataDf['Device'].to_numpy()[0]
                tVecLineage = singleLineageDF['time']
                plt.figure(figsize=(10,10))
                plt.suptitle(f'{model}, Id = {Id}, User = {user}, Device = {device}')
                plt.subplot(3, 1, 1)
                plt.plot(tVecLineage, singleLineageDF['acc_x'], label='acc_x')
                plt.plot(tVecLineage, singleLineageDF['acc_y'], label='acc_y')
                plt.plot(tVecLineage, singleLineageDF['acc_z'], label='acc_z')
                plt.xlabel('sec')
                plt.grid()
                plt.legend()
                plt.subplot(3, 1, 2)
                plt.plot(tVecLineage, singleLineageDF['gyr_x'], label='gyr_x')
                plt.plot(tVecLineage, singleLineageDF['gyr_y'], label='gyr_y')
                plt.plot(tVecLineage, singleLineageDF['gyr_z'], label='gyr_z')
                plt.xlabel('sec')
                plt.grid()
                plt.legend()
                plt.subplot(3, 1, 3)
                plt.plot(tVecLineage, singleLineageDF['gt'])
                plt.ylim([minClass, maxClass])
                plt.xlabel('sec')
                plt.grid()
                plt.show()
        
    
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
        if modelDict['useSelectedFeatures']:
            self.input_dim = len(modelDict['featuresIncludeInTrainIndices'])
        else:
            self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.fs = modelDict['fs']  # [hz]
        
        if not('enableSparse' in modelDict.keys()): modelDict['enableSparse'] = False
        
        self.enableSparse = modelDict['enableSparse']
        if self.enableSparse:
            self.windowSize = 4  # [sec]
        else:
            self.windowSize = 4  # [sec]
        self.windowSize_samples = int(np.round(self.windowSize*self.fs))
        self.windowSize_samples = self.windowSize_samples + (np.mod(self.windowSize_samples, 2) == 0)
        if self.enableSparse:
            self.overlap = 0.25/4
            self.strideHorizontal = int(np.round(self.overlap*self.windowSize_samples))
        self.nConv_out_channels = 12
        
        if 'smoother' in modelDict.keys():
            self.bidirectional = modelDict['smoother']
        else:
            self.bidirectional = False
        
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
        if self.enableSparse: 
            self.downSampleWindowSize = int(np.round(self.windowSize_samples/self.strideHorizontal))
            self.zeroPadd2 = nn.parameter.Parameter(torch.zeros(1, 1, self.nConv_out_channels, self.downSampleWindowSize-1), requires_grad=False)
        else:
            self.zeroPadd2 = nn.parameter.Parameter(torch.zeros(1, 1, self.nConv_out_channels, self.windowSize_samples-1), requires_grad=False)
        
        if self.enableSparse:
            self.conv2d_0 = nn.Conv2d(in_channels=1, out_channels=self.nConv_out_channels, kernel_size=(self.input_dim, self.windowSize_samples), stride=(1, self.strideHorizontal), padding='valid')
            self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=self.nConv_out_channels, kernel_size=(self.nConv_out_channels, self.downSampleWindowSize), padding='valid')
            self.conv2d_2 = nn.Conv2d(in_channels=1, out_channels=self.nConv_out_channels, kernel_size=(self.nConv_out_channels, self.downSampleWindowSize), padding='valid')
            self.conv2d_3 = nn.Conv2d(in_channels=1, out_channels=self.nConv_out_channels, kernel_size=(self.nConv_out_channels, self.downSampleWindowSize), padding='valid')
        else:
            self.conv2d_0 = nn.Conv2d(in_channels=1, out_channels=self.nConv_out_channels, kernel_size=(self.input_dim, self.windowSize_samples), padding='valid')
            self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=self.nConv_out_channels, kernel_size=(self.nConv_out_channels, self.windowSize_samples), padding='valid')
            self.conv2d_2 = nn.Conv2d(in_channels=1, out_channels=self.nConv_out_channels, kernel_size=(self.nConv_out_channels, self.windowSize_samples), padding='valid')
            self.conv2d_3 = nn.Conv2d(in_channels=1, out_channels=self.nConv_out_channels, kernel_size=(self.nConv_out_channels, self.windowSize_samples), padding='valid')
        
        self.activation = nn.ReLU()

        # setup RNN layer        
        self.Filter_rnn = nn.LSTM(input_size=self.nConv_out_channels, hidden_size=self.hidden_dim, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True)

        # setup output layer
        #self.linear = nn.Linear(self.hidden_dim*(1+self.bidirectional), self.output_dim)
        self.linearFilter = nn.Linear(self.hidden_dim, self.output_dim)
        if self.bidirectional:
            self.linearSmoother = nn.Linear(self.hidden_dim*(1+self.bidirectional), self.output_dim)
            
        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, z):
        #
        nPatients, nTime, nFeatures = z.shape
        if self.trainOnNormalizedData:
            means = self.means[None, None, :, :].expand(nPatients, nTime, -1, -1)
            Sigma_minus_half = self.Sigma_minus_half[None, None, :, :].expand(nPatients, nTime, -1, -1)
            tilde_z = z[:, :, :, None]            
            normalized_tilde_z = torch.matmul(Sigma_minus_half, tilde_z - means)[:, :, :, 0]
        else:
            normalized_tilde_z = z
        
        # the zero padding ensures causal output
        zeroPadd = self.zeroPadd.expand(nPatients, -1, -1)
        normalized_tilde_z_padded = torch.cat((zeroPadd, normalized_tilde_z), axis=1)
        
        zeroPadd = self.zeroPadd2.expand(nPatients, -1, -1, -1)
        conv_0 = self.activation(self.conv2d_0(normalized_tilde_z_padded.transpose(2, 1).unsqueeze(1)))
        conv_1 = self.activation(self.conv2d_1(torch.cat((zeroPadd, conv_0.transpose(1,2)), axis=3)))
        conv_2 = self.activation(self.conv2d_2(torch.cat((zeroPadd, conv_1.transpose(1,2)), axis=3)))
        conv_3 = self.activation(self.conv2d_3(torch.cat((zeroPadd, conv_2.transpose(1,2)), axis=3)))
        
        if self.enableSparse:
            conv_out = conv_3.squeeze(2).transpose(2, 1)
        else:
            conv_out = conv_1.squeeze(2).transpose(2, 1)
        
        self.Filter_rnn.flatten_parameters() 
        controlHiddenDim, hidden = self.Filter_rnn(conv_out)
        # controlHiddenDim of shape like normalized_tilde_z
        hat_x_k_plus_1_given_k = self.logSoftmax(self.linearFilter(controlHiddenDim[:, :, :self.hidden_dim]))
        if self.bidirectional:
            hat_x_k_plus_1_given_N_minus_1 = self.logSoftmax(self.linearSmoother(controlHiddenDim))
        else:
            hat_x_k_plus_1_given_N_minus_1 = None        
        
        if self.enableSparse:
            hat_x_k_plus_1_given_k = torch.repeat_interleave(hat_x_k_plus_1_given_k, self.strideHorizontal, dim=1)
            if self.bidirectional:
                hat_x_k_plus_1_given_N_minus_1 = torch.repeat_interleave(hat_x_k_plus_1_given_N_minus_1, self.strideHorizontal, dim=1)
                

        return hat_x_k_plus_1_given_k, hat_x_k_plus_1_given_N_minus_1

def augCrop(measurements, measurements_tvec, lengthOfSeries, nMinimalSamples, nMaximalSamples, fillValue):    
    nSeries, nTime, nFeature = measurements.shape    
    
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
    
def trainModel(stateEst_ANN, trainLoader, validationLoader, patientsDataset, enableDataParallel, modelDict, enablePlots, mode, trainOnSmoother=False, smoother_rnn=None):    
    lowThrLr = 1e-5
    nClasses = stateEst_ANN.output_dim
    if not('enableSparse' in modelDict.keys()): modelDict['enableSparse'] = False
    if modelDict['enableSparse']: 
        strideHorizontal = stateEst_ANN.strideHorizontal
    else:
        strideHorizontal = 1
    fillValue = -1e-4
    if mode == 'train':
        nValidationEpochs = 10  # for averaging the results over many augmentations
        nMinimalSamples = np.floor(10*modelDict['fs'])  # equal to 10 seconds at the sample rate 
        nMaximalSamples = np.floor(30*modelDict['fs'])
    elif mode == 'test':
        nValidationEpochs = 10  # for averaging the results over many augmentations
        nMinimalSamples = np.floor(10*modelDict['fs'])  # equal to 10 seconds at the sample rate 
        nMaximalSamples = np.floor(30*modelDict['fs'])
    
    if trainOnSmoother: smoother_rnn.eval()
    # training method:    
    if mode == 'train':
        lr = 0.001
        optimizer = optim.Adam(stateEst_ANN.parameters(), lr=lr, weight_decay=0)
        patience = 60
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.1*2, patience=patience, threshold=1e-6)
        
    weights = 1/modelDict['statisticsDict']['classDistribution'][1]
    weights = torch.tensor(weights/weights.sum(), dtype=torch.float)
    criterion = nn.NLLLoss(weight=weights, ignore_index=0)  # ignoring the noClass category
    
    # moving model to cuda:    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    if enableDataParallel and torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        stateEst_ANN = nn.DataParallel(stateEst_ANN)    
        if trainOnSmoother: smoother_rnn = nn.DataParallel(smoother_rnn)    

    stateEst_ANN.to(device)
    if trainOnSmoother: smoother_rnn.to(device)
    criterion.to(device)    
    #model.apply(init_weights)

    # training and saving the model when validation is best:
    if mode == 'train': print('start training')
    min_valid_loss, previousValidationMeanStd, previousValidationFilterMeanCorrectFraction, previousValidationSmootherMeanCorrectFraction, previousValidationLikelihoodFilterFraction, previousValidationLikelihoodSmootherFraction, previousValidationLoss = np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf
    epoch = -1
    minimalSeriesLength = 0
        
    trainLossList, validationLossList, trainFilterCorrectFractionist, trainFilterLikelihoodList, validationCorrectFilterFractionist, validationFilterLikelihoodList, trainMeanStdList, validationMeanStdList, validationPearsonList, trainPearsonList = list(), list(), list(), list(), list(), list(), list(), list(), list(), list()
    trainSmootherCorrectFractionist, trainSmootherLikelihoodList, validationCorrectSmootherFractionist, validationSmootherLikelihoodList = list(), list(), list(), list()
    
    while True:
        epoch += 1
        train_loss, train_Filtercorrect, train_FilterCorrect_nSamples, train_FilterLikelihood, train_Smoothercorrect, train_SmootherCorrect_nSamples, train_SmootherLikelihood = 0.0, np.zeros(nClasses+1), np.zeros(nClasses+1), np.zeros(nClasses+1), np.zeros(nClasses+1), np.zeros(nClasses+1), np.zeros(nClasses+1)
        if mode == 'train' and epoch > patience and np.mod(epoch, 10) == 0: print(f'training: starting epoch {epoch}')
        
        if mode == 'train':
            stateEst_ANN.train()
        elif mode == 'test':
            stateEst_ANN.eval()
        
        if mode == 'train':
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
                measurements, measurements_tvec, lengthOfSeries = augCrop(measurements, measurements_tvec, lengthOfSeries, nMinimalSamples, int(np.min([int(maxLengthOfSeries/strideHorizontal)*strideHorizontal, nMaximalSamples])), fillValue)
                
                #if not(trainOnSmoother):
                labels = measurements[:, :, -1][:, :, None].type(torch.int64)
                filteringLabels = labels[:, 1:]
                if trainOnSmoother:
                    DefinedClassIndices = filteringLabels > 0
                    DefinedClassIndices = DefinedClassIndices.to(device)
                    
                measurements = measurements[:, :, :-1]
                currentBatchSize = measurements.shape[0]
                
                if modelDict['useSelectedFeatures']: measurements = measurements[:, :, modelDict['featuresIncludeInTrainIndices']]      
                
                measurements = measurements.to(device)            
                lengthOfSeries = lengthOfSeries.to(device)
                
                if trainOnSmoother:
                    hat_x_k_plus_1_given_N = smoother_rnn(measurements)          
                    # now hat_x_k_plus_1_given_N at [:,k] has the estimation of the state at time [k+1] given measurements up to and including time N-1                               
                    hat_x_k_plus_1_given_N = hat_x_k_plus_1_given_N[:, :-1]
                    filteringLabels = torch.argmax(hat_x_k_plus_1_given_N, dim=2).unsqueeze(2).detach()
                else:
                    filteringLabels = filteringLabels.to(device)
                # measurements, funcOfMeas of shape [batchSize, nTime, nFeatures]
    
                hat_x_k_plus_1_given_k, hat_x_k_plus_1_given_N_minus_1 = stateEst_ANN(measurements)
                hat_x_k_plus_1_given_k, hat_x_k_plus_1_given_N_minus_1 = hat_x_k_plus_1_given_k[:, :-1], hat_x_k_plus_1_given_N_minus_1[:, :-1] # filtering
                nPatients, nTime, _ = hat_x_k_plus_1_given_k.shape
                # now hat_x_k_plus_1_given_k at [:,k] has the estimation of the state at time [k+1] given measurements up to and including time k
                # filteringLabels at [:k] has the label of time [k+1]
                
                hat_x_k_plus_1_given_k_flatten = torch.flatten(hat_x_k_plus_1_given_k, start_dim=0, end_dim=-2)
                hat_x_k_plus_1_given_N_minus_1_flatten = torch.flatten(hat_x_k_plus_1_given_N_minus_1, start_dim=0, end_dim=-2)
                filteringLabels_flatten = torch.flatten(filteringLabels, start_dim=0, end_dim=-2)
                if trainOnSmoother: DefinedClassIndices_flatten = torch.flatten(DefinedClassIndices, start_dim=0, end_dim=-2)
                # torch.flatten(hat_x_k_plus_1_given_k, start_dim=0, end_dim=-2)[0] - hat_x_k_plus_1_given_k[0,0]
                # torch.flatten(hat_x_k_plus_1_given_k, start_dim=0, end_dim=-2)[1] - hat_x_k_plus_1_given_k[0,1]
                # torch.flatten(hat_x_k_plus_1_given_k, start_dim=0, end_dim=-2)[hat_x_k_plus_1_given_k.shape[1]] - hat_x_k_plus_1_given_k[1,0]
                validIndices = torch.zeros_like(hat_x_k_plus_1_given_k_flatten[:, 0]).bool()
                for s in range(lengthOfSeries.shape[0]):
                    seriesLength = lengthOfSeries[s] - 1
                    seriesStartIdx = nTime*s
                    seriesStopIdx = seriesStartIdx + seriesLength
                    validIndices[seriesStartIdx : seriesStopIdx] = True
                    
                if trainOnSmoother: validIndices = torch.logical_and(validIndices, DefinedClassIndices_flatten[:, 0])
                                                    
                lossFilter = criterion(hat_x_k_plus_1_given_k_flatten[validIndices], filteringLabels_flatten[validIndices, 0])            
                lossSmoother = criterion(hat_x_k_plus_1_given_N_minus_1_flatten[validIndices], filteringLabels_flatten[validIndices, 0])            
                
                loss = lossFilter + lossSmoother
                loss = measurements.shape[0]/len(trainLoader.dataset) * loss
                
                if mode == 'train':                        
                    loss.backward()
                    optimizer.step()  # parameter update
    
                train_loss += loss.item()  
                
                correctFilteringClassification, filterLikelihoods, correctFilterClassification_nSamples = np.zeros(nClasses+1), np.zeros(nClasses+1), np.zeros(nClasses+1)
                correctSmoothingClassification, smootherLikelihoods, correctSmootherClassification_nSamples = np.zeros(nClasses+1), np.zeros(nClasses+1), np.zeros(nClasses+1)
                filterEstimations, smootherEstimations, labels = hat_x_k_plus_1_given_k_flatten[validIndices], hat_x_k_plus_1_given_N_minus_1_flatten[validIndices], filteringLabels_flatten[validIndices, 0]
                for c in range(nClasses):
                    labelIndices = labels == c
                    correctFilterClassification_nSamples[c] = labelIndices.sum().item()
                    correctSmootherClassification_nSamples[c] = labelIndices.sum().item()
                    if labelIndices.sum() > 0:
                        correctFilteringClassification[c] = ((torch.argmax(filterEstimations[labelIndices], dim=1) == labels[labelIndices]).sum()).item()     
                        correctSmoothingClassification[c] = ((torch.argmax(smootherEstimations[labelIndices], dim=1) == labels[labelIndices]).sum()).item()     
                        filterLikelihoods[c] = np.exp(filterEstimations[labelIndices, c].detach().cpu().numpy()).sum()
                        smootherLikelihoods[c] = np.exp(smootherEstimations[labelIndices, c].detach().cpu().numpy()).sum()
                    else:
                        correctFilteringClassification[c], filterLikelihoods[c] = 0, 0
                        correctSmoothingClassification[c], smootherLikelihoods[c] = 0, 0
                classesIndices = labels > 0#-1   # 0 is the no class that the filter is not trained on
                correctFilteringClassification[-1] = ((torch.argmax(filterEstimations[classesIndices], dim=1) == labels[classesIndices]).sum()).item()   
                correctSmoothingClassification[-1] = ((torch.argmax(smootherEstimations[classesIndices], dim=1) == labels[classesIndices]).sum()).item()   
                allFilterEstimations, allSmootherEstimations, allLabels = filterEstimations[classesIndices].detach().cpu().numpy(), smootherEstimations[classesIndices].detach().cpu().numpy(), labels[classesIndices].detach().cpu().numpy()
                for i in range(allFilterEstimations.shape[0]):
                    filterLikelihoods[-1] = filterLikelihoods[-1] + np.exp(allFilterEstimations[i, allLabels[i]])
                    smootherLikelihoods[-1] = smootherLikelihoods[-1] + np.exp(allSmootherEstimations[i, allLabels[i]])
                correctFilterClassification_nSamples[-1] = classesIndices.sum().item()
                correctSmootherClassification_nSamples[-1] = classesIndices.sum().item()
                train_Filtercorrect = train_Filtercorrect + correctFilteringClassification
                train_FilterLikelihood = train_FilterLikelihood + filterLikelihoods
                train_FilterCorrect_nSamples = train_FilterCorrect_nSamples + correctFilterClassification_nSamples
                train_Smoothercorrect = train_Smoothercorrect + correctSmoothingClassification
                train_SmootherLikelihood = train_SmootherLikelihood + smootherLikelihoods
                train_SmootherCorrect_nSamples = train_SmootherCorrect_nSamples + correctSmootherClassification_nSamples
    
            #scheduler.step(train_loss)
            trainLossList.append(train_loss)
            tFilterIndices = train_FilterCorrect_nSamples > 0
            trainFilterCorrectFraction, trainFilterLikelihoodFractions = np.zeros(nClasses+1), np.zeros(nClasses+1)
            trainFilterCorrectFraction[tFilterIndices] = np.divide(train_Filtercorrect[tFilterIndices], train_FilterCorrect_nSamples[tFilterIndices])
            trainFilterLikelihoodFractions[tFilterIndices] = np.divide(train_FilterLikelihood[tFilterIndices], train_FilterCorrect_nSamples[tFilterIndices])
            trainFilterCorrectFractionist.append(trainFilterCorrectFraction)
            trainFilterLikelihoodList.append(trainFilterLikelihoodFractions)
            
            tSmootherIndices = train_SmootherCorrect_nSamples > 0
            trainSmootherCorrectFraction, trainSmootherLikelihoodFractions = np.zeros(nClasses+1), np.zeros(nClasses+1)
            trainSmootherCorrectFraction[tSmootherIndices] = np.divide(train_Smoothercorrect[tSmootherIndices], train_SmootherCorrect_nSamples[tSmootherIndices])
            trainSmootherLikelihoodFractions[tSmootherIndices] = np.divide(train_SmootherLikelihood[tSmootherIndices], train_SmootherCorrect_nSamples[tSmootherIndices])
            trainSmootherCorrectFractionist.append(trainSmootherCorrectFraction)
            trainSmootherLikelihoodList.append(trainSmootherLikelihoodFractions)
            # print("Outside: measurements size", measurements.size(), "combination size", combination.size())
        
        if np.mod(epoch,10) == 0:
            validation_loss, validation_correctFilterFraction, validation_likelihoodFilterFraction = 0.0, np.zeros(nClasses+1), np.zeros(nClasses+1)
            validation_correctSmootherFraction, validation_likelihoodSmootherFraction = np.zeros(nClasses+1), np.zeros(nClasses+1)
            stateEst_ANN.eval()
            meanFilterCorrect_vsTime, meanFilterLikelihood_vsTime, meanLikelihood_vsTime_nSamples = np.zeros(int(10*modelDict['fs'])), np.zeros(int(10*modelDict['fs'])), np.zeros(int(10*modelDict['fs']))
            meanSmootherCorrect_vsTime, meanSmootherLikelihood_vsTime = np.zeros(int(10*modelDict['fs'])), np.zeros(int(10*modelDict['fs']))
            meanLikelihood_tVec = (1 + np.arange(meanFilterLikelihood_vsTime.shape[0])) / modelDict['fs']
            for validationEpoch in range(nValidationEpochs):
                if mode == 'test': print(f'starting validation epoch {validationEpoch} out of {nValidationEpochs}')
                validation_loss_singleEpoch, validation_correctFilterFraction_singleEpoch, validation_FilterLikelihood_singleEpoch, validation_correctFilterFraction_singleEpoch_nSamples = 0.0, 0.0, 0.0, 0.0
                validation_correctSmootherFraction_singleEpoch, validation_SmootherLikelihood_singleEpoch, validation_correctSmootherFraction_singleEpoch_nSamples = 0.0, 0.0, 0.0
                for i_batch, sample_batched in enumerate(validationLoader):
                    lengthOfSeries = sample_batched["lengthOfSeries"]
                    validSeries = lengthOfSeries >= minimalSeriesLength
                    if not(validSeries.any()): continue
                    lengthOfSeries = lengthOfSeries[validSeries]
                    maxLengthOfSeries = lengthOfSeries.max()
                    measurements = sample_batched["measurements"][:, :maxLengthOfSeries][validSeries]  # [nBatch, nTime, nFeature]
                    measurements_tvec = sample_batched["tVec"][:, :maxLengthOfSeries][validSeries]
                    
                    #  crop measurements for augmentations purpose:                
                    measurements, measurements_tvec, lengthOfSeries = augCrop(measurements, measurements_tvec, lengthOfSeries, nMinimalSamples, int(np.min([int(maxLengthOfSeries/strideHorizontal)*strideHorizontal, nMaximalSamples])), fillValue)
                        
                    #if not(trainOnSmoother):
                    labels = measurements[:, :, -1][:, :, None].type(torch.int64)
                    filteringLabels = labels[:, 1:]
                    if trainOnSmoother:
                        DefinedClassIndices = filteringLabels > 0
                        DefinedClassIndices = DefinedClassIndices.to(device)
                        
                    measurements = measurements[:, :, :-1]
                    currentBatchSize = measurements.shape[0]
                    
                    if modelDict['useSelectedFeatures']: measurements = measurements[:, :, modelDict['featuresIncludeInTrainIndices']]                                                
    
                    measurements = measurements.to(device)            
                    lengthOfSeries = lengthOfSeries.to(device)
                    
                    if trainOnSmoother:
                        hat_x_k_plus_1_given_N = smoother_rnn(measurements)          
                        # now hat_x_k_plus_1_given_N at [:,k] has the estimation of the state at time [k+1] given measurements up to and including time N-1                               
                        hat_x_k_plus_1_given_N = hat_x_k_plus_1_given_N[:, :-1]
                        filteringLabels = torch.argmax(hat_x_k_plus_1_given_N, dim=2).unsqueeze(2).detach()
                    else:
                        filteringLabels = filteringLabels.to(device)
                    # measurements, funcOfMeas of shape [batchSize, nTime, nFeatures]
    
                    hat_x_k_plus_1_given_k, hat_x_k_plus_1_given_N_minus_1 = stateEst_ANN(measurements)
                    hat_x_k_plus_1_given_k, hat_x_k_plus_1_given_N_minus_1 = hat_x_k_plus_1_given_k[:, :-1], hat_x_k_plus_1_given_N_minus_1[:, :-1], # filtering
                    nPatients, nTime, _ = hat_x_k_plus_1_given_k.shape
                    # now hat_x_k_plus_1_given_k at [:,k] has the estimation of the state at time [k+1] given measurements up to and including time k
                    # filteringLabels at [:k] has the label of time [k+1]
                    
                    hat_x_k_plus_1_given_k_flatten = torch.flatten(hat_x_k_plus_1_given_k, start_dim=0, end_dim=-2)
                    hat_x_k_plus_1_given_N_minus_1_flatten = torch.flatten(hat_x_k_plus_1_given_N_minus_1, start_dim=0, end_dim=-2)
                    filteringLabels_flatten = torch.flatten(filteringLabels, start_dim=0, end_dim=-2)
                    if trainOnSmoother: DefinedClassIndices_flatten = torch.flatten(DefinedClassIndices, start_dim=0, end_dim=-2)
                    # torch.flatten(hat_x_k_plus_1_given_k, start_dim=0, end_dim=-2)[0] - hat_x_k_plus_1_given_k[0,0]
                    # torch.flatten(hat_x_k_plus_1_given_k, start_dim=0, end_dim=-2)[1] - hat_x_k_plus_1_given_k[0,1]
                    # torch.flatten(hat_x_k_plus_1_given_k, start_dim=0, end_dim=-2)[hat_x_k_plus_1_given_k.shape[1]] - hat_x_k_plus_1_given_k[1,0]
                    validIndices = torch.zeros_like(hat_x_k_plus_1_given_k_flatten[:, 0]).bool()
                    if mode == 'test': 
                        newStateStartIndices = torch.zeros_like(hat_x_k_plus_1_given_k_flatten[:, 0]).bool()
                        timeFromStateStart = torch.zeros_like(hat_x_k_plus_1_given_k_flatten[:, 0], dtype=torch.int64)  # [samples]
                        singleOne = torch.ones(1, dtype=torch.int64, device=hat_x_k_plus_1_given_k_flatten.device)
                    for s in range(lengthOfSeries.shape[0]):
                        seriesLength = lengthOfSeries[s] - 1
                        seriesStartIdx = nTime*s
                        seriesStopIdx = seriesStartIdx + seriesLength
                        validIndices[seriesStartIdx : seriesStopIdx] = True
                        
                        if mode == 'test':
                            startOfStateIndices = (seriesStartIdx + torch.nonzero(torch.diff(filteringLabels_flatten[seriesStartIdx : seriesStopIdx][:, 0], dim=0))+1)[:, 0]
                            startOfStateIndices = torch.cat((seriesStartIdx*singleOne, startOfStateIndices), dim=0)                                                
                            newStateStartIndices[startOfStateIndices] = True
                            for i in range(startOfStateIndices.shape[0]):
                                timeFromStateStart[startOfStateIndices[i]:seriesStopIdx] = 1 + torch.arange(seriesStopIdx-startOfStateIndices[i])
                            
                    if trainOnSmoother: validIndices = torch.logical_and(validIndices, DefinedClassIndices_flatten[:, 0])                    
                    lossFilter = criterion(hat_x_k_plus_1_given_k_flatten[validIndices], filteringLabels_flatten[validIndices, 0])            
                    lossSmoother = criterion(hat_x_k_plus_1_given_N_minus_1_flatten[validIndices], filteringLabels_flatten[validIndices, 0])            
                    
                    loss = lossFilter + lossSmoother
             
                    loss = measurements.shape[0]/len(validationLoader.dataset) * loss
                    
                    validation_loss_singleEpoch += loss.item()
                    
                    correctFilterClassification, correctFilterClassification_nSamples, FilterLikelihoods = np.zeros(nClasses+1), np.zeros(nClasses+1), np.zeros(nClasses+1)
                    correctSmootherClassification, correctSmootherClassification_nSamples, SmootherLikelihoods = np.zeros(nClasses+1), np.zeros(nClasses+1), np.zeros(nClasses+1)
                    FilterEstimations, SmootherEstimations, labels = hat_x_k_plus_1_given_k_flatten[validIndices], hat_x_k_plus_1_given_N_minus_1_flatten[validIndices], filteringLabels_flatten[validIndices, 0]
                    if mode == 'test': 
                        FilterLikelihoods_flatten = np.exp(FilterEstimations[range(FilterEstimations.shape[0]), labels].detach().cpu().numpy())
                        SmootherLikelihoods_flatten = np.exp(SmootherEstimations[range(SmootherEstimations.shape[0]), labels].detach().cpu().numpy())
                        correctFilter_flatten = torch.argmax(FilterEstimations, dim=1) == labels
                        correctSmoother_flatten = torch.argmax(SmootherEstimations, dim=1) == labels
                        '''
                        likelihoods_flatten_normalized = likelihoods_flatten
                        currentWeights, currentWeightsValidIndices = np.zeros(nClasses), np.zeros(nClasses, dtype=bool)
                        for c in range(nClasses):
                            labelIndices = (labels == c).detach().cpu().numpy()
                            if labelIndices.any():
                                currentWeights[c] = labelIndices.sum()/labels.shape[0]
                                currentWeightsValidIndices[c] = True
                        currentWeights[currentWeightsValidIndices] = 1/currentWeights[currentWeightsValidIndices]
                        currentWeights[currentWeightsValidIndices] = currentWeights[currentWeightsValidIndices]/currentWeights[currentWeightsValidIndices].sum()
                        for c in range(nClasses):
                            labelIndices = (labels == c).detach().cpu().numpy()                        
                            likelihoods_flatten_normalized[labelIndices] = nClasses*currentWeights[c]*likelihoods_flatten_normalized[labelIndices]
                        '''
                        timeFromStateStart = timeFromStateStart.detach().cpu().numpy()  # [samples]
                        validIndices = validIndices.detach().cpu().numpy()
                        for t in range(meanFilterLikelihood_vsTime.shape[0]):
                            specificTimeIndices = np.logical_and(timeFromStateStart == t+1, validIndices)[validIndices]
                            meanFilterLikelihood_vsTime[t] = meanFilterLikelihood_vsTime[t] + FilterLikelihoods_flatten[specificTimeIndices].sum()
                            meanFilterCorrect_vsTime[t] = meanFilterCorrect_vsTime[t] + correctFilter_flatten[specificTimeIndices].sum()
                            meanSmootherLikelihood_vsTime[t] = meanSmootherLikelihood_vsTime[t] + SmootherLikelihoods_flatten[specificTimeIndices].sum()
                            meanSmootherCorrect_vsTime[t] = meanSmootherCorrect_vsTime[t] + correctSmoother_flatten[specificTimeIndices].sum()
                            meanLikelihood_vsTime_nSamples[t] = meanLikelihood_vsTime_nSamples[t] + specificTimeIndices.sum()
                        
                    for c in range(nClasses):
                        labelIndices = labels == c
                        correctFilterClassification_nSamples[c] = labelIndices.sum().item()
                        correctSmootherClassification_nSamples[c] = labelIndices.sum().item()
                        if labelIndices.sum() > 0:
                            correctFilterClassification[c] = ((torch.argmax(FilterEstimations[labelIndices], dim=1) == labels[labelIndices]).sum()).item()                    
                            FilterLikelihoods[c] = np.exp(FilterEstimations[labelIndices, c].detach().cpu().numpy()).sum()
                            correctSmootherClassification[c] = ((torch.argmax(SmootherEstimations[labelIndices], dim=1) == labels[labelIndices]).sum()).item()                    
                            SmootherLikelihoods[c] = np.exp(SmootherEstimations[labelIndices, c].detach().cpu().numpy()).sum()
                        else:
                            correctFilterClassification[c] = 0
                            correctSmootherClassification[c] = 0
                    classesIndices = labels > 0#-1    # 0 is the no class that the filter is not trained on
                    correctFilterClassification[-1] = ((torch.argmax(FilterEstimations[classesIndices], dim=1) == labels[classesIndices]).sum()).item()   
                    correctSmootherClassification[-1] = ((torch.argmax(SmootherEstimations[classesIndices], dim=1) == labels[classesIndices]).sum()).item()   
                    allFilterEstimations, allLabels = FilterEstimations[classesIndices].detach().cpu().numpy(), labels[classesIndices].detach().cpu().numpy()
                    allSmootherEstimations, allLabels = SmootherEstimations[classesIndices].detach().cpu().numpy(), labels[classesIndices].detach().cpu().numpy()
                    for i in range(allFilterEstimations.shape[0]):
                        FilterLikelihoods[-1] = FilterLikelihoods[-1] + np.exp(allFilterEstimations[i, allLabels[i]])
                        SmootherLikelihoods[-1] = SmootherLikelihoods[-1] + np.exp(allSmootherEstimations[i, allLabels[i]])
                    correctFilterClassification_nSamples[-1] = classesIndices.sum().item()
                    correctSmootherClassification_nSamples[-1] = classesIndices.sum().item()
                    validation_correctFilterFraction_singleEpoch = validation_correctFilterFraction_singleEpoch + correctFilterClassification
                    validation_FilterLikelihood_singleEpoch = validation_FilterLikelihood_singleEpoch + FilterLikelihoods
                    validation_correctFilterFraction_singleEpoch_nSamples = validation_correctFilterFraction_singleEpoch_nSamples + correctFilterClassification_nSamples                                                                        
                    validation_correctSmootherFraction_singleEpoch = validation_correctSmootherFraction_singleEpoch + correctSmootherClassification
                    validation_SmootherLikelihood_singleEpoch = validation_SmootherLikelihood_singleEpoch + SmootherLikelihoods
                    validation_correctSmootherFraction_singleEpoch_nSamples = validation_correctSmootherFraction_singleEpoch_nSamples + correctSmootherClassification_nSamples                                                                        
                
                validation_loss += validation_loss_singleEpoch/nValidationEpochs
                vFilterIndices = validation_correctFilterFraction_singleEpoch_nSamples > 0
                validation_correctFilterFraction[vFilterIndices] = validation_correctFilterFraction[vFilterIndices] + np.divide(validation_correctFilterFraction_singleEpoch[vFilterIndices], validation_correctFilterFraction_singleEpoch_nSamples[vFilterIndices])/nValidationEpochs
                validation_likelihoodFilterFraction[vFilterIndices] = validation_likelihoodFilterFraction[vFilterIndices] + np.divide(validation_FilterLikelihood_singleEpoch[vFilterIndices], validation_correctFilterFraction_singleEpoch_nSamples[vFilterIndices])/nValidationEpochs
                vSmootherIndices = validation_correctSmootherFraction_singleEpoch_nSamples > 0
                validation_correctSmootherFraction[vSmootherIndices] = validation_correctSmootherFraction[vSmootherIndices] + np.divide(validation_correctSmootherFraction_singleEpoch[vSmootherIndices], validation_correctSmootherFraction_singleEpoch_nSamples[vSmootherIndices])/nValidationEpochs
                validation_likelihoodSmootherFraction[vSmootherIndices] = validation_likelihoodSmootherFraction[vSmootherIndices] + np.divide(validation_SmootherLikelihood_singleEpoch[vSmootherIndices], validation_correctSmootherFraction_singleEpoch_nSamples[vSmootherIndices])/nValidationEpochs
            
            #if mode == 'train' and epoch > patience: scheduler.step(train_loss)
            # print(f'epoch: {epoch}; Validation Mean error w.r.t mean(abs(labels)) = {valid_loss / len(validationLoader) / combination_mean}; mean error in theta w.r.t mean(theta)= {meanErrorIn_theta}')
            validationLossList.append(validation_loss)
            validationCorrectFilterFractionist.append(validation_correctFilterFraction)
            validationFilterLikelihoodList.append(validation_likelihoodFilterFraction)
            validationCorrectSmootherFractionist.append(validation_correctSmootherFraction)
            validationSmootherLikelihoodList.append(validation_likelihoodSmootherFraction)
            #currentValidationMeanStd = patientsDataset.calcMeanStd(validationData.indices, combination_ANN)
            #currentTrainMeanStd = patientsDataset.calcMeanStd(trainData.indices, combination_ANN)
            #validationMeanStdList.append(currentValidationMeanStd)
            #trainMeanStdList.append(currentTrainMeanStd)
    
            if previousValidationLoss > validation_loss:
                #print(f'reference Std value = {referenceStdValue}')
                #print(f'epoch {epoch}, Validation Mean Std Decreased({previousValidationMeanStd:.6f}--->{currentValidationMeanStd:.6f})')
                print(f'epoch {epoch}, Validation loss Decreased({previousValidationLoss:.6f}--->{validation_loss:.6f})')
                #print(f'epoch {epoch}, Validation correct fraction ({np.round(100*previousValidationFilterMeanCorrectFraction)}--->{np.round(100*validation_correctFilterFraction)})')
                print(f'epoch {epoch}, Validation filter likelihood ({np.round(100*previousValidationLikelihoodFilterFraction)}--->{np.round(100*validation_likelihoodFilterFraction)})')
                #print(f'epoch {epoch}, Validation correct fraction ({np.round(100*previousValidationSmootherMeanCorrectFraction)}--->{np.round(100*validation_correctSmootherFraction)})')
                print(f'epoch {epoch}, Validation smoother likelihood ({np.round(100*previousValidationLikelihoodSmootherFraction)}--->{np.round(100*validation_likelihoodSmootherFraction)})')
                #print(f'epoch {epoch}, Train Mean Std {currentTrainMeanStd:.6f}')
                if mode == 'train':
                    print(f'epoch {epoch}, Train loss {train_loss:.6f}')
                    #print(f'epoch {epoch}, Train correct fraction {np.round(100*trainFilterCorrectFraction)}')
                    print(f'epoch {epoch}, Train filter likelihood {np.round(100*trainFilterLikelihoodFractions)}')
                    #print(f'epoch {epoch}, Train correct fraction {np.round(100*trainSmootherCorrectFraction)}')
                    print(f'epoch {epoch}, Train smoother likelihood {np.round(100*trainSmootherLikelihoodFractions)}')
                
                #previousValidationMeanStd = currentValidationMeanStd
            if mode == 'train' and previousValidationLoss > validation_loss:# and epoch > patience:
                previousValidationLoss = validation_loss
                previousValidationFilterMeanCorrectFraction = validation_correctFilterFraction
                previousValidationLikelihoodFilterFraction = validation_likelihoodFilterFraction
                previousValidationSmootherMeanCorrectFraction = validation_correctSmootherFraction
                previousValidationLikelihoodSmootherFraction = validation_likelihoodSmootherFraction
                if enableDataParallel:
                    model_state_dict = copy.deepcopy(stateEst_ANN.module.state_dict())
                    #torch.save(combination_ANN.module.state_dict(), fileName + '_model.pt')
                else:
                    model_state_dict = copy.deepcopy(stateEst_ANN.state_dict())
                    #torch.save(combination_ANN.state_dict(), fileName + '_model.pt')
            '''    
            else:
                print(f'reference Std value = {referenceStdValue}')
                print(f'epoch {epoch}, Validation Mean Std ({previousValidationMeanStd:.6f}--->{currentValidationMeanStd:.6f})')
                print(f'epoch {epoch}, Validation loss Increased({min_valid_loss:.6f}--->{validation_loss:.6f})')
                print(f'epoch {epoch}, Train Mean Std {currentTrainMeanStd:.6f}')
                print(f'epoch {epoch}, Train loss {train_loss:.6f}')
            '''
    
            
            if mode == 'train' and epoch > patience and np.min(validationLossList[-4:]) >= validationLossList[-4-1]:
                print(f'Stoping optimization due to zero improvement in validation loss')
                break  
            if mode == 'test': 
                if enableDataParallel:
                    model_state_dict = copy.deepcopy(stateEst_ANN.module.state_dict())
                    #torch.save(combination_ANN.module.state_dict(), fileName + '_model.pt')
                else:
                    model_state_dict = copy.deepcopy(stateEst_ANN.state_dict())
                previousValidationLoss = np.inf
                break
            #if epoch > 9: break

    epochVec = np.arange(0, len(validationCorrectFilterFractionist))    
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
        
    if mode == 'test':
        meanLikelihood_vsTime_tuple = (np.divide(meanFilterLikelihood_vsTime, meanLikelihood_vsTime_nSamples), meanLikelihood_tVec, np.divide(meanFilterCorrect_vsTime, meanLikelihood_vsTime_nSamples), np.divide(meanSmootherLikelihood_vsTime, meanLikelihood_vsTime_nSamples), meanLikelihood_tVec, np.divide(meanSmootherCorrect_vsTime, meanLikelihood_vsTime_nSamples))
    else:
        meanLikelihood_vsTime_tuple = None
        
    return model_state_dict, previousValidationLoss, meanLikelihood_vsTime_tuple