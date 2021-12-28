#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 15:32:38 2021

@author: ront
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
from scipy import interpolate
import pickle

def united_acc_gyr_singleUserSingleDevice(Phones_acc_specificUserDevice_DF, Phones_gyr_specificUserDevice_DF, cutObservationsGapThr, minObservedDuration):
    Phone_specificUserSpecificDevice = pd.concat([Phones_acc_specificUserDevice_DF, Phones_gyr_specificUserDevice_DF], ignore_index=True).reset_index(drop=True)
    Phone_specificUserSpecificDevice = Phone_specificUserSpecificDevice.sort_values(by=['Creation_Time'], ignore_index=True)
    Phone_specificUserSpecificDevice = Phone_specificUserSpecificDevice.drop(columns=['Arrival_Time'])
    
    gtNanIndices = Phone_specificUserSpecificDevice['gt'].isna()
    Phone_specificUserSpecificDevice.loc[gtNanIndices, 'gt'] = 'noClass'
    
    
    Phone_specificUserSpecificDevice = Phone_specificUserSpecificDevice.interpolate(method='zero')
    Phone_specificUserSpecificDevice = Phone_specificUserSpecificDevice.dropna(axis='index')
    Phone_specificUserSpecificDevice['Creation_Time'] = Phone_specificUserSpecificDevice['Creation_Time'] - Phone_specificUserSpecificDevice['Creation_Time'].to_numpy()[0]
    Phone_specificUserSpecificDevice['Creation_Time'] = Phone_specificUserSpecificDevice['Creation_Time']/1e9
    
    Creation_Time = Phone_specificUserSpecificDevice['Creation_Time'].to_numpy()
    Creation_Time_diff = np.diff(Creation_Time)
    startIndices = np.concatenate((np.array([0]), np.where(Creation_Time_diff > cutObservationsGapThr)[0]+1))
    stopIndices = np.concatenate((startIndices[1:]-1, np.array([Creation_Time.shape[0]])-1))
    batch = np.zeros(Creation_Time.shape[0])
    for i in range(startIndices.shape[0]):
        batch[startIndices[i]:stopIndices[i]+1] = i
    Phone_specificUserSpecificDevice.insert(0, 'batch', batch)
    durations = Creation_Time[stopIndices] - Creation_Time[startIndices]
    for i in range(startIndices.shape[0]):
        if durations[i] < minObservedDuration:
            Phone_specificUserSpecificDevice.drop(Phone_specificUserSpecificDevice.loc[Phone_specificUserSpecificDevice['batch']==i].index, inplace=True)
    Phone_specificUserSpecificDevice = Phone_specificUserSpecificDevice.reset_index(drop=True)
    Phone_specificUserSpecificDevice.drop(columns=['batch'], inplace=True)
    Creation_Time = Phone_specificUserSpecificDevice['Creation_Time'].to_numpy()
    Creation_Time_diff = np.diff(Creation_Time)
    startIndices = np.concatenate((np.array([0]), np.where(Creation_Time_diff > cutObservationsGapThr)[0]+1))
    stopIndices = np.concatenate((startIndices[1:]-1, np.array([Creation_Time.shape[0]])-1))
    batch = np.zeros(Creation_Time.shape[0])
    for i in range(startIndices.shape[0]):
        batch[startIndices[i]:stopIndices[i]+1] = i
    Phone_specificUserSpecificDevice.insert(0, 'batch', batch)
    durations = Creation_Time[stopIndices] - Creation_Time[startIndices]
    assert (durations >= minObservedDuration).all()
    return Phone_specificUserSpecificDevice
    

def DataFrameResample(patientDf, fs):
    assert len(patientDf['Id'].unique().tolist()) == 1  # this function works for a single Id Df
    Id = patientDf['Id'].unique().tolist()[0]
    columns = patientDf.columns.tolist()
    patientDfResampled = pd.DataFrame(columns=columns)
    batchList = patientDf['batch'].unique().tolist()
    for batch in batchList:
        singleBatch = patientDf[patientDf['batch'] == batch]
        tVec, data = singleBatch['time'].to_numpy(), singleBatch[columns[3:]].to_numpy()
        
        phases = np.mod(tVec, 1/fs)
        values, counts = np.unique(phases, return_counts=True)
        mostPopularPhase = values[np.argmax(counts)]
        
        duration = tVec[-1] - tVec[0]  # [sec]
        nSamplesNew = int(np.floor(duration*fs + 1))
        tVecNew = tVec[0] + np.arange(0, nSamplesNew)/fs
        tVecNew = tVecNew + (np.mod(tVecNew[0], 1/fs) - mostPopularPhase)
        tVecNew = tVecNew[np.logical_and(tVecNew >= tVec.min(), tVecNew <= tVec.max())]
        nSamplesNew = tVecNew.shape[0]
        
        f = interpolate.interp1d(tVec, data, kind='zero', axis=0)
        dataResampled = f(tVecNew)  # use interpolation function returned by `interp1d`
        
        IdValues, batchValues = Id*np.ones((nSamplesNew, 1)), batch*np.ones((nSamplesNew, 1))
        newData = np.concatenate((tVecNew[:, None], IdValues, batchValues, dataResampled), axis=1)
        df = pd.DataFrame(columns=columns, data=newData)
        
        patientDfResampled = pd.concat([patientDfResampled, df], ignore_index=True)
        
    return patientDfResampled

plotAllData = False

cutObservationsGapThr = 250/1000  # [sec]
minObservedDuration = 10  # [sec]
fs = 1/5e-3  # [hz]

phone_acc_path = './Activity recognition exp/Phones_accelerometer.csv'
phone_gyr_path = './Activity recognition exp/Phones_gyroscope.csv'
watch_acc_path = './Activity recognition exp/Watch_accelerometer.csv'
watch_gyr_path = './Activity recognition exp/Watch_gyroscope.csv'

createType = 'phone'  # {'phone', 'watch'}
if createType == 'phone':
    files = [(phone_acc_path, 'Pacc'), (phone_gyr_path, 'Pgyr')]  #, (watch_acc_path, 'Wacc'), (watch_gyr_path, 'Wgyr')]
elif createType == 'watch':
    files = [(watch_acc_path, 'Wacc'), (watch_gyr_path, 'Wgyr')]

Phones_acc_DF = pd.read_csv(files[0][0])
Models = Phones_acc_DF['Model'].unique().tolist()

metaDataDf = pd.DataFrame(columns=['Id', 'Classification', 'Device', 'User'])
phonesDf = pd.DataFrame(columns=['time', 'Id', 'batch', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'gt'])
Id = -1

for Model in Models:
    print(f'starting model {Model}')
#Model = ['nexus4']
#files = [watch_gyr_path]
#names = ['Wgyr']
    Creation_Time_diff_array = np.ndarray(0)
    continualObservationsDurations_array = np.ndarray(0)
    deviceDict = dict()
    
    path2file = files[0][0]
    Phones_acc_DF = pd.read_csv(path2file).drop(columns=['Index'])
    Phones_acc_DF = Phones_acc_DF[Phones_acc_DF['Model'].isin([Model])]
    Phones_acc_DF = Phones_acc_DF.rename(columns={"x": "acc_x", "y": "acc_y", "z": "acc_z"})
    Phones_acc_DF.insert(5, "gyr_x", np.nan*np.ones(Phones_acc_DF.shape[0]))
    Phones_acc_DF.insert(6, "gyr_y", np.nan*np.ones(Phones_acc_DF.shape[0]))
    Phones_acc_DF.insert(7, "gyr_z", np.nan*np.ones(Phones_acc_DF.shape[0]))
    
    
    path2file = files[1][0]
    Phones_gyr_DF = pd.read_csv(path2file).drop(columns=['Index'])
    Phones_gyr_DF = Phones_gyr_DF[Phones_gyr_DF['Model'].isin([Model])]
    Phones_gyr_DF = Phones_gyr_DF.rename(columns={"x": "gyr_x", "y": "gyr_y", "z": "gyr_z"})
    Phones_gyr_DF.insert(2, "acc_x", np.nan*np.ones(Phones_gyr_DF.shape[0]))
    Phones_gyr_DF.insert(3, "acc_y", np.nan*np.ones(Phones_gyr_DF.shape[0]))
    Phones_gyr_DF.insert(4, "acc_z", np.nan*np.ones(Phones_gyr_DF.shape[0]))
    
    Users = list(set(Phones_acc_DF['User'].unique().tolist()) & set(Phones_gyr_DF['User'].unique().tolist()))
    
    
    for user in Users:
        Phones_acc_specificUser_DF = Phones_acc_DF[Phones_acc_DF['User'] == user]
        Phones_gyr_specificUser_DF = Phones_gyr_DF[Phones_gyr_DF['User'] == user]
        Devices = list(set(Phones_acc_specificUser_DF['Device'].unique().tolist()) & set(Phones_gyr_specificUser_DF['Device'].unique().tolist()))
        for device in Devices:            
            
            if createType == 'phone':
                if (user == 'a' and device == 'nexus4_1') or \
                    (user == 'f' and device == 's3_1') or \
                        (device in ['s3mini_1', 's3mini_2']):
                    continue
            elif createType == 'watch':
                if (user == 'i' and device in ['gear_1','gear_2']) or \
                            (user == 'i' and device in ['gear_1']) or \
                                (user == 'g' and device in ['lgwatch_1']) or \
                                    (user in ['b', 'h'] and device in ['lgwatch_2']):
                    continue
            
            print(f'model, user, device: {Model}, {user}, {device}')
            Phones_acc_specificUserDevice_DF = Phones_acc_specificUser_DF[Phones_acc_specificUser_DF['Device'] == device]
            Phones_gyr_specificUserDevice_DF = Phones_gyr_specificUser_DF[Phones_gyr_specificUser_DF['Device'] == device]
            Phone_specificUserSpecificDevice = united_acc_gyr_singleUserSingleDevice(Phones_acc_specificUserDevice_DF, Phones_gyr_specificUserDevice_DF, cutObservationsGapThr, minObservedDuration)
            
            gt = Phone_specificUserSpecificDevice['gt']
            gt[gt=='noClass'], gt[gt=='stand'], gt[gt=='sit'], gt[gt=='walk'], gt[gt=='stairsup'], gt[gt=='stairsdown'], gt[gt=='bike'] = 0, 1, 2, 3, 4, 5, 6
            # this proceedure updated the values of the dataframe (because .copy() wasn't used)
            Phone_specificUserSpecificDevice = Phone_specificUserSpecificDevice.rename(columns={'Creation_Time': 'time'})
            columns = Phone_specificUserSpecificDevice.columns.tolist()
            columns = columns[1:2] + columns[0:1] + columns[2:8] + [columns[-1]]
            Phone_specificUserSpecificDevice = Phone_specificUserSpecificDevice[columns]
            Id += 1
            Phone_specificUserSpecificDevice.insert(1, 'Id', Id*np.ones(Phone_specificUserSpecificDevice.shape[0]))                    
            Phone_specificUserSpecificDevice_resampled = DataFrameResample(Phone_specificUserSpecificDevice, fs)
                
            phonesDf = pd.concat([phonesDf, Phone_specificUserSpecificDevice_resampled], ignore_index=True)
            
            metaData = pd.DataFrame.from_dict({'Id': [Id], 'Classification': [Model], 'Device': [device], 'User': [user]})
            metaDataDf = pd.concat([metaDataDf, metaData], ignore_index=True)
            
if createType == 'phone':
    pickle.dump({'phonesDf': phonesDf, 'metaDataDf': metaDataDf}, open('./Activity recognition exp/phonesData.pt', 'wb'))
elif createType == 'watch':
    pickle.dump({'phonesDf': phonesDf, 'metaDataDf': metaDataDf}, open('./Activity recognition exp/watchData.pt', 'wb'))
            
            
            
        
