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
import pickle

plotAllData = False

cutObservationsGapThr = 250/1000  # [sec]
minObservedDuration = 10  # [sec]

phone_acc_path = './Activity recognition exp/Phones_accelerometer.csv'
phone_gyr_path = './Activity recognition exp/Phones_gyroscope.csv'
watch_acc_path = './Activity recognition exp/Watch_accelerometer.csv'
watch_gyr_path = './Activity recognition exp/Watch_gyroscope.csv'

files = [phone_acc_path, phone_gyr_path, watch_acc_path, watch_gyr_path]
names = ['Pacc', 'Pgyr', 'Wacc', 'Wgyr']

#files = [watch_gyr_path]
#names = ['Wgyr']
Creation_Time_diff_array = np.ndarray(0)
continualObservationsDurations_array = np.ndarray(0)
deviceDict = dict()

for path2file, name in zip(files, names):

    Phones_accelerometer_DF = pd.read_csv(path2file)
    (Phones_accelerometer_DF['Creation_Time'] <= Phones_accelerometer_DF['Arrival_Time']).all()
    
    Devices = Phones_accelerometer_DF['Device'].unique()
    print(f'Devices are {Devices}')
    Users = Phones_accelerometer_DF['User'].unique()
    print(f'Users are {Users}')   
    
    #Devices = ['lgwatch_2']
    #Users = ['i']
    
    for Device in Devices:
        continualObservationsDurations_SpecificDevice_array = np.ndarray(0)
        for User in Users:
    
            if (name == 'Pacc' and User == 'a' and Device == 'nexus4_1') or \
                (name == 'Pacc' and User == 'f' and Device == 's3_1') or \
                    (name == 'Wacc' and User == 'i' and Device in ['gear_1','gear_2']) or \
                        (name == 'Wgyr' and User == 'i' and Device in ['gear_1']) or \
                            (name == 'Wgyr' and User == 'g' and Device in ['lgwatch_1']) or \
                                (name == 'Wgyr' and User in ['b', 'h'] and Device in ['lgwatch_2']):
                continue
            
            Phones_accelerometer_singleUser_DF = Phones_accelerometer_DF[Phones_accelerometer_DF['User'] == User]
            Phones_accelerometer_singleUser_singleDevice_DF = Phones_accelerometer_singleUser_DF[Phones_accelerometer_singleUser_DF['Device'] == Device]
            
            Creation_Time = Phones_accelerometer_singleUser_singleDevice_DF['Creation_Time'].to_numpy()  # [sec]
            sortIndices = np.argsort(Creation_Time)
    
            Arrival_Time = (Phones_accelerometer_singleUser_singleDevice_DF['Arrival_Time'].to_numpy()/(1e3))[sortIndices]  # [sec]
            Creation_Time = (Phones_accelerometer_singleUser_singleDevice_DF['Creation_Time'].to_numpy()/(1e9))[sortIndices]  # [sec]
            x, y, z = Phones_accelerometer_singleUser_singleDevice_DF['x'].to_numpy()[sortIndices], Phones_accelerometer_singleUser_singleDevice_DF['y'].to_numpy()[sortIndices], Phones_accelerometer_singleUser_singleDevice_DF['z'].to_numpy()[sortIndices]
            gt = Phones_accelerometer_singleUser_singleDevice_DF['gt'].copy()
            gt[pd.isna(gt)] = -1
            gt[gt=='stand'], gt[gt=='sit'], gt[gt=='walk'], gt[gt=='stairsup'], gt[gt=='stairsdown'], gt[gt=='bike'] = 0, 2, 3, 4, 5, 6
            gt = gt.to_numpy()[sortIndices]
            
            Creation_Time_diff = np.diff(Creation_Time)
            Creation_Time_diff_array = np.concatenate((Creation_Time_diff_array, Creation_Time_diff))
            
            startIndices = np.concatenate((np.array([0]), np.where(Creation_Time_diff > cutObservationsGapThr)[0]+1))
            stopIndices = np.concatenate((startIndices[1:]-1, np.array([Creation_Time.shape[0]])-1))
            
            continualObservationsDurations = Creation_Time[stopIndices] - Creation_Time[startIndices]
            continualObservationsDurations = continualObservationsDurations[continualObservationsDurations >= minObservedDuration]
            continualObservationsDurations_array = np.concatenate((continualObservationsDurations_array, continualObservationsDurations))
            
            continualObservationsDurations_SpecificDevice_array = np.concatenate((continualObservationsDurations_SpecificDevice_array, continualObservationsDurations))
            
            if plotAllData:
                #print('name = ' + name + ', user = ' + User + ', Device = ' + Device)
                StartCreationTime, StartArrivalTime = Creation_Time[0], Arrival_Time[0]
                Creation_Time = Creation_Time - StartCreationTime
                Arrival_Time = Arrival_Time - StartCreationTime
                
                plt.plot(Creation_Time, label=f'creation time. median, std of ts = {round(1e3*np.median(np.diff(Creation_Time)), 2)}, {round(1e3*np.diff(Creation_Time).std(), 2)} ms')
                plt.ylabel('sec')
                plt.grid()
                plt.legend()
                plt.title(name + f' User = {User}, Device = {Device}')
                plt.show()
        
                
                '''
                plt.plot(Arrival_Time, label='arrival time')
                plt.ylabel('sec')
                plt.grid()
                plt.legend()
                plt.title(f'User = {User}, Device = {Device}')
                plt.show()
                
                plt.plot(Arrival_Time - Creation_Time, label='arrival minus creation time')
                plt.ylabel('sec')
                plt.grid()
                plt.legend()
                plt.title(f'User = {User}, Device = {Device}')
                plt.show()
                '''
                plt.plot(Creation_Time, x, label='x')
                plt.plot(Creation_Time, y, label='y')
                plt.plot(Creation_Time, z, label='z')
                plt.plot(Creation_Time, 2*gt, label='gt')
                plt.xlabel('sec')
                plt.legend()
                plt.title(f'User = {User}, Device = {Device}')
                plt.grid()
                plt.show()
                
            deviceDict[name + '_' + Device] = continualObservationsDurations_SpecificDevice_array

plt.figure()
data = 1e3*Creation_Time_diff_array
n, bins, _ = plt.hist(data, data.shape[0], histtype='step', density=True, cumulative=True, label='hist')
plt.grid()
plt.legend()
plt.title('CDF plot of creationTimeDiff')
plt.xlabel('[ms]')
plt.xlim([0, 100])
plt.show()

plt.figure()            
data = continualObservationsDurations_array
n, bins, _ = plt.hist(data, data.shape[0], histtype='step', density=True, cumulative=True, label='hist all')
plt.grid()
plt.legend()
plt.title(f'CDF plot of continual observations; cutThr = {1000*cutObservationsGapThr} ms')
plt.xlabel('[sec]')
plt.xlim([0, 100])
plt.grid()
plt.show()

for key in deviceDict.keys():
    plt.figure()   
    data = deviceDict[key]
    n, bins, _ = plt.hist(data, data.shape[0], histtype='step', density=True, cumulative=True, label=key)
    plt.legend()
    plt.title(f'CDF plot of continual observations, total = {round(data.sum())} sec')
    plt.xlabel('[sec]')
    plt.grid()
    plt.xlim([0, data.max()])
    plt.show()
