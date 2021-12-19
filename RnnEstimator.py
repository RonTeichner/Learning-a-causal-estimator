import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from RnnEstimator_func import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim
import pickle
import time

fileName = 'sys2D_FilteringVsSmoothing'
savedGammaResults = pickle.load(open(fileName + '_gammaResults.pt', 'rb'))
sysModel, N, gammaResultList = savedGammaResults

# training properties:
batchSize = 8*40
validation_fraction = 0.2
nSeriesForTrain = 10000
nSeriesForTest = 10000

nIndependentTrainings = 1  # set a value larger than 1 to have an estimation of the std between different training sessions.

learnedFilterErrorWattList = list()
for trainItr in range(nIndependentTrainings):
    print(f'starting train iteration {trainItr}')
    # create dataset and split into train and test sets:
    print(f'creating dataset')
    patientsTrainDataset = MeasurementsDataset(sysModel, N, nSeriesForTrain)

    nValidation = int(validation_fraction*len(patientsTrainDataset))
    nTrain = len(patientsTrainDataset) - nValidation
    trainData, validationData = random_split(patientsTrainDataset,[nTrain, nValidation])
    trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True, num_workers=0)
    validationLoader = DataLoader(validationData, batch_size=batchSize, shuffle=False, num_workers=0)

    # patient Id's in trainLoader are obtained by, trainLoader.dataset.indices or trainData.indices
    useCuda = True
    if useCuda:
        device = 'cuda'
    else:
        device = 'cpu'

    pytorchEstimator = Pytorch_filter_smoother_Obj(sysModel, enableSmoothing=True, useCuda=useCuda)

    # create the trained model
    hidden_dim = 10
    num_layers = 2
    Filter_rnn = RNN_Filter(input_dim = pytorchEstimator.dim_z, hidden_dim=hidden_dim, output_dim=pytorchEstimator.dim_x, num_layers=num_layers)

    trainModel(Filter_rnn, pytorchEstimator, trainLoader, validationLoader)

    # test
    print('starting test')
    pytorchEstimator = Pytorch_filter_smoother_Obj(sysModel, enableSmoothing=True, useCuda=False)
    Filter_rnn = RNN_Filter(input_dim = pytorchEstimator.dim_z, hidden_dim=hidden_dim, output_dim=pytorchEstimator.dim_x, num_layers=num_layers)
    device = 'cpu'
    Filter_rnn.load_state_dict(torch.load('saved_modelFilter.pt'))
    Filter_rnn.eval().to(device)
    patientsTestDataset = MeasurementsDataset(sysModel, N, nSeriesForTest)
    testLoader = DataLoader(patientsTestDataset, batch_size=nSeriesForTest, shuffle=True, num_workers=0)
    x_0_test_given_minus_1 = torch.zeros((1, testLoader.batch_size, pytorchEstimator.dim_x), dtype=torch.float, device=device)
    filter_P_init = pytorchEstimator.theoreticalBarSigma.cpu().numpy()  # filter @ time-series but all filters have the same init
    for i_batch, sample_batched in enumerate(testLoader):
        tilde_z = sample_batched["tilde_z"].transpose(1, 0)
        z = sample_batched["z"].transpose(1, 0)
        x = sample_batched["x"].transpose(1, 0)

        currentBatchSize = z.shape[1]
        hat_x_k_plus_1_given_k = Filter_rnn(z)
        learned_tilde_x_est_f = torch.cat((x_0_test_given_minus_1[:, :currentBatchSize], hat_x_k_plus_1_given_k[:-1]), dim=0)[:, :, :, None]

        # estimator init values:
        filterStateInit = np.matmul(np.linalg.cholesky(filter_P_init), np.random.randn(currentBatchSize, pytorchEstimator.dim_x, 1))
        filterStateInit = torch.tensor(filterStateInit, dtype=torch.float, requires_grad=False, device=device).contiguous()
        # filterStateInit = tilde_x[0]  This can be used if the initial state is known

        # kalman filter on z:
        tilde_x_est_f, tilde_x_est_s = pytorchEstimator(z, filterStateInit)

        learned_e_k_give_k_minus_1 = (x - learned_tilde_x_est_f).detach().numpy()
        kalman_e_k_give_k_minus_1 = (x - tilde_x_est_f).detach().numpy()
        kalman_e_k_give_N_minus_1 = (x - tilde_x_est_s).detach().numpy()

        learned_e_k_give_k_minus_1_watt = np.power(learned_e_k_give_k_minus_1, 2).sum(axis=2).flatten()
        kalman_e_k_give_k_minus_1_watt = np.power(kalman_e_k_give_k_minus_1, 2).sum(axis=2).flatten()
        kalman_e_k_give_N_minus_1_watt = np.power(kalman_e_k_give_N_minus_1, 2).sum(axis=2).flatten()

        # kalman filter on tilde_z:
        pure_tilde_x_est_f, pure_tilde_x_est_s = pytorchEstimator(tilde_z, filterStateInit)

        pure_kalman_e_k_give_k_minus_1 = (x - pure_tilde_x_est_f).detach().numpy()
        pure_kalman_e_k_give_N_minus_1 = (x - pure_tilde_x_est_s).detach().numpy()

        pure_kalman_e_k_give_k_minus_1_watt = np.power(pure_kalman_e_k_give_k_minus_1, 2).sum(axis=2).flatten()
        pure_kalman_e_k_give_N_minus_1_watt = np.power(pure_kalman_e_k_give_N_minus_1, 2).sum(axis=2).flatten()

        learnedFilterErrorWattList.append(learned_e_k_give_k_minus_1_watt.mean())

        print(f'pure kalman filter MSE {watt2dbm(pure_kalman_e_k_give_k_minus_1_watt.mean())} dbm; {(pure_kalman_e_k_give_k_minus_1_watt.mean())} W')
        print(f'kalman filter MSE {watt2dbm(kalman_e_k_give_k_minus_1_watt.mean())} dbm; {(kalman_e_k_give_k_minus_1_watt.mean())} W')
        print(f'learned filter MSE {watt2dbm(learned_e_k_give_k_minus_1_watt.mean())} dbm; {(learned_e_k_give_k_minus_1_watt.mean())} W')

        print(f'pure kalman smoother MSE {watt2dbm(pure_kalman_e_k_give_N_minus_1_watt.mean())} dbm; {(pure_kalman_e_k_give_N_minus_1_watt.mean())} W')
        print(f'kalman smoother MSE {watt2dbm(kalman_e_k_give_N_minus_1_watt.mean())} dbm; {(kalman_e_k_give_N_minus_1_watt.mean())} W')

        totalKalmanFilterIncrease = kalman_e_k_give_k_minus_1_watt.mean() - pure_kalman_e_k_give_k_minus_1_watt.mean()
        totalKalmanSmootherIncrease = kalman_e_k_give_N_minus_1_watt.mean() - pure_kalman_e_k_give_N_minus_1_watt.mean()
        print(f'kalman filter error increases by {totalKalmanFilterIncrease} [W] due to the unmodeled behavior; {totalKalmanFilterIncrease/pure_kalman_e_k_give_k_minus_1_watt.mean()*100} %')
        print(f'kalman smoother error increases by {totalKalmanSmootherIncrease} [W] due to the unmodeled behavior; {totalKalmanSmootherIncrease/pure_kalman_e_k_give_N_minus_1_watt.mean()*100} %')
        increaseOfLearned = learned_e_k_give_k_minus_1_watt.mean() - pure_kalman_e_k_give_k_minus_1_watt.mean()
        decreaseOfLearned = kalman_e_k_give_k_minus_1_watt.mean() - learned_e_k_give_k_minus_1_watt.mean()
        print(f'learned filter error increases by {increaseOfLearned} [W] (w.r.t kalman filter) due to the unmodeled behavior; {increaseOfLearned/pure_kalman_e_k_give_k_minus_1_watt.mean()*100} %')
        print(f'learned filter error is {decreaseOfLearned} [W] below the standard kalman filter')
        print(f'learned filter removed {decreaseOfLearned/totalKalmanFilterIncrease*100} % of the increase in error of the standard filter')

        print(f'the std of the error of the learned filter for a single run is {np.power(learned_e_k_give_k_minus_1, 2).sum(axis=2).sum(axis=0)[:, 0].std()} W')

if nIndependentTrainings > 1:
    print(f'The standard deviation between independent trainings is {np.asarray(learnedFilterErrorWattList).std()} watts')

plt.figure()
n_bins = 1000
n, bins, patches = plt.hist(watt2dbm(pure_kalman_e_k_give_k_minus_1_watt), n_bins, color='green', linestyle = 'dashed', density=True, histtype='step', cumulative=True)
n, bins, patches = plt.hist(watt2dbm(pure_kalman_e_k_give_N_minus_1_watt), n_bins, color='blue', linestyle = 'dashed', density=True, histtype='step', cumulative=True)

n, bins, patches = plt.hist(watt2dbm(kalman_e_k_give_k_minus_1_watt), n_bins, color='green', density=True, histtype='step', cumulative=True, label=r'Kalman filter')
n, bins, patches = plt.hist(watt2dbm(kalman_e_k_give_N_minus_1_watt), n_bins, color='blue', density=True, histtype='step', cumulative=True, label=r'Kalman smoother')
n, bins, patches = plt.hist(watt2dbm(learned_e_k_give_k_minus_1_watt), n_bins, color='orange', density=True, histtype='step', cumulative=True, label=r'learned filter')
plt.xlabel('dbm')
plt.title(r'CDF of estimation errors')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()
