import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
import torch.optim as optim
import numpy as np

class MeasurementsDataset(Dataset):
    def __init__(self, sysModel, nTime, nSeries, transform=None):

        self.completeDataSet = self.getBatch(sysModel, nTime, nSeries)

        self.transform = transform

    def getBatch(self, sysModel, nTimePoints, nSeries):
        device = 'cpu'
        tilde_z, tilde_x, processNoises, measurementNoises = GenMeasurements(nTimePoints, nSeries, sysModel, startAtZero=False, dp=False)  # z: [N, nSeries, dim_z]

        # unmodeled behavior:
        tr_Q = np.trace(sysModel['Q'])
        u = np.zeros_like(tilde_x)
        #u[:, :, 0, 0] = np.power(tilde_x[:, :, 1, 0], 1)
        u[:, :, 0, 0] = np.power(tilde_x[:, :, 1, 0], 2) * (tilde_x[:, :, 1, 0] > 0)
        u[:, :, 1, 0] = np.power(tilde_x[:, :, 0, 0], 2) * (tilde_x[:, :, 0, 0] > 0)
        #u_mean_watt = np.power(u, 2).sum(axis=2).flatten().mean()
        #alpha = 0
        #u = alpha*u
        u_mean_watt = np.power(u, 2).sum(axis=2).flatten().mean()
        gamma_wrt_tr_Q = u_mean_watt/tr_Q
        print(f'$\gamma/tr(Q)$ = {gamma_wrt_tr_Q}')
        np.power(np.matmul(np.transpose(sysModel['H']), u), 2).sum(axis=2).flatten().mean()

        z = tilde_z + np.matmul(np.transpose(sysModel['H']), u)

        tilde_z, tilde_x, processNoises, measurementNoises = torch.tensor(tilde_z, dtype=torch.float, device=device), torch.tensor(tilde_x, dtype=torch.float, device=device), torch.tensor(processNoises, dtype=torch.float, device=device), torch.tensor(measurementNoises, dtype=torch.float, device=device)
        z = torch.tensor(z, dtype=torch.float, device=device)

        return {'z': z, 'tilde_z': tilde_z, 'tilde_x': tilde_x}

    def __len__(self):
        return self.completeDataSet['tilde_z'].shape[1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'z': self.completeDataSet['z'][:, idx], 'tilde_z': self.completeDataSet['tilde_z'][:, idx], 'x': self.completeDataSet['tilde_x'][:, idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

# class definition
class RNN_Filter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(RNN_Filter, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # setup RNN layer
        #self.Adv_rnn = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers)
        self.Filter_rnn = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # setup output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, z):
        # z_k of shape: [N, batchSize, dim_z]
        controlHiddenDim, hidden = self.Filter_rnn(z[:, :, :, 0])
        # controlHiddenDim shape: [N, batchSize, hidden_dim]
        hat_x_k_plus_1_given_k = self.linear(controlHiddenDim)

        return hat_x_k_plus_1_given_k

def trainModel(model, pytorchEstimator, trainLoader, validationLoader):
    filter_P_init = pytorchEstimator.theoreticalBarSigma.cpu().numpy()  # filter @ time-series but all filters have the same init
    criterion = nn.MSELoss()
    lowThrLr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, threshold=1e-6)
    # moving model to cuda:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.to(device)
    x_0_train_given_minus_1 = torch.zeros((1, trainLoader.batch_size, pytorchEstimator.dim_x), dtype=torch.float, device=device)
    x_0_validation_given_minus_1 = torch.zeros((1, validationLoader.batch_size, pytorchEstimator.dim_x), dtype=torch.float, device=device)
    # training and saving the model when validation is best:
    print('start training')
    min_valid_loss = np.inf
    epoch = -1
    while True:
        epoch += 1
        train_loss = 0.0
        model.train()

        for i_batch, sample_batched in enumerate(trainLoader):
            # print(f'starting epoch {epoch}, batch {i_batch}')
            optimizer.zero_grad()

            z = sample_batched["z"].transpose(1, 0)
            z = z.to(device)
            currentBatchSize = z.shape[1]
            hat_x_k_plus_1_given_k = model(z)
            learned_tilde_x_est_f = torch.cat((x_0_train_given_minus_1[:, :currentBatchSize], hat_x_k_plus_1_given_k[:-1]), dim=0)[:, :, :, None]

            # estimator init values:
            filterStateInit = np.matmul(np.linalg.cholesky(filter_P_init), np.random.randn(currentBatchSize, pytorchEstimator.dim_x, 1))
            filterStateInit = torch.tensor(filterStateInit, dtype=torch.float, requires_grad=False, device=device).contiguous()
            # filterStateInit = tilde_x[0]  This can be used if the initial state is known

            # kalman filter on z:
            tilde_x_est_f, tilde_x_est_s = pytorchEstimator(z, filterStateInit)
            loss = criterion(learned_tilde_x_est_f, tilde_x_est_s)

            loss.backward()
            optimizer.step()  # parameter update

            train_loss += loss.item()

        scheduler.step(train_loss)

        validation_loss = 0.0
        model.eval()
        for i_batch, sample_batched in enumerate(validationLoader):
            z = sample_batched["z"].transpose(1, 0)
            z = z.to(device)
            currentBatchSize = z.shape[1]
            hat_x_k_plus_1_given_k = model(z)
            learned_tilde_x_est_f = torch.cat((x_0_validation_given_minus_1[:, :currentBatchSize], hat_x_k_plus_1_given_k[:-1]), dim=0)[:, :, :, None]

            # estimator init values:
            filterStateInit = np.matmul(np.linalg.cholesky(filter_P_init), np.random.randn(currentBatchSize, pytorchEstimator.dim_x, 1))
            filterStateInit = torch.tensor(filterStateInit, dtype=torch.float, requires_grad=False, device=device).contiguous()
            # filterStateInit = tilde_x[0]  This can be used if the initial state is known

            # kalman filter on z:
            tilde_x_est_f, tilde_x_est_s = pytorchEstimator(z, filterStateInit)
            loss = criterion(learned_tilde_x_est_f, tilde_x_est_s)
            validation_loss += loss.item()

        validation_loss = validation_loss/(i_batch+1)
        if min_valid_loss > validation_loss:
            print(f'epoch {epoch}, Validation loss Decreased({min_valid_loss:.6f}--->{validation_loss:.6f}); lr: {scheduler._last_lr[-1]}')
            min_valid_loss = validation_loss
            torch.save(model.module.state_dict(), 'saved_modelFilter.pt')

        if scheduler._last_lr[-1] < lowThrLr:
            print(f'Stoping optimization due to learning rate of {scheduler._last_lr[-1]}')
            break

# class definition
class Pytorch_filter_smoother_Obj(nn.Module):
    def __init__(self, sysModel, enableSmoothing = True, useCuda=True):
        super(Pytorch_filter_smoother_Obj, self).__init__()
        self.useCuda = useCuda
        self.enableSmoothing = enableSmoothing
        # filter_P_init: [1, batchSize, dim_x, dim_x] is not in use because this filter works from the start on the steady-state-gain
        # filterStateInit: [1, batchSize, dim_x, 1]
        # z: [N, batchSize, dim_z, 1]
        F, H, Q, R = sysModel["F"], sysModel["H"], sysModel["Q"], sysModel["R"]

        self.dim_x, self.dim_z = F.shape[0], H.shape[1]

        theoreticalBarSigma = solve_discrete_are(a=np.transpose(F), b=H, q=Q, r=R)
        Ka_0 = np.dot(theoreticalBarSigma, np.dot(H, np.linalg.inv(np.dot(np.transpose(H), np.dot(theoreticalBarSigma, H)) + R)))  # first smoothing gain
        K = np.dot(F, Ka_0)  # steadyKalmanGain
        tildeF = F - np.dot(K, np.transpose(H))
        Sint = np.matmul(np.linalg.inv(np.matmul(F, theoreticalBarSigma)), K)
        thr = 1e-20 * np.abs(tildeF).max()

        DeltaFirstSample = np.dot(Ka_0, np.dot(np.transpose(H), theoreticalBarSigma))
        theoreticalSmoothingFilteringDiff = solve_discrete_lyapunov(a=np.dot(theoreticalBarSigma, np.dot(np.transpose(tildeF), np.linalg.inv(theoreticalBarSigma))) , q=DeltaFirstSample)
        theoreticalSmoothingSigma = theoreticalBarSigma - theoreticalSmoothingFilteringDiff

        A_N = solve_discrete_lyapunov(a=tildeF, q=np.eye(self.dim_x))
        A_N_directSum = calcDeltaR(a=tildeF, q=np.eye(self.dim_x))
        assert np.abs(A_N_directSum - A_N).max() < 1e-5
        normalizedNoKnowledgePlayerContribution = np.trace(np.matmul(np.dot(H, np.transpose(K)), np.matmul(np.transpose(A_N), np.dot(K, np.transpose(H)))))

        smootherRecursiveGain = np.matmul(theoreticalBarSigma, np.matmul(np.transpose(tildeF), np.linalg.inv(theoreticalBarSigma)))
        smootherGain = np.linalg.inv(F) - smootherRecursiveGain

        '''
        print(f'The eigenvalues of tildeF: {np.linalg.eig(tildeF)[0]}')
        print(f'The eigenvalues of KH\': {np.linalg.eig(np.matmul(K, np.transpose(H)))[0]}')
        print(f'The eigenvalues of smootherRecursiveGain: {np.linalg.eig(smootherRecursiveGain)[0]}')
        print(f'The eigenvalues of smootherGain: {np.linalg.eig(smootherGain)[0]}')
        '''
        # stuff to cuda:
        if self.useCuda:
            self.F = torch.tensor(F, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.tildeF = torch.tensor(tildeF, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.tildeF_transpose = torch.tensor(tildeF.transpose(), dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.K = torch.tensor(K, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.H = torch.tensor(H, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.Sint = torch.tensor(Sint, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.H_transpose = torch.tensor(H.transpose(), dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.thr = torch.tensor(thr, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.theoreticalBarSigma = torch.tensor(theoreticalBarSigma, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.theoreticalSmoothingSigma = torch.tensor(theoreticalSmoothingSigma, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.smootherRecursiveGain = torch.tensor(smootherRecursiveGain, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.smootherGain = torch.tensor(smootherGain, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.Ka_0 = torch.tensor(Ka_0, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.normalizedNoKnowledgePlayerContribution = torch.tensor(normalizedNoKnowledgePlayerContribution, dtype=torch.float, requires_grad=False).contiguous().cuda()
        else:
            self.F = torch.tensor(F, dtype=torch.float, requires_grad=False).contiguous()
            self.tildeF = torch.tensor(tildeF, dtype=torch.float, requires_grad=False).contiguous()
            self.tildeF_transpose = torch.tensor(tildeF.transpose(), dtype=torch.float, requires_grad=False).contiguous()
            self.K = torch.tensor(K, dtype=torch.float, requires_grad=False).contiguous()
            self.H = torch.tensor(H, dtype=torch.float, requires_grad=False).contiguous()
            self.Sint = torch.tensor(Sint, dtype=torch.float, requires_grad=False).contiguous()
            self.H_transpose = torch.tensor(H.transpose(), dtype=torch.float, requires_grad=False).contiguous()
            self.thr = torch.tensor(thr, dtype=torch.float, requires_grad=False).contiguous()
            self.theoreticalBarSigma = torch.tensor(theoreticalBarSigma, dtype=torch.float, requires_grad=False).contiguous()
            self.theoreticalSmoothingSigma = torch.tensor(theoreticalSmoothingSigma, dtype=torch.float, requires_grad=False).contiguous()
            self.smootherRecursiveGain = torch.tensor(smootherRecursiveGain, dtype=torch.float, requires_grad=False).contiguous()
            self.smootherGain = torch.tensor(smootherGain, dtype=torch.float, requires_grad=False).contiguous()
            self.Ka_0 = torch.tensor(Ka_0, dtype=torch.float, requires_grad=False).contiguous()
            self.normalizedNoKnowledgePlayerContribution = torch.tensor(normalizedNoKnowledgePlayerContribution, dtype=torch.float, requires_grad=False).contiguous()

    def forward(self, z, filterStateInit):
        # z, filterStateInit are cuda

        # filtering
        N, batchSize = z.shape[0], z.shape[1]

        if self.useCuda:
            hat_x_k_plus_1_given_k = torch.zeros(N, batchSize, self.dim_x, 1, dtype=torch.float, requires_grad=False).cuda()  #  hat_x_k_plus_1_given_k is in index [k+1]
        else:
            hat_x_k_plus_1_given_k = torch.zeros(N, batchSize, self.dim_x, 1, dtype=torch.float, requires_grad=False)  #  hat_x_k_plus_1_given_k is in index [k+1]

        hat_x_k_plus_1_given_k[0] = filterStateInit

        hat_x_k_plus_1_given_k[1] = torch.matmul(self.tildeF, hat_x_k_plus_1_given_k[0]) + torch.matmul(self.K, z[0])
        K_dot_z = torch.matmul(self.K, z)
        for k in range(N - 1):
            hat_x_k_plus_1_given_k[k + 1] = torch.matmul(self.tildeF, hat_x_k_plus_1_given_k[k].clone()) + K_dot_z[k]

        # smoothing:
        if self.useCuda:
            hat_x_k_given_N = torch.zeros(N, batchSize, self.dim_x, 1, dtype=torch.float, requires_grad=False).cuda()
        else:
            hat_x_k_given_N = torch.zeros(N, batchSize, self.dim_x, 1, dtype=torch.float, requires_grad=False)

        if self.enableSmoothing:
            bar_z_N_minus_1 = z[N - 1] - torch.matmul(self.H_transpose, hat_x_k_plus_1_given_k[N - 1]) # smoother init val
            hat_x_k_given_N[N-1] = hat_x_k_plus_1_given_k[N-1] + torch.matmul(self.Ka_0, bar_z_N_minus_1)
            filteringInput = torch.matmul(self.smootherGain, hat_x_k_plus_1_given_k)
            for k in range(N-2, -1, -1):
                hat_x_k_given_N[k] = torch.matmul(self.smootherRecursiveGain, hat_x_k_given_N[k+1].clone()) + filteringInput[k+1]#torch.matmul(self.smootherGain, hat_x_k_plus_1_given_k[k+1])

        #  x_est_f, x_est_s =  hat_x_k_plus_1_given_k, hat_x_k_given_N - these are identical values

        return hat_x_k_plus_1_given_k, hat_x_k_given_N

def dbm2var(x_dbm):
    return np.power(10, np.divide(x_dbm - 30, 10))

def volt2dbm(x_volt):
    return 10*np.log10(np.power(x_volt, 2)) + 30

def volt2dbW(x_volt):
    return 10*np.log10(np.power(x_volt, 2))

def volt2db(x_volt):
    return 10*np.log10(np.power(x_volt, 2))

def watt2dbm(x_volt):
    return 10*np.log10(x_volt) + 30

def watt2db(x_volt):
    return 10*np.log10(x_volt)

def GenMeasurements(N, batchSize, sysModel, startAtZero=False, dp=True):
    F, H, Q, R = sysModel["F"], sysModel["H"], sysModel["Q"], sysModel["R"]
    dim_x, dim_z = F.shape[0], H.shape[1]
    # generate state
    x, z = np.zeros((N, batchSize, dim_x, 1)), np.zeros((N, batchSize, dim_z, 1))

    if startAtZero:
        x[0] = np.matmul(np.linalg.cholesky(Q), np.zeros((batchSize, dim_x, 1)))
    else:
        x[0] = np.matmul(np.linalg.cholesky(Q), np.random.randn(batchSize, dim_x, 1))

    processNoises = np.matmul(np.linalg.cholesky(Q), np.random.randn(N, batchSize, dim_x, 1))
    measurementNoises = np.matmul(np.linalg.cholesky(R), np.random.randn(N, batchSize, dim_z, 1))

    if dp: print(f'amount of energy into the system is {watt2dbm(np.mean(np.power(np.linalg.norm(processNoises[:,0:1], axis=2, keepdims=True), 2), axis=0)[0,0,0])} dbm')

    for i in range(1, N):
        x[i] = np.matmul(F, x[i - 1]) + processNoises[i - 1]

    if dp: print(f'amount of energy out from the system is {watt2dbm(np.mean(np.power(np.linalg.norm(x[:,0:1], axis=2, keepdims=True), 2), axis=0)[0,0,0])} dbm')

    z = np.matmul(H.transpose(), x) + measurementNoises

    return z, x, processNoises, measurementNoises

def calcDeltaR(a, q):
    dim_x = a.shape[0]
    tildeR = np.zeros((dim_x, dim_x))
    thr = 1e-20 * np.abs(a).max()
    maxValAboveThr = True
    k = 0
    while maxValAboveThr:
        a_k = np.linalg.matrix_power(a, k)
        summed = np.dot(a_k, np.dot(q, np.transpose(a_k)))
        tildeR = tildeR + summed
        k+=1
        if np.abs(summed).max() < thr:
            break
    return tildeR