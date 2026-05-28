import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def getStable(data, w=1440):
    trend = pd.DataFrame(data).rolling(w, center=True).median().values
    stable = data - trend
    return data[w // 2:-w // 2, :], stable[w // 2:-w // 2, :]


def getData(path='./dataset/', dataset='SWaT', period=1440, train_rate=0.8):
    init_data = np.load(path + dataset + '/' + dataset + '_train_data.npy')
    test_data = np.load(path + dataset + '/' + dataset + '_test_data.npy')
    test_label = np.load(path + dataset + '/' + dataset + '_test_label.npy')

    scaler = StandardScaler()
    scaler.fit(init_data)
    init_data = pd.DataFrame(scaler.transform(init_data)).fillna(0).values
    test_data = pd.DataFrame(scaler.transform(test_data)).fillna(0).values

    init_data, init_stable = getStable(init_data, w=period)
    init_label = np.zeros((len(init_data), 1))
    test_stable = np.zeros_like(test_data)

    train_data = init_data[:int(train_rate * len(init_data)), :]
    train_stable = init_stable[:int(train_rate * len(init_stable)), :]
    train_label = init_label[:int(train_rate * len(init_label)), :]

    valid_data = init_data[int(train_rate * len(init_data)):, :]
    valid_stable = init_stable[int(train_rate * len(init_stable)):, :]
    valid_label = init_label[int(train_rate * len(init_label)):, :]

    data = {
        'train_data': train_data, 'train_stable': train_stable, 'train_label': train_label,
        'valid_data': valid_data, 'valid_stable': valid_stable, 'valid_label': valid_label,
        'init_data': init_data, 'init_stable': init_stable, 'init_label': init_label,
        'test_data': test_data, 'test_stable': test_stable, 'test_label': test_label
    }

    return data
