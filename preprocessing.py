import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

X_train_npy = np.load('X_train.npy')
X_test_npy = np.load('X_test.npy')
y_train_npy = np.load('y_train.npy')
y_test_npy = np.load("y_test.npy")

scalers = {}
X_train_scaled = np.zeros_like(X_train_npy)
X_test_scaled = np.zeros_like(X_test_npy)

for i in range(X_train_npy.shape[2]):
    scaler = StandardScaler()
    X_train_scaled[:,:,i] = scaler.fit_transform(X_train_npy[:,:,i])
    X_test_scaled[:,:,i] = scaler.transform(X_test_npy[:,:,i])

seq_len = 50
def create_sequences(data, label, seq_len):
    # data:  (time, feature, channel)
    # label: (time)
    X, y = [], []
    for i in range(0,len(data) - seq_len+1):
        X.append(data[i:i+seq_len])
        y.append(label[i+seq_len-1])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(data = X_train_scaled, label = y_train_npy, seq_len = seq_len)
X_test, y_test = create_sequences(data = X_test_scaled, label = y_test_npy, seq_len = seq_len)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)

X_train_tensor = torch.tensor(X_train, dtype = torch.float32)
X_test_tensor = torch.tensor(X_test, dtype = torch.float32)
y_train_tensor = torch.tensor(y_train, dtype = torch.long)
y_test_tensor = torch.tensor(y_test, dtype = torch.long)

train_set = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
test_set = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

BATCH_SIZE = 32
train_dataLoader = torch.utils.data.DataLoader(dataset = train_set, batch_size = BATCH_SIZE, shuffle = True)
test_dataLoader = torch.utils.data.DataLoader(dataset = test_set, batch_size = BATCH_SIZE, shuffle = False)

