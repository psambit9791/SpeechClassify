import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import signal
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data

torch.cuda.set_device(2)
torch.manual_seed(0)

ROOT_PATH = "../"

# Load datasets

X_train = np.load(ROOT_PATH+"numpy_ds/x_train.npy")
X_val = np.load(ROOT_PATH+"numpy_ds/x_val.npy")
X_test = np.load(ROOT_PATH+"numpy_ds/x_test.npy")

y_train = np.load(ROOT_PATH+"numpy_ds/y_train.npy")
y_val = np.load(ROOT_PATH+"numpy_ds/y_val.npy")
y_test = np.load(ROOT_PATH+"numpy_ds/y_test.npy")


######################################

# LSTM Model

class LSTMSpeechMode(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(LSTMSpeechMode, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, 
                             batch_first=True, num_layers=self.n_layers)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        
        out, hidden = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

INPUT_SIZE = 13
HIDDEN_SIZE = 100
OUTPUT_SIZE = 3
N_LAYERS = 2

random.seed(0)

def train_lstm(model, x_tr, y_tr, x_v, y_v, criterion, optimizer, epoch_range, batch_size):
    losses = []
    v_losses = []
   
    for num_epoch in range(epoch_range):
        selected = random.sample(range(0, x_tr.shape[0]), batch_size)
        x = np.zeros((batch_size, 299, 13))
        y = np.zeros((batch_size, 3))
        for i, feats in enumerate(selected):
            x[i, :, :] = x_tr[feats, :, :]
            y[i, :] = y_tr[feats, :]
        x = torch.from_numpy(x).float().cuda()
        y = torch.from_numpy(y).long().cuda()
        hidden = torch.randn(N_LAYERS, batch_size, 8)
        #---------FORWARD------------#
        out = model.forward(x)
        out = torch.tensor(out).float().cuda()
        out = out.view(N_LAYERS, batch_size, OUTPUT_SIZE)[-1]
        loss = criterion(out, torch.max(y,1)[1])
        #---------BACKWARD------------#
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        v = random.sample(range(0, x_v.shape[0]), 8)
        xv = np.zeros((8, 299, 13))
        yv = np.zeros((8, 3))
        for i, feats in enumerate(v):
            xv[i, :, :] = x_v[feats, :, :]
            yv[i, :] = y_v[feats, :]
        xv = torch.from_numpy(xv).float().cuda()
        yv = torch.from_numpy(yv).long().cuda()
        hidden_v = torch.randn(N_LAYERS, 8, 8)
        v_out = model.forward(xv)
        v_out = torch.tensor(v_out).float().cuda()
        v_out = v_out.view(N_LAYERS, 8, OUTPUT_SIZE)[-1]
        vloss = criterion(v_out, torch.max(yv, 1)[1])
        
        losses.append(loss.data[0])
        v_losses.append(vloss.data[0])
        if num_epoch%100 == 0:
            print('epoch [{}/{}], loss:{:.8f}, val:{:.8f}'.format(num_epoch, epoch_range, 
                                                              loss.data[0], vloss.data[0]))
    return losses, v_losses

def init_weights(layer):
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
        nn.init.xavier_normal_(layer.weight)

epoch_range = 5000
batch_size = 48
learning_rate = 0.0001

lstm_1 = LSTMSpeechMode(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, N_LAYERS).cuda()
lstm_1.apply(init_weights)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    lstm_1.parameters(), lr=learning_rate)

losses_1, vloss = train_lstm(lstm_1, X_train, y_train, X_val, y_val,
                                 criterion, optimizer, epoch_range, batch_size)

fig = plt.figure(figsize=(9, 5))
plt.title("Loss Curve for LSTM-RNN", fontsize=14)
plt.plot(vloss, label="val")
plt.plot(losses_1, label="train")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
fig.savefig(ROOT_PATH+"images/lstm_performance.pdf", bbox_inches='tight')

#############################

# Testing

total = y_test.shape[0]
correct = 0

for i in range(X_test.shape[0]):
    data = X_test[i, :, :]
    data = torch.from_numpy(data).float().cuda()
    data = data.view(1, 299, 13)
    true_out = y_test[i, :]
    mdl_out = lstm_1.forward(data)
    mdl_out = mdl_out.view(N_LAYERS, 3)[-1]
    out = (mdl_out.data).cpu().numpy()
    if np.argmax(out) == np.argmax(true_out):
        correct += 1
print("Accuracy = ", correct*100/total)

torch.save(lstm_1, ROOT_PATH+'models/lstm_speechmode.pt')
