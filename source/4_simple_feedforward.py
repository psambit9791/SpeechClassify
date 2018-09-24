import matplotlib.pyplot as plt
import numpy as np
import random

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

# Model Design

class FFSpeechMode(nn.Module):
    def __init__(self):
        super(FFSpeechMode, self).__init__()
        self.activationLayer = nn.PReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(299*13, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 8)
        self.fc5 = nn.Linear(8,3)
    
    def forward(self, x):
        x = self.activationLayer(self.fc1(x))
        #x = self.dropout(x)
        x = self.activationLayer(self.fc2(x))
        x = self.dropout(x)
        x = self.activationLayer(self.fc3(x))
        x = self.activationLayer(self.fc4(x))
        x = self.fc5(x)

        return x

# Training

random.seed(0)

def train_ffsm(model, x_tr, y_tr, x_v, y_v, criterion, optimizer, epoch_range, batch_size):
    losses = []
    v_losses = []
   
    for num_epoch in range(epoch_range):
        selected = random.sample(range(0, x_tr.shape[0]), batch_size)
        x = np.zeros((batch_size, 299*13))
        y = np.zeros((batch_size, 3))
        for i, feats in enumerate(selected):
            x[i, :] = x_tr[feats, :, :].flatten()
            y[i, :] = y_tr[feats, :]
        x = torch.from_numpy(x).float().cuda()
        y = torch.from_numpy(y).long().cuda()
        #---------FORWARD------------#
        out = model.forward(x)
        out = torch.tensor(out).float().cuda()
        loss = criterion(out, torch.max(y,1)[1])
        #---------BACKWARD------------#
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        v = random.sample(range(0, x_v.shape[0]), 8)
        xv = np.zeros((8, 299*13))
        yv = np.zeros((8, 3))
        for i, feats in enumerate(v):
            xv[i, :] = x_v[feats, :, :].flatten()
            yv[i, :] = y_v[feats, :]
        xv = torch.from_numpy(xv).float().cuda()
        yv = torch.from_numpy(yv).long().cuda()
        v_out = model.forward(xv)
        v_out = torch.tensor(v_out).float().cuda()
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
##################################################


mdl1 = FFSpeechMode().cuda()
mdl1.apply(init_weights)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0005
optimizer = torch.optim.Adam(
    mdl1.parameters(), lr=learning_rate, weight_decay=0.001)

epoch_range = 25000
batch_size = 48
losses, vlosses = train_ffsm(mdl1, X_train, y_train, X_val, y_val, 
                                 criterion, optimizer, epoch_range, batch_size)

###################################################

# Loss Plot

fig = plt.figure(figsize=(9, 5))
plt.title("Loss Curve for FeedForward Network", fontsize=14)
plt.plot(vlosses, label="val")
plt.plot(losses, label="train")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
fig.savefig(ROOT_PATH+"images/ffsm_performance.pdf", bbox_inches='tight')

# Testing

# Indicated by maximum energy

total = y_test.shape[0]
correct = 0

for i in range(X_test.shape[0]):
    data = X_test[i, :, :].flatten()
    #print(data)
    data = torch.from_numpy(data).float().cuda()
    true_out = y_test[i, :]
    mdl_out = mdl1.forward(data)
    out = (mdl_out.data).cpu().numpy()
    if np.argmax(out) == np.argmax(true_out):
        correct += 1
print("Accuracy = ", correct*100/total)

torch.save(mdl1, ROOT_PATH+'models/feedforward_speechmode.pt')
