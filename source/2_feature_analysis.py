from scipy.io import wavfile
import pandas as pd
import python_speech_features as psf
import numpy as np
import matplotlib.pyplot as plt

mfcc_feat = []
ROOT_PATH = "../"

# Sample MFCC for plotting

fs, data = wavfile.read(ROOT_PATH+'data/rap/sing002/sing002_1.wav')
mfcc_feat.append(psf.mfcc(data,fs, winfunc=np.hamming))

fs, data = wavfile.read(ROOT_PATH+'data/singing/sing003/sing003_1.wav')
mfcc_feat.append(psf.mfcc(data,fs, winfunc=np.hamming))

fs, data = wavfile.read(ROOT_PATH+'data/speech/01b/01bo030x.wav')
mfcc_feat.append(psf.mfcc(data,fs, winfunc=np.hamming))

fig1 = plt.figure(figsize=(20, 5))
title_list = ['Rap', 'Singing', 'Speech']
fig1.suptitle('MFCC Visualisation Sample', fontsize=19)
for i, title in enumerate(title_list):
    plt.subplot(131+i)
    plt.title(title, fontsize=15)
    plt.plot(mfcc_feat[i].T)
    plt.xlabel('MFC Coefficients', fontsize=12)
    plt.ylabel('Coefficicent Value', fontsize=12)
    plt.grid()
plt.show()
fig1.savefig(ROOT_PATH+"images/mfcc_sample.pdf", bbox_inches='tight')


###################################################

# Data-structure updated with MFCC

song_df = pd.read_hdf(ROOT_PATH+'song_df.h5', 'song_df')

mfcc_feat = []

for i in range(len(song_df.index)):
    mfcc_feat.append(psf.mfcc(song_df.iloc[i].Data,song_df.iloc[i].Freq, winfunc=np.hamming))

song_df['MFCC'] = pd.Series(mfcc_feat, index=song_df.index)

song_df.to_hdf(ROOT_PATH+'song_df.h5', key='song_df')

####################################################


# Train and Test Data Generation


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

song_df = pd.read_hdf(ROOT_PATH+'song_df.h5', 'song_df')
X = song_df.MFCC.values
y = song_df.Type.values

onehot_encoder = OneHotEncoder(sparse=False)
y = onehot_encoder.fit_transform(y.reshape(len(y), 1))
print(y)

X_new = np.zeros((1440, 299, 13))
for i,d in enumerate(X):
    X_new[i,:,:] = d[:, :]
print(X_new.shape)
X = X_new

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

# Saving the generated datasets

# Training set
np.save(ROOT_PATH+"numpy_ds/x_train", X_train)
np.save(ROOT_PATH+"numpy_ds/y_train", y_train)

# Validation set
np.save(ROOT_PATH+"numpy_ds/x_val", X_val)
np.save(ROOT_PATH+"numpy_ds/y_val", y_val)

# Test set
np.save(ROOT_PATH+"numpy_ds/x_test", X_test)
np.save(ROOT_PATH+"numpy_ds/y_test", y_test)

# Test data distribution - check for balance of test set

list_of_y = {0:0, 1:0, 2:0}
for i in range(y_test.shape[0]):
    num = 0
    for j in range(3):
        num+=y_test[i,j]*j
    list_of_y[num] += 1
print("Distribution of Examples: ", list_of_y)

total = list_of_y[0]+list_of_y[1]+list_of_y[2]
perc_of_y = {0:list_of_y[0]*100/total, 1:list_of_y[1]*100/total, 2:list_of_y[2]*100/total}
print("Percentage Distribution: ", perc_of_y)
