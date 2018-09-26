import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import signal
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT_PATH = "../"
song_df = pd.read_hdf(ROOT_PATH+'song_df.h5', 'song_df')
X = song_df.MFCC.values
y = song_df.Type.values

X_new = np.zeros((1440, 299*13))
for i,d in enumerate(X):
    X_new[i,:] = d[:, :].flatten()
print(X_new.shape)
X = X_new

#################################

# PCA and t-SNE

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

print(pca.explained_variance_ratio_)
print(np.sum(pca.explained_variance_ratio_))

#### First two components only explain 11% of the data which is not sufficient for clustering
#### Using T-SNE for further analysis. Reducing components for sufficient representation

# Reducing Dimensionality to a sufficient degree of explainability
pca = PCA(n_components=200)
pca_result = pca.fit_transform(X)

print(np.sum(pca.explained_variance_ratio_))

n_sne = 1440

def get_tsne_results(pca_result, perp):
    tsne = TSNE(n_components=2, verbose=1, perplexity=perp, n_iter=5000)
    tsne_results = tsne.fit_transform(pca_result)
    return tsne_results


perp_list = [20, 50, 100, 500, 800, 1000]
for i in perp_list:
    fig = plt.figure(figsize=(9, 9))
    tsne_results = get_tsne_results(X, i)
    plt.title("T-SNE Clustering of the MFCC features with perplexity = "+ str(i), fontsize=15)
    plt.scatter(tsne_results[:, 0], tsne_results[:,1], c=y, s=15)
    plt.show()
    fig.savefig(ROOT_PATH+"images/tsne_cluster_"+str(i)+".pdf", bbox_inches='tight')

##############################################


# KNN Test

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn.fit(X_train, y_train)

from sklearn import metrics

y_pred = knn.predict(X_test)
print("Accuracy = ",metrics.accuracy_score(y_test, y_pred)*100)

################################################


# SVM Test

from sklearn import svm

# Creates a C-SVC
clf = svm.LinearSVC()

# Trains the model with training data
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy = ",metrics.accuracy_score(y_test, y_pred)*100)
