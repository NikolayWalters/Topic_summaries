# self contained PCA demo
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from timeit import default_timer as timer
import numpy as np

#Load dataset
iris = load_iris()
X = iris.data
y = iris.target

#Perform train/test split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=4)
#Scale the data:
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Define PCA transform:
n_components=2 #Number of Principal Components (PCs) to use
pca = PCA(n_components)

#Learn PCA transform on training data:
pca.fit(X_train)

#Apply transformation on both the training data & the test data:
Z_train = pca.transform(X_train)
Z_test = pca.transform(X_test)

#Fit 2 classifiers: one on raw data, one on transformed data
#Create the model objects:
start_tr_raw = timer()
clf_raw = LogisticRegression()
end_tr_raw = timer()
time_tr_raw = end_tr_raw -  start_tr_raw

start_tr_transformed = timer()
clf_transformed = LogisticRegression()
end_tr_transformed = timer()
time_tr_transformed = end_tr_transformed -  start_tr_transformed

#Fit the 2 models on the training data (one on raw, the other on transformed)
#--keep track of time:
clf_raw.fit(X_train, y_train)
clf_transformed.fit(Z_train, y_train)

#Evaluate the 2 models on the test data (one on raw, the other on transformed):
acc_raw = clf_raw.score(X_test, y_test)
acc_transformed = clf_transformed.score(Z_test, y_test)

print('Test set accuracy using original features: '+str(acc_raw))
print('Test set accuracy using PCA-transformed features (first '+str(n_components)+' PCs): '+str(acc_transformed))

print('Training time for classifier using lower-dimensional features was '+
      str(np.round(100*(time_tr_raw - time_tr_transformed)/ time_tr_raw))+'% faster.')



# PCA reconstruction demo
from sklearn.metrics import mean_squared_error

#Choose a datapoint to shw before & after transformatin:
datapoint_index = 5

#Choose number of PCs to retain (1-4)
n_components = 2

#Apply transform on full data without standardizing
pca = PCA(n_components)
Z = pca.fit_transform(X)#applies transform to the same data it learns it on (X)

print('Datapoint #'+str(datapoint_index))
print('In original space:')
print(X[datapoint_index, :])
print('In transformed space:')
print(Z[datapoint_index, :])
print('-------------------------------------------------')

#Reconstruct the data in the original space from their projection:
X_rec = pca.inverse_transform(Z)

#Calculating the reconstruction error of the specified datapoint: 
rec_error = mean_squared_error(X[datapoint_index, :], X_rec[datapoint_index, :])

print('Reconstructed in original from transformed space:')
print(X_rec[datapoint_index, :])
print('Reconstruction error (quadratic) on Datapoint #'+str(datapoint_index)+' is: ')
print(rec_error)


# Reconstruction error whole dataset to pick No. of PCs
from sklearn.metrics import mean_squared_error

n_components_to_keep = [1,2,3,4]

avg_reconstruction_error = []
for i in n_components_to_keep:

    #Apply transform on full data without standardizing
    pca = PCA(n_components=i)
    Z = pca.fit_transform(X)
    
    X_rec = pca.inverse_transform(Z)

    avg_reconstruction_error.append(mean_squared_error(X, X_rec))
    

fig = plt.figure(figsize = (9,6))
plt.plot(n_components_to_keep, avg_reconstruction_error)
plt.axhline(0, color='k', linestyle='--', label='Lossless')
plt.axhline(0.01, color='r', linestyle='--', label='Tol = 0.01')
plt.xlabel('# PCs retained')
plt.ylabel('Avg. Reconstruction Error')
plt.legend()
plt.show()


# variance explained per PC
#Choose number of PCs to retain (1-4)
n_components = 4

#Apply transform on full data without standardizing
pca = PCA(n_components)
Z = pca.fit_transform(X)

ratio_var_explained = pca.explained_variance_ratio_

print('Explained variation per principal component:')
for i in range(len(ratio_var_explained)):
      print('PC'+str(i+1)+': '+str(ratio_var_explained[i]))


# and plot it
import numpy as np

yy = ratio_cum_var_explained = np.cumsum(ratio_var_explained)*100

xx = list(range(1, len(ratio_cum_var_explained)+1))

fig = plt.figure(figsize = (9,6))
plt.plot(xx, yy)
plt.axhline(100, color='k', linestyle='--', label = 'Total Variance Explained')
plt.axhline(99, color='r', linestyle='--', label = '99% Variance Explained')
plt.legend()
plt.xlabel('# PCs retained')
plt.ylabel('% of variance explained')
plt.show()


# pca visualisation
import matplotlib.pyplot as plt

#Select number of PCs to use:
dim_new = 4

#Scale & transform data (full dataset) via PCA:
scaler = StandardScaler()
Z = PCA(n_components=dim_new).fit_transform(scaler.fit_transform(X))

#Plot 2-d scatterplots of the original data for all pairs of features:
fig = plt.figure(figsize = (10,18))
for i in range(dim):
    for j in range(dim):
        if i<j:
            x_min, x_max = Z[:, i].min() - 0.5, Z[:, i].max() + 0.5
            y_min, y_max = Z[:, j].min() - 0.5, Z[:, j].max() + 0.5

            # Plot the data
            plt.subplot(dim, dim, i*(dim-1)+j+1)
            plt.scatter(Z[:, i], Z[:, j], c=y, cmap=plt.cm.Set1, edgecolor="k")
            plt.xlabel("PC "+str(i+1))
            plt.ylabel("PC "+str(j+1))

            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
plt.subplots_adjust(wspace=0.8)
plt.suptitle('Scatterplots of the projected data for all pairs of PCs -- '+
             'Classes: red = setosa, gray = versicolor, orange = virginica')
plt.show()


