import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

class CirclesData:
    def __init__(self):
        # Grid
        x1,x2 = np.meshgrid(np.arange(-2, 2, 0.1), np.arange(-2, 2, 0.1))
        self._Xgrid = np.array([x1.flatten(), x2.flatten()]).T.astype('float32')

        X, y = make_circles(n_samples=1000, noise=0.09, random_state=42, factor=0.75)
        self._Xtrain, self._Xtest, self._Ytrain, self._Ytest = train_test_split(X.astype('float32'),y,test_size=0.2,random_state=42)

        self._Xgrid_th = torch.from_numpy(self._Xgrid)
        self._Xtrain_th = torch.from_numpy(self._Xtrain)
        self._Xtest_th = torch.from_numpy(self._Xtest)
        self._Ytrain_th = F.one_hot(torch.from_numpy(self._Ytrain))
        self._Ytest_th = F.one_hot(torch.from_numpy(self._Ytest))

        self.loss_train = []
        self.loss_test = []
        self.acc_train = []
        self.acc_test = []

    def __getattr__(self, key):
        if key == "Xgrid": return self._Xgrid_th
        if key == "Xtrain": return self._Xtrain_th
        if key == "Xtest": return self._Xtest_th
        if key == "Ytrain": return self._Ytrain_th
        if key == "Ytest": return self._Ytest_th
        return None

    def plot_loss(self, loss_train, loss_test, acc_train, acc_test):
        self.loss_train.append(loss_train)
        self.loss_test.append(loss_test)
        self.acc_train.append(acc_train)
        self.acc_test.append(acc_test)
        plt.figure(3)
        plt.clf()
        plt.subplot(1,2,1)
        plt.plot(np.array(self.acc_train), label="acc. train")
        plt.plot(np.array(self.acc_test), label="acc. test")
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(np.array(self.loss_train), label="loss train")
        plt.plot(np.array(self.loss_test), label="loss test")
        plt.legend()
        plt.show()

def plot_data(data):
    plt.figure(1, figsize=(5,5))
    plt.plot(data.Xtrain[data.Ytrain[:,0] == 1,0], data.Xtrain[data.Ytrain[:,0] == 1,1], 'bo', label="Train")
    plt.plot(data.Xtrain[data.Ytrain[:,1] == 1,0], data.Xtrain[data.Ytrain[:,1] == 1,1], 'ro')
    plt.plot(data.Xtest[data.Ytest[:,0] == 1,0], data.Xtest[data.Ytest[:,0] == 1,1], 'b+', label="Test")
    plt.plot(data.Xtest[data.Ytest[:,1] == 1,0], data.Xtest[data.Ytest[:,1] == 1,1], 'r+')
    plt.legend()
    plt.show()

def plot_data_with_grid(data, Ygrid, title=""):
    plt.figure(2)
    Ygrid = Ygrid[:,1]
    plt.clf()
    plt.imshow(np.reshape(Ygrid, (40,40)))
    plt.plot(data.Xtrain[data.Ytrain[:,0] == 1,0]*10+20, data.Xtrain[data.Ytrain[:,0] == 1,1]*10+20, 'bo', label="Train")
    plt.plot(data.Xtrain[data.Ytrain[:,1] == 1,0]*10+20, data.Xtrain[data.Ytrain[:,1] == 1,1]*10+20, 'ro')
    plt.plot(data.Xtest[data.Ytest[:,0] == 1,0]*10+20, data.Xtest[data.Ytest[:,0] == 1,1]*10+20, 'b+', label="Test")
    plt.plot(data.Xtest[data.Ytest[:,1] == 1,0]*10+20, data.Xtest[data.Ytest[:,1] == 1,1]*10+20, 'r+')
    plt.xlim(0,39)
    plt.ylim(0,39)
    plt.clim(0.3,0.7)
    plt.title(title)
    plt.draw()
    plt.pause(1e-3)