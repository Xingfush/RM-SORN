# This file provide some helper function to analyse the results.
import numpy as np
import copy
from matplotlib import pyplot as plt
from scipy import linalg as LA
from scipy.stats import variation
from collections import Counter

def get_ConnFraction(M):
    """
    Get the conncetion fraction of weight matrix.
    :param M: boolean array
    :return: float
    """
    return np.sum(M)/(M.shape[0]*M.shape[1])

def plot_ConnDistribution(W):
    """
    Plot the weight distribution of synapses.
    """
    eps = 10e-3
    weight = W[W>eps]
    plt.figure()
    plt.hist(weight,bins=100,density=1,alpha=0.6)
    plt.xlabel('Weights Distribution')
    plt.ylabel('Fraction')
    plt.show()

def get_PCAFraction(W,k=5):
    """
    Compute the fraction of top-k principle component of weights.
    """
    w = copy.copy(W)
    w = w - np.mean(w,axis=0)
    cov = np.cov(w,rowvar=False)
    evals,_ = LA.eigh(cov)
    idx = np.sort(evals)[-k:]
    return np.sum(idx)/np.sum(evals)

def plot_synapses(frac,life,born,die):
    """
    :param frac: existing connections fraction of matrix.
    :param life: lift-time record of synapses.
    :param born: newly-borned synapses each timestep
    :param die: just-died synapses each timestep
    :return: Four figures.
    """
    print("Plot the dynamics of synapses...")
    counter = Counter(life).most_common(100)
    x,y = list(zip(*counter))
    # import csv
    # with open('E:/pycode/RM-SORN/examples/scatter-200.csv', 'a+', newline='') as f:
    #     csv.writer(f).writerow(x)
    #     csv.writer(f).writerow(y)
    n = frac.shape[0]
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(frac,'g-')
    plt.ylabel('E-E connection fraction')
    plt.xlabel('steps')
    plt.xlim((0,n))
    plt.subplot(2,2,2)
    plt.scatter(np.log10(x),np.log10(y),alpha=0.6)
    plt.ylabel('Frequence-power')
    plt.xlabel('Life timesteps-power')
    plt.subplot(2,2,3)
    plt.scatter(np.arange(n),born,marker='^')
    plt.ylabel('New created synapses')
    plt.xlabel('steps')
    plt.xlim((0,n))
    plt.subplot(2,2,4)
    plt.scatter(np.arange(n),die,marker='^')
    plt.ylabel('New created synapses')
    plt.xlabel('steps')
    plt.xlim((0,n))
    plt.show()

def plot_spikes(X_train,Y_train):
    """
    Plot Inter-Spike-Interval statistics and dynamics of spikes.
    :param X_train:
    :param Y_train:
    :return:
    """
    print("Plot the dynamics of spikes...")
    (n,N_e) = X_train.shape
    ISI = [None]*N_e
    CV = np.zeros(N_e)
    raveled = []
    for i in range(N_e):
        ind = np.where(X_train[:,i]>0)[0]
        if len(ind)>1:
            ISI[i] = ind[1:]-ind[:-1]
            CV[i] = variation(ISI[i])
            raveled.extend(ISI[i])
        else:
            ISI[i] = np.zeros(1)
            CV[i] = 0
    counter = Counter(raveled).most_common(50) # 使用histgram
    x,y = list(zip(*counter))
    plt.figure()
    plt.subplot(2,2,1)
    plt.scatter(x,y,marker='^',alpha=0.6)
    # plt.hist(x,weights=y,bins=200,alpha=0.6)
    plt.xlabel('ISI(time steps)')
    plt.ylabel('Frequence')
    plt.subplot(2,2,2)
    plt.hist(CV,bins=50,density=0,alpha=0.6)
    plt.xlabel('ISI CV')
    plt.ylabel('Frequence')
    plt.subplot(2,2,3)
    plt.imshow(X_train[:2000,:].T, aspect='auto', interpolation='nearest')
    plt.ylabel('Neuron Index')
    plt.xlabel('Steps')
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(Y_train[:2000, :].T, aspect='auto', interpolation='nearest')
    plt.ylabel('Neuron Index')
    plt.xlabel('Steps')
    plt.show()



