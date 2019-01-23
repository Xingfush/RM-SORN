from __future__ import division
import numpy as np
import pickle
import gzip
from collections import deque
import os
import sys

import utils
from common.synapses import create_matrix
from common.sources import CountingSource

def compute_accuracy(arr,N):
    """
    Compute the accuracy of Generation Task.
    :param output: 1-D array
    :param N: int, length of target sequence
    :return: float
    """
    ind = np.nonzero(arr==0)[0]
    if len(ind)==0:
        return 0
    repeat = len(arr)//N
    target = list(range(N))*repeat
    perf = compute_lcs(arr,target)
    return perf

def compute_lcs(x,y):
    m = len(x)
    n = len(y)
    c = np.zeros((m + 1, n + 1))
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                c[i, j] = c[i - 1, j - 1] + 1
            elif c[i - 1, j] >= c[i, j - 1]:
                c[i, j] = c[i - 1, j]
            else:
                c[i, j] = c[i, j - 1]
    return c[m,n]/m

def compute_reward(arr,N):
    ind = np.nonzero(arr==0)[0]
    if len(ind)==0:
        return 0
    else:
        ind = ind[-1]
    temp = np.arange(N-ind)
    if ind==N-1 and arr[ind]==arr[ind-1]:
        return -1
    if np.all(arr[ind:]==temp):
        return (N-ind)/N
    else:
        return 0

# Intrinsic Plasticity
def ip(T, x, eta_ip, h_ip):
    """
    Perform intrinsic plasticity
    :param T: array, The current threshold
    :param x: array, The state of the network
    :param eta_ip: float
    :param h_ip: array or list
    """
    if isinstance(h_ip,float):
        h_ip = np.ones(x.shape)*h_ip
    assert len(h_ip)==len(x)
    T += eta_ip * (x - h_ip)
    return T

class FBSorn():
    """
    Reward-Modulated Self-Organizing Recurrent Neural Network.
    """
    def __init__(self,c,source):
        """
        Initialize the variables of SORN
        :param c: Bunch(), the bunch of parameters
        :param source: the input source
        """
        self.c = c
        self.source = source

        # Initialize weight matrix
        # W_oe: connections from excitatory to output(excitatory) neurons.
        self.W_ie = create_matrix((c.N_i,c.N_e),c.W_ie)
        self.W_ei = create_matrix((c.N_e,c.N_i),c.W_ei)
        self.W_ee = create_matrix((c.N_e,c.N_e),c.W_ee)
        self.W_oe = create_matrix((c.N_o,c.N_e),c.W_oe)
        self.W_eu = self.source.generate_connection_e(c.N_e)
        self.W_iu = self.source.generate_connection_i(c.N_i)

        self.x = np.random.rand(c.N_e) < c.h_ip_e
        self.y = np.zeros(c.N_i)
        self.o = np.zeros(c.N_o)

        # Initialize the Modulation-Factors
        self.m_r = 1 # reward modulation factor of recurrent layer
        self.m_o = 1 # reward modualtion factor of readout layer
        self.r = 0 # reward
        self.reward = deque(maxlen=c.window_size) # helper list to record
        self.output = deque(maxlen=c.N_o) # helper list to record
        # Initialize output
        self.output.append(0)

        # Initialize the pre-threshold variables
        self.R_x = np.zeros(c.N_e)
        self.R_y = np.zeros(c.N_i)
        self.R_o = np.zeros(c.N_o)

        if c.ordered_thresholds:
            self.T_i = (np.arange(c.N_i)+0.5)*((c.T_i_max-c.T_i_min)/(1.*c.N_i))+c.T_i_min
            self.T_e = (np.arange(c.N_e)+0.5)*((c.T_e_max-c.T_e_min)/(1.*c.N_e))+c.T_e_min
            self.T_o = (np.arange(c.N_o)+0.5)*((c.T_o_max-c.T_o_min)/(1.*c.N_o))+c.T_o_min
            np.random.shuffle(self.T_e)
            np.random.shuffle(self.T_o)
        else:
            self.T_i = c.T_i_min + np.random.rand(c.N_i)*(c.T_i_max-c.T_i_min)
            self.T_e = c.T_e_min + np.random.rand(c.N_e)*(c.T_e_max-c.T_e_min)
            self.T_o = c.T_o_min + np.random.rand(c.N_o)*(c.T_o_max-c.T_o_min)

        # Seperately activate the plasticity mechanism in Recurrent and Output.
        self.update = True
        self.display = False

    def step(self,u_new):
        """
        Perform a one-step update of the SORN
        :param u_new: array, 1-D
        """
        c = self.c
        # Compute new state
        self.R_x = self.W_ee*self.x - self.W_ei*self.y - self.T_e
        if not c.noise_sig == 0:
            self.R_x += c.noise_sig*np.random.randn(c.N_e)
        if not c.ff_inhibition_broad == 0:
            self.R_x -= c.ff_inhibition_broad
        x_temp = self.R_x + c.input_gain*(self.W_eu*u_new)

        if c.k_winner_take_all:
            expected = int(round(c.N_e*c.h_ip_e))
            ind = np.argsort(x_temp)
            x_new = (x_temp>x_temp[ind[-expected-1]])+0
        else:
            x_new = (x_temp>=0.0)+0

        # Output layer fire only one neuron each step.(Winner-Take-All)
        # No sigma noise and other structure noise
        self.R_o = self.W_oe*self.x -self.T_o
        o_new = (self.R_o>=np.amax(self.R_o))+0

        if self.c.fast_inhibit:
            x_used = x_new
        else:
            x_used = self.x
        self.R_y = self.W_ie*x_used - self.T_i
        if self.c.ff_inhibition:
            self.R_y += self.W_iu*u_new
        if not c.noise_sig ==0:
            self.R_y += c.noise_sig*np.random.randn(c.N_i)
        y_new = (self.R_y>=0.0)+0

        # Calculate the Reward-Modulation factor
        # Suitable for generation tasks
        self.r = compute_reward(self.output,c.N_o)
        if c.window_size == 0:
            self.m_o = self.r
        else:
            self.m_o = self.r - sum(self.reward)/c.window_size
            self.reward.append(self.r)
        if c.recurrent_reward:
            self.m_r = self.m_o

        if self.update:
            ip(self.T_e, x_new, c.eta_ip_e, c.h_ip_e)
            ip(self.T_o, o_new, c.eta_ip_o, c.h_ip_o)
            assert self.sane_before_update()
            self.W_ee.mstdp(self.x, x_new, self.m_r)
            self.W_oe.mstdp(self.x, x_new, self.m_o,
                            to_old=self.o,to_new=o_new)

            self.W_ee.struct_p()
            self.W_ei.istdp(self.y,x_new)
            self.synaptic_scaling()
            assert self.sane_after_update()

        self.output.append(np.argmax(o_new))
        self.x = x_new
        self.y = y_new
        self.o = o_new

    def synaptic_scaling(self):
        """
        Perform syanptic scaling for all matrix.
        """
        self.W_ee.ss()
        self.W_oe.ss()
        self.W_ei.ss()
        if 'eta_stdp' in self.W_eu.c and self.W_eu.c.eta_stdp>0:
            self.W_eu.ss()

    def sane_before_update(self):
        """
        Basic sanity checks for thresholds and states before plasticity.
        """
        assert all(np.isfinite(self.T_e))
        assert all(np.isfinite(self.T_o))
        assert all(np.isfinite(self.T_i))

        assert all((self.x==0) | (self.x==1))
        assert all((self.y==0) | (self.y==1))
        assert all((self.o==0) | (self.o==1))
        return True

    def sane_after_update(self):
        """
        Basic sanity checks for matrix after plasticity.
        """
        assert self.W_ee.sane_after_update()
        assert self.W_oe.sane_after_update()
        # assert self.W_ie.sane_after_update()
        assert self.W_ei.sane_after_update()
        return True

    def simulation(self,N,toReturn=[]):
        """
        Simulates RM-SORN for a defined number of steps.
        :param N: Simulation steps
        :param toReturn: Tracking variables to return
        :return: state records
        """
        c = self.c
        ans = {}
        toReturn =['X','Y','O','C']

        # Initialize tracking variables
        if 'X' in toReturn:
            ans['X'] = np.zeros((N,c.N_e))
        if 'Y' in toReturn:
            ans['Y'] = np.zeros((N,c.N_i))
        if 'O' in toReturn:
            ans['O'] = np.zeros(N)
        if 'R' in toReturn:
            ans['R'] = np.zeros(N)

        for n in range(N):
            self.step(self.o)
            # Tracking
            if 'X' in toReturn:
                ans['X'][n,:] = self.x
            if 'Y' in toReturn:
                ans['Y'][n,:] = self.y
            if 'O' in toReturn:
                ans['O'][n] = np.argmax(self.o)
            if 'R' in toReturn:
                ans['R'][n] = self.r

            # Command line progress message
            if c.display and (N>10) and \
                    ((n%((N-1)//10)==0) or (n==N-1)):
                sys.stdout.write('\rSimulation: %3d%%'%((int)(n/(N-1)*100)))
                sys.stdout.flush()

        accuracy = compute_accuracy(ans['O'],c.N_o)
        if self.display:
            print('Online RM-SORN performance assess: %0.2f%%' % (accuracy*100))
        if N == c.steps_test: # TODO
            filepath = os.path.join('..','rm_models')
            with open(os.path.join(filepath,'generation.csv'),'w') as f:
                np.savetxt(f,ans['O'].astype(int),delimiter=',')

        return accuracy

    def quicksave(self,filename=None):
        """
        Saves this sorn object
        :param filename: str, Default: "net.pickle"
        """
        if filename == None:
            filename = utils.logfilename("sorn.pickle")
        print("\nThe RM-SORN network are saved to:",filename)
        pickle.dump(self, gzip.open(filename,"wb"),pickle.HIGHEST_PROTOCOL)

    @classmethod
    def quickload(cls,filename):
        """
        Loads a SORN.
        :param filename: File to load from.
        """
        return pickle.load(gzip.open(filename,'rb'))

