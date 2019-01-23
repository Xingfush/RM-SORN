from __future__ import division
import numpy as np

import pickle
import gzip
import sys
import utils
from common.synapses import create_matrix

# Intrinsic Plasticity
def ip(T, x, c):
    """
    Performs intrinsic plasticity
    Parameters:
        T: array
            The current thresholds
        x: array
            The state of the network
        c: Bunch
            The parameter bunch
    """
    T += c.eta_ip * (x - c.h_ip)
    return T

class Sorn():
    """
    Self-Organizing Recurrent Neural Network.
    """
    def __init__(self,c,source):
        """
        Initializes the variables of SORN
        :param c: Bunch(), the bunch of parameters
        :param source: the input source
        """
        self.c = c
        self.source = source

        # Initialize weight matrix
        # W_to_from (W_ie = from excitatory to inhibitory)
        self.W_ie = create_matrix((c.N_i,c.N_e),c.W_ie)
        self.W_ei = create_matrix((c.N_e,c.N_i),c.W_ei)
        self.W_ee = create_matrix((c.N_e,c.N_e),c.W_ee)
        self.W_eu = self.source.generate_connection_e(c.N_e)
        self.W_iu = self.source.generate_connection_i(c.N_i)

        # Initialize the neurons in SORN
        self.x = np.random.rand(c.N_e) < c.h_ip
        self.y = np.zeros(c.N_i)
        self.u = source.next()

        # Initialize the pre-threshold variables
        self.R_x = np.zeros(c.N_e)
        self.R_y = np.zeros(c.N_i)

        if c.ordered_thresholds:
            self.T_i = (np.arange(c.N_i)+0.5)*((c.T_i_max-c.T_i_min)/(1.*c.N_i))+c.T_i_min
            self.T_e = (np.arange(c.N_e)+0.5)*((c.T_e_max-c.T_e_min)/(1.*c.N_e))+c.T_e_min
            np.random.shuffle(self.T_e)
        else:
            self.T_i = c.T_i_min + np.random.rand(c.N_i)*(c.T_i_max-c.T_i_min)
            self.T_e = c.T_e_min + np.random.rand(c.N_e)*(c.T_e_max-c.T_e_min)

        # Activate plasticity mechanisms
        self.update = True

    def step(self,u_new):
        """
        Perform a one-step update of the SORN
        :param u_new: 1-D array, input for this step
        :return:
        """
        c = self.c
        # compute new state
        self.R_x = self.W_ee*self.x - self.W_ei*self.y - self.T_e
        if not c.noise_sig == 0:
            self.R_x += c.noise_sig*np.random.randn(c.N_e)
        if not c.ff_inhibition_broad == 0:
            self.R_x -= c.ff_inhibition_broad

        x_temp = self.R_x + c.input_gain*(self.W_eu*u_new)

        if c.k_winner_take_all:
            expected = int(round(c.N_e*c.h_ip))
            ind = np.argsort(x_temp)
            x_new = (x_temp > x_temp[ind[-expected-1]])+0
        else:
            x_new = (x_temp >= 0.0)+0

        # New noise -prob. of each neuron being active
        if not c.noise_fire == 0:
            x_new += ((np.random.random(c.N_e)<c.noise_fire)+0)
            x_new[x_new > 1] =1

        if self.c.fast_inhibit:
            x_used = x_new
        else:
            x_used = self.x

        self.R_y = self.W_ie*x_used - self.T_i
        if self.c.ff_inhibition:
            self.R_y += self.W_iu*u_new
        if not c.noise_sig == 0:
            self.R_y += c.noise_sig*np.random.randn(c.N_i)
        y_new = (self.R_y>=0.0)+0

        # Apply plasticity mechanisms
        # Always apply IP # different from Lezer et al. 2009
        ip(self.T_e ,x_new,self.c)
        # STDP, iSTDP, SS, Prune_weights, struct_plasticity
        if self.update:
            assert self.sane_before_update()
            self.W_ee.stdp(self.x, x_new)
            self.W_eu.stdp(self.u,u_new,to_old=self.x,to_new=x_new)

            self.W_ee.struct_p()
            self.W_ei.istdp(self.y, x_new)

            self.synaptic_scaling()

            assert self.sane_after_update()

        self.x = x_new
        self.y = y_new
        self.u = u_new

    def synaptic_scaling(self):
        """
        Performs synaptic scaling for all matrix
        :return:
        """
        self.W_ee.ss()
        self.W_ei.ss()
        if 'eta_stdp' in self.W_eu.c and self.W_eu.c.eta_stdp>0:
            self.W_eu.ss()

    def sane_before_update(self):
        """
        Basic sanity checks for thresholds and states before plasticity.
        :return:
        """
        eps = 1e-6
        assert all(np.isfinite(self.T_e))
        assert all(np.isfinite(self.T_i))

        assert all((self.x==0) | (self.x==1))
        assert all((self.y==0) | (self.y==1))
        if self.c.noise_sig == -1.0:
            assert all( self.R_x+self.T_e <= +1.0 + eps)
            assert all( self.R_x+self.T_e >= -1.0 - eps)
            assert all( self.R_y+self.T_i <= +1.0 + eps)
            assert all( self.R_y+self.T_i >= 0.0 - eps)
        return True

    def sane_after_update(self):
        """Basic sanity checks for matrix after plasticity"""
        assert self.W_ee.sane_after_update()
        assert self.W_ie.sane_after_update()
        return True

    def simulation(self,N,toReturn=[]):
        """
        Simulates SORN for a defined number of steps
        :param N: Simulation steps
        :param toReturn: Tracking variables to return
        :return:
        """
        c = self.c
        source = self.source
        ans = {}
        toReturn = ['X','Y','R_x','R_y','C','T']
        # Initialize tracking variables
        if 'X' in toReturn:
            ans['X'] = np.zeros((N,c.N_e))
        if 'Y' in toReturn:
            ans['Y'] = np.zeros((N,c.N_i))
        if 'R_x' in toReturn:
            ans['R_x'] = np.zeros((N,c.N_e))
        if 'R_y' in toReturn:
            ans['R_y'] = np.zeros((N, c.N_i))
        if 'T' in toReturn:
            ans['T'] = np.zeros((N, c.N_e))
        if 'C' in toReturn:
            ans['C'] = [None]*N

        # Simulation loop
        for n in range(N):
            # Simulation step
            self.step(source.next())
            # Tracking
            if 'X' in toReturn:
                ans['X'][n,:] = self.x
            if 'Y' in toReturn:
                ans['Y'][n,:] = self.y
            if 'R_x' in toReturn:
                ans['R_x'][n,:] = self.R_x
            if 'R_y' in toReturn:
                ans['R_y'][n,:] = self.R_y
            if 'T' in toReturn:
                # ans['U'][n,source.global_index()] = 1
                ans['T'][n,:] = self.T_e
            if 'C' in toReturn:
                ans['C'][n] = source.index()

            # Command line progress message
            if c.display and (N>100) and \
                    ((n%((N-1)//100)==0) or (n==N-1)):
                sys.stdout.write('\rSimulation: %3d%%'%((int)(n/(N-1)*100)))
                sys.stdout.flush()
        return ans

    def quicksave(self,filename=None):
        """
        Saves this sorn object
        :param filename: Filename to save in. Default: "net.pickle"
        """
        if filename == None:
            filename = utils.logfilename("net.pickle")
        print("The SORN network are saved to:",filename)
        pickle.dump(self, gzip.open(filename,"wb"),pickle.HIGHEST_PROTOCOL)

    @classmethod
    def quickload(cls,filename):
        """
        Loads a SORN.
        :param filename: File to load from.
        :return:
        """
        return pickle.load(gzip.open(filename,'rb'))
































