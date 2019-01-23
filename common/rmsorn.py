from __future__ import division
import numpy as np
import pickle
import gzip
from collections import deque
import sys

import utils
from common.synapses import create_matrix
from common.sources import CountingSource

def compute_accuracy(outputs,targets):
    """
    Compute accuracy for Counting Tasking, excluding the unpredictable first letter.
    :param outputs: 1-D array
    :param targets: 1-D array
    :return: float, accuracy
    """
    assert len(outputs)==len(targets)
    except_first = np.where((targets!=0)&(targets!=3))[0]
    only_last = np.where((targets==2)|(targets==5))[0]
    targets_red = targets[except_first]
    outputs_red = outputs[except_first]
    # compute the last prediction accuracy
    targets_only = targets[only_last]
    outputs_only = outputs[only_last]
    perf_red = (targets_red==outputs_red).sum() /float(len(targets_red))
    perf_only = (targets_only==outputs_only).sum() /float(len(targets_only)+1)
    return perf_red,perf_only

def predict_accuracy(outputs,targets):
    """
    Compute accuracy for Motion Predition task..
    :param outputs: 1-D array
    :param targets: 1-D array
    :return: float, accuracy
    """
    assert len(outputs)==len(targets)
    except_first = np.where((targets!=0) & (targets!=7))[0]
    targets_red = targets[except_first]
    outputs_red = outputs[except_first]
    perf_red = (targets_red==outputs_red).sum() /float(len(targets_red))
    return perf_red

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

class RMSorn():
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
        self.u = source.next()

        # Initialize the Modulation-Factors
        self.m_r = 1 # reward modulation factor of recurrent layer
        self.m_o = 1 # reward modualtion factor of readout layer
        self.r = 0 # reward
        self.reward = deque(maxlen=c.window_size) # helper list to record

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
        self.statistics = False

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
        # Not suitable for generation tasks
        target = self.source.next()
        if np.all(target == o_new):
            self.r = 1
        else:
            if c.punishment:
                self.r = -1
            else:
                self.r = 0
        if c.window_size == 0:
            self.m_o = self.r
        else:
            self.m_o = self.r - sum(self.reward)/c.window_size
            self.reward.append(self.r)
        if c.recurrent_reward:
            self.m_r = self.m_o

        # Apply plasticity mechanisms
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
        source = self.source
        ans = {}
        toReturn =['X','Y','O','C']
        # Statistics Variables # Will
        if self.statistics:
            SynapsesLife = []
            eeFraction = np.zeros(N)
            SynapsesMat = np.zeros((N//100,c.N_e,c.N_e))
            TimeBorn = np.zeros((c.N_e,c.N_e))-1
            BornedRecord = np.zeros(N)
            DiedRecord = np.zeros(N)

        # Initialize tracking variables
        if 'X' in toReturn:
            ans['X'] = np.zeros((N,c.N_e))
        if 'Y' in toReturn:
            ans['Y'] = np.zeros((N,c.N_i))
        if 'O' in toReturn:
            ans['O'] = np.zeros(N)
        if 'C' in toReturn:
            ans['C'] = np.zeros(N)
        if 'R' in toReturn:
            ans['R'] = np.zeros(N)

        for n in range(N):
            # old_mask = self.W_ee.M.copy() # Will
            self.step(source.current())
            # new_mask = self.W_ee.M.copy() # Will
            # Tracking
            if 'X' in toReturn:
                ans['X'][n,:] = self.x
            if 'Y' in toReturn:
                ans['Y'][n,:] = self.y
            if 'O' in toReturn:
                ans['O'][n] = np.argmax(self.o)
            if 'C' in toReturn:
                ans['C'][n] = source.index()
            if 'R' in toReturn:
                ans['R'][n] = self.r
            # Statistics Record
            if self.statistics:
                Borned = np.where(1*new_mask-1*old_mask>0)
                Died = np.where(1*new_mask-1*old_mask<0)
                BornedRecord[n] = len(Borned[0])
                DiedRecord[n] = len(Died[0])
                if len(Borned[0])>0:
                    TimeBorn[Borned] = n
                if len(Died[0])>0:
                    SynapsesLife.extend(n-TimeBorn[Died])
                eeFraction[n] = np.sum(new_mask)/(c.N_e*c.N_e)
                if (n+1)%100 == 0:
                    SynapsesMat[n//100,:,:] = self.W_ee.W

            # Command line progress message
            if c.display and (N>10) and \
                    ((n%((N-1)//10)==0) or (n==N-1)):
                sys.stdout.write('\rSimulation: %3d%%'%((int)(n/(N-1)*100)))
                sys.stdout.flush()
        accuracy1,accuracy2 = compute_accuracy(ans['O'], ans['C']) # TODO
        # accuracy = predict_accuracy(ans['O'], ans['C']) # Motion task has only one accuracy
        # if self.display:
            # print('Online RM-SRON performance assess: %0.2f%%\n' % (accuracy*100))
        if self.display:
            print('Online RM-SORN performance assess: %0.2f%% and %0.2f%%' % (accuracy1*100,accuracy2*100))
        if self.statistics:
            SynapsesLife.extend(N-TimeBorn[self.W_ee.M])
            ans['frac'] = eeFraction
            ans['life'] = np.array(SynapsesLife)+1
            ans['mat'] = SynapsesMat
            ans['born'] = BornedRecord
            ans['die'] = DiedRecord

        return accuracy1,accuracy2

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

