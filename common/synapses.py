from __future__ import division
import numpy as np
import utils

def _find_new(W,aviod_self_connections):
    (i_max,j_max) = W.shape
    counter = 0
    while True:
        i = np.random.randint(i_max)
        j = np.random.randint(j_max)
        connected = W[i,j] > 0
        valid = (not aviod_self_connections) or (i!=j)
        if valid and not connected:
            break
        if valid and (counter >= 100):
            print("Leaving find_new early")
            break
        counter += 1
    return (i,j)

def create_matrix(shape,c):
    ''' Expecting a c with the following fields:
        c.lamb = 10 (or inf to get full connectivity)
        c.avoid_self_connections = True
        c.eta_stdp = 0.001 (or 0.0 to disable)
        c.eta_istdp = 0.001 (or 0.0 to disable)

        c.sp_prob = 0.1 (or 0 to disable)
        c.sp_initial = 0.001
    '''
    return FullSynapticMatrix(shape,c)

class AbstractSynapticMatrix():
    """
    Interface for synaptic connection matrices
    """
    def __init__(self,shpae,c):
        """
        Initializes the variables of the matrix.

        Parameters:
            shape: array
                row/col --> size(to), size(from)
            c: bunch
                parameters as stated previously
        """
        raise NotImplementedError
    def prune_weights(self):
        """
        Prunes weights that are 0
        """
        raise NotImplementedError
    def struct_p(self):
        """
        Performs structural plasticity on the matrix
        defined as random instertion of a new connection
        """
        raise NotImplementedError
    def stdp(self,from_old,from_new,to_old=None,to_new=None):
        """
        Performs Spike-Timing-Dependent-Plasticity

        Parameters:
            from_old: array
                The state vector of the projecting population from the
                last time step
            from_new: array
                The state vector of the projecting population from
                this time step
            to_old: array
                The old state vector of the receiving population
                Default: from_old
            to_new: array
                The new state vector of the receiving population
                Default: to_new
        """
        raise NotImplementedError
    def istdp(self,y_old,x):
        """
        Performs Inhibitory Spike-Timing-Dependent-Plasticity as defined
        in Zheng 2013

        Parameters:
            y_old: array
                The old state of the inhibitory population
            x: array
                The new state of the excitatory population
        """
        raise NotImplementedError
    def istdp_pos(self,y_old,x):
        """
        Performs only the positive part of istdp (see this for details).
        """
        raise NotImplementedError
    def ss(self):
        """
        Performs synaptic scaling defined as normalizing all incoming
        weights to a sum of 1 for each unit
        """
        raise NotImplementedError
    def __mul__(self,x):
        """
        Multiplication of this matrix with a vector

        Parameters:
            x: array
                The vector to multiply with
        Returns:
            The product self*x
        """
        raise NotImplementedError
    def get_synapses(self):
         """
         Returns a dense copy of the connection matrix
         """
         raise NotImplementedError
    def set_synapses(self,W_new):
        """
        Sets the synapses.

        Parameters:
            W_new: matrix
                The new dense connection matrix
        """
        raise NotImplementedError
    def sane_after_update(self):
        """
        Checks if the incomming connections to each neuron sum to 1 and
        if the weights are in the range (0,1)
        """
        raise NotImplementedError

class FullSynapticMatrix(AbstractSynapticMatrix):
    """
    Dense connection matrix class for SORN synapses.
    """
    def __init__(self,shape,c):

        self.c = c
        if 'eat_ss' not in c:
            self.c.eta_ss = 1
        # M is a mask with 1s for all existing connections
        if c.lamb >= shape[0]:
            self.M = np.ones(shape)==True # cast to boolean
        else:
            self.M = np.random.rand(*shape) < (c.lamb/shape[0])
            # Iteratively creates connection matrices until all neurons get
            # some input
            # TODO while(True) just has to blow up at some point
            assert(c.lamb > 0)
            while(True):
                num = np.sum(np.sum(self.M,1)==0)
                self.M[np.sum(self.M,1)==0] = np.random.rand(num,shape[1])<(c.lamb/shape[0])
                if c.avoid_self_connections:
                    np.fill_diagonal(self.M,False)
                if np.all(np.sum(self.M,1)>0):
                    break
        self.W = np.random.rand(*shape)
        self.W[~self.M] = 0.0
        self.ss()
        # Create eligibility trace of weight matrix
        self.e = np.zeros(shape)

    def prune_weights(self):
        c = self.c
        if 'no_lower_bound' in c and c.no_lower_bound:
            return
        self.W[self.W<0.0] = 0.0
        if 'upper_bound' in c:
            self.W[self.W > c.upper_bound] = c.upper_bound
        if 'no_prune' in c and c.no_prune:
            return
        self.M[self.W<=0.0] = False

    def struct_p(self):
        c = self.c
        if 'sp_prob' in c and np.random.rand() < c.sp_prob:
            (i,j) = _find_new(self.W,c.avoid_self_connections)

            self.W[i,j] = c.sp_initial
            self.M[i,j] = True

    def stdp(self,from_old,from_new,to_old=None,to_new=None):
        c = self.c
        if 'eta_stdp' not in c:
            return

        if to_old is None:
            to_old = from_old
        if to_new is None:
            to_new = from_new

        dw = c.eta_stdp*(to_new[:,None]*from_old[None,:]
                         -to_old[:,None]*from_new[None,:])

        # Uses M to only change connections that actually exist
        self.W[self.M] += dw[self.M]
        self.W[self.W>1.0] = 1.0
        self.prune_weights()

    def mstdp(self,from_old,from_new,factor,to_old=None,to_new=None):
        """
        Implement the Reward-Modulated STDP in RM-SORN.
        """
        c = self.c
        if not 'eta_stdp' in c:
            return
        if to_old is None:
            to_old = from_old
        if to_new is None:
            to_new = from_new
        dw = factor*c.eta_stdp*(to_new[:,None]*from_old[None,:]
                             -c.f*to_old[:,None]*from_new[None,:])

        self.W[self.M] += dw[self.M]
        self.W[self.W>1.0] = 1.0
        self.prune_weights()

    def rstdp(self,from_old,from_new,factor,to_old=None,to_new=None):
        """
        Implement the Reward-Modulated STDP from Izh et.al.
        """
        c = self.c
        if 'eta_stdp' not in c:
            return
        if to_old is None:
            to_old = from_old
        if to_new is None:
            to_new = from_new
        dw = (to_new[:,None]*from_old[None,:]
                         -c.f*to_old[:,None]*from_new[None,:])
        self.e = self.e*np.exp(-c.dt/c.t_e) + dw*c.td
        self.W[self.M] += factor*c.eta_stdp*self.e[self.M]
        self.prune_weights()

    def istdp(self,y_old,x):
        c = self.c
        if 'eta_istdp' not in c:
            return
        self.W[self.M] += -c.eta_istdp*((1-(x[:,None]*(1+1.0/c.h_ip)))\
                                        *y_old[None,:])[self.M]
        # can't use W < 0 because W get's 0 often
        self.W[self.W<=0] = 0.001
        self.W *= self.M
        self.W[self.W>1.0] = 1.0

    def istdp_pos(self,y_old,x):
        c = self.c
        if 'eta_istdp' not in c:
            return
        self.W[self.M] += c.eta_istdp*((1-x[:,None])\
                                       *y_old[None,:])[self.M]
         # can't use W < 0 because W get's 0 often
        self.W[self.W<=0] = 0.001
        self.W *= self.M
        self.W[self.W>1.0] = 1.0

    def ss(self):
        z = abs(self.W).sum(1)
        z[z < 1e-6] = 1e-6
        if self.c.eta_ss<1:
            self.W *= 1.-self.c.eta_ss*(1.-1./z[:,None])
        else:
            self.W /= z[:,None]

    def __mul__(self,x):
        return self.W.dot(x)

    def get_synapses(self):
        return self.W.copy()

    def set_synapses(self,W_new,scale=True):
        self.W = W_new.copy()
        self.M = W_new>0
        if scale:
            self.ss()
            self.prune_weights()

    def sane_after_update(self):
        eps = 1e-6
        Z = self.W.sum(1)
        if any(abs(Z-1.0)>eps):
            print(shape(self.W))
            print(self.W.sum(0))
            print(self.W.sum(1))
            ind = abs(Z-1.0)>eps
            print(find(ind))
            print("Difference from 1:",Z[ind]-1.0)
            self.ss()
            Z = self.W.sum(1)
            print("Difference after trying to fix it:",Z[ind]-1.0)

        assert np.all(self.W >= 0.0)
        assert np.all(self.W <= 1.0)

        return True



