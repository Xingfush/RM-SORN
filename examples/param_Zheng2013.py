from __future__ import division
import numpy as np
import utils
from common.defaults import *

c.N_e = 200                      # number of excitatory neurons
c.N_i = int(np.floor(0.2*c.N_e)) # number of inhibitory neurons
c.N = c.N_e + c.N_i
c.N_u_e = 0                      # no input

c.W_ee = utils.Bunch(lamb = 0.1*c.N_e,
                     avoid_self_connections=True,
                     eta_stdp = 0.004,
                     sp_prob =  c.N_e*(c.N_e-1)*(0.1/(200*199)),
                     sp_initial = 0.001,
                     no_prune = False,
                     upper_bound = 1
                     )

c.W_ei = utils.Bunch(lamb=0.2*c.N_e,
                     avoid_self_connections=True,
                     eta_istdp = 0.001,
                     h_ip=0.1)

c.W_ie = utils.Bunch(use_sparse=False,
                     lamb=1.0*c.N_i,
                     avoid_self_connections=True)

c.steps_plastic = 5000000
c.N_steps = c.steps_plastic
c.eta_ip = 0.01
c.h_ip = 0.1

# noise parameters
c.noise_sig =  np.sqrt(0.05)  # Gaussian noise
c.noise_fire = 0              # Random spike noise
c.noise_fire_struc = 0        # Random spike noise (subset), set to 1

c.display = True
# c.stats.only_last_spikes = 1000 # save all last spikes (careful!)
# c.stats.save_spikes = True

c.experiment.module = 'examples.experiment_Zheng2013'
c.experiment.name = 'Experiment_test'
