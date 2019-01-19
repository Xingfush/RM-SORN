from __future__ import division
import numpy as np
import utils

from common.defaults import *

c.N_e = 200
c.N_i = int(np.floor(0.2*c.N_e))
c.N = c.N_e + c.N_i
c.N_u_e = int(np.floor(0.05*c.N_e))
c.N_u_i = 0

c.W_ee = utils.Bunch(lamb =0.1*c.N_e,
                     avoid_self_connections=True,
                     eta_stdp= 0.004,
                     sp_prob =0,
                     sp_initial =0,
                     no_prune =False,
                     upper_bound =1)
c.W_ei = utils.Bunch(lamb=1*c.N_e,
                     avoid_self_connections=True,
                     eta_istdp=0,
                     h_ip=0.1)
c.W_ie = utils.Bunch(lamb=1.0*c.N_i,
                     avoid_self_connections=True)

c.steps_plastic = 5000
c.steps_readouttrain = 5000
c.steps_readouttest = 5000

c.eta_ip = 0.01
c.h_ip = 0.1
c.noise_sig = np.sqrt(0.05)
c.noise_fire = 0
c.display = True

c.input_gain = 1 # needs to be 1!

c.experiment.module = 'examples.experiment_CountingTask'
c.experiment.name = 'Experiment_test'