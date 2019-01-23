from __future__ import division
import numpy as np
import utils

from common.defaults import *

c.N_e = 200
c.N_o = 32
c.N_i = int(np.floor(0.2*c.N_e))
c.N = c.N_e + c.N_i
c.N_u_e = int(np.floor(0.15*c.N_e))
c.N_u_i = 0

c.W_ee = utils.Bunch(lamb = 0.05*c.N_e,
                     avoid_self_connections=True,
                     eta_stdp=0.001,
                     f = 1,
                     no_prune=False,
                     sp_prob=c.N_e*(c.N_e-1)*(0.1/(200*199)),
                     sp_initial=0.001,
                     upper_bound=1)
c.W_ei = utils.Bunch(lamb=1*c.N_e,
                     avoid_self_connections=True,
                     eta_istdp=0.0,
                     h_ip=0.1)
c.W_ie = utils.Bunch(lamb=1.0*c.N_i,
                     avoid_self_connections=True)
c.W_oe = utils.Bunch(lamb=1.0*c.N_o,
                     avoid_self_connections=True,
                     eta_stdp=0.005,
                     f = 0,
                     no_prune=True,
                     upper_bound=1)

c.T_o_max = 1.0
c.T_o_min = 0.0
c.T_e_max = 1.0
c.T_e_min = 0.0
c.T_i_max = 1.0
c.T_i_min = 0.0
c.eta_ip_e = 0.002
c.eta_ip_o = 0.005
c.h_ip_e = 0.2
c.h_ip_o = 0.03125


c.punishment = True
c.recurrent_reward = False
c.window_size = 20

c.steps_train = 20000
c.steps_test = 10000
c.interval_train = 100
c.interval_test = 1000
c.steps_plastic = 50000

c.fast_inhibition = False
c.noise_sig = 0
c.noise_fire = 0
c.display = True

c.input_gain = 1
c.experiment.module = 'examples.experiment_MotionTask'
c.experiment.name = 'Experiment_test'

