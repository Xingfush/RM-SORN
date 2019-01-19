import numpy as np
import utils

"""
Default Parameters that are used when no other parameters are specified
"""

c = utils.Bunch()

# Number of units (e: excitatory, i: inhibitory, u: input)
c.N_e = 200
c.N_i = int(0.2*c.N_e)
c.N_u_e = int(0.05*c.N_e)
c.N = c.N_e + c.N_i

# Each submatrix expects a Bunch with some of the following fields:
# c.use_sparse = True
# Number of connections per neuron
# c.lamb = 10 (or inf to get full connectivity)
# c.avoid_self_connections = True
# c.eta_stdp = 0.001 (or 0.0 to disable)
# c.eta_istdp = 0.001 (or 0.0 to disable)
# c.sp_prob = 0.1 (or 0 to disable)
# c.sp_initial = 0.001

c.W_ee = utils.Bunch(lamb=10,
                     avoid_self_connections=True,
                     eta_ip=0.001,
                     eta_stdp = 0.001,
                     sp_prob = 0.1,
                     sp_initial=0.001)

c.W_ei = utils.Bunch(lamb=np.inf,
                     avoid_self_connections=False,
                     eta_ip = 0.0,
                     eta_istdp = 0.0,
                     h_ip=0.1)

c.W_ie = utils.Bunch(lamb=np.inf,
                     avoid_self_connections=False)

# Std of noise added at each step (mean = 0)
c.noise_sig = 0.0
c.display = False
c.N_iterations = 10
# IP Parameters
c.eta_ip = 0.001
c.h_ip = 2.0*c.N_u_e/c.N_e
# Thresholds
c.T_e_max = 1.0
c.T_e_min = 0.0
c.T_i_max = 0.5
c.T_i_min = 0.0
c.steps_plastic = 7000
# When testing performance how many steps to train classifier
c.steps_noplastic_train = 5000
# When testing performance how many steps to test classifier
c.steps_noplastic_test = 5000
c.N_steps = c.steps_plastic + c.steps_noplastic_train \
                            + c.steps_noplastic_test
c.fast_inhibit = True
c.ordered_thresholds = False
c.ff_inhibition = False
c.ff_inhibition_broad = 0
c.with_plasticity = True
c.k_winner_take_all = False
c.input_gain = 1

c.source = utils.Bunch()
c.experiment = utils.Bunch()