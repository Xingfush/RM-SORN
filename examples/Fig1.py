import gzip
import numpy as np
import pickle as pickle
import os
import gzip
import utils

from examples.plot_result import *
data = utils.Bunch()

filepath = os.path.abspath(os.path.join(os.getcwd(),"..","art_models","countingTask"))

# # ---------------- Plot Raw Statistics----------------
# filename = os.path.join(filepath,'stats-8-400.pickle')
# with gzip.open(filename,'rb') as f:
#     temp = pickle.load(f)
#     data.X = temp['X']
#     data.Y = temp['Y']
#     # data.O = temp['O']
#     # data.R = temp['R']
#     data.frac = temp['frac']
#     data.life = temp['life']
#     data.mat = temp['mat']
#     data.born = temp['born']
#     data.die = temp['die']
#
# plot_synapses(data.frac,data.life,data.born,data.die)
# plot_spikes(data.X,data.Y)


# -------------- Plot the weight distribution --------------
filepath = os.path.abspath(os.path.join(os.getcwd(),"..","rm_models"))
with gzip.open(os.path.join(filepath, '00-sorn.pickle'),'rb') as f:
    sorn = pickle.load(f)
    print("The E-E fraction of connections is: %.2f." % get_ConnFraction(sorn.W_ee.M))
    plot_ConnDistribution(sorn.W_ee.W)

