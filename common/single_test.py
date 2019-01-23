from __future__ import division
import os
import sys
import ipdb
from importlib import import_module

import utils

from utils.backup import dest_directory
from common.sorn import Sorn
from common.rmsorn import RMSorn
from common.fbsorn import FBSorn

import pickle as pickle
import gzip
import numpy as np

def debugger(type,flag):
    print("In debugger! (single_test.py)")
    import ipdb
    ipdb.set_trace()

param = import_module('examples.param_rmCounting')

# experiment_module: experiment_CountingTask
experiment_module = import_module(param.c.experiment.module)
experiment_name = param.c.experiment.name
experiment = getattr(experiment_module,experiment_name)(param)

c = param.c
c.logfilepath = os.path.abspath(os.path.join(os.getcwd(),".."))

source = experiment.start()
# sorn = Sorn(c,source)
sorn = RMSorn(c,source)
# sorn = FBSorn(c,source)

experiment.reset(sorn)

# Run experiment once, return the Input_Source
# pickle_objects = experiment.run(sorn)
experiment.run(sorn)

# # Save sources etc
# for key in pickle_objects:
#     filename = os.path.join(c.logfilepath,"%s.pickle"%key)
#     print("\nThe state variables are saved into:" ,filename)
#     topickle = pickle_objects[key]
#     pickle.dump(topickle,gzip.open(filename,'wb'),
#                 pickle.HIGHEST_PROTOCOL)

