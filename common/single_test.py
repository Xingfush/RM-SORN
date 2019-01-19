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
# from common.lifsorn import LIFSorn

import pickle as pickle
import gzip
import numpy as np

def debugger(type,flag):
    print("In debugger! (single_test.py)")
    import ipdb
    ipdb.set_trace()

# Parameters are read from the second command line argument

# param: para_CountingTask
# 从命令行读入参数
# param = import_module(utils.param_file())
param = import_module('examples.param_rmCounting')

# experiment_module: experiment_CountingTask
# 导入params and experiment module，可以直接输入
experiment_module = import_module(param.c.experiment.module)
experiment_name = param.c.experiment.name
# 得到一个初始化后的experiment object.
experiment = getattr(experiment_module,experiment_name)(param)

# 从params 模块导入 params.c
c = param.c
c.logfilepath = os.path.abspath(os.path.join(os.getcwd(),".."))

# 从experiments 模块中导入 experiment，得到input source
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

