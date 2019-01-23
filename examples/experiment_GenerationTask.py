from __future__ import division
import utils

from common.sources import CountingSource
from common.experiments import AbstractExperiment
from common.fbsorn import FBSorn
import os
import numpy as np
import gzip
import pickle

n_middle = 8
n_middles = [4,8,12,16,20,24,28,32]

class Experiment_test(AbstractExperiment):
    def start(self):
        super().start()
        c = self.params.c

        word = 'ABCDEFGHIJ'
        m_trans = np.ones((1,1))

        self.inputsource = CountingSource([word],m_trans,
                                          c.N_u_e,c.N_u_i,avoid=False)
        return self.inputsource

    def reset(self,sorn):
        super().reset(sorn)
        c = self.params.c
        sorn.__init__(c,self.inputsource)

    def run(self,sorn):
        super().run(sorn)
        c = self.params.c
        accs = []
        filepath = os.path.abspath(os.path.join(os.getcwd(),"..","rm_models"))

        # --------- Training Phase One ----------
        print('\nTraining Phase One: %d models are generated...'
                                %int(round(c.steps_train/c.interval_train)))
        nums = int(round(c.steps_train/c.interval_train))
        for i in range(nums):
            _ = sorn.simulation(c.interval_train)
            # filename = os.path.join(filepath,'stats.pickle')
            # with gzip.open(filename,'wb') as f:
            #     pickle.dump(ans,f,pickle.HIGHEST_PROTOCOL)
            filename = os.path.join(filepath,'%d-sorn.pickle'%i)
            sorn.quicksave(filename)
            sorn.update = False
            sorn.display = True
            acc = sorn.simulation(c.interval_test)
            accs.append(acc)
            sorn.update = True
            sorn.display = False
            # sorn = RMSorn.quickload(filename)
        print('\nThe %d th model achieve best performance: %0.2f%%'
                                %(np.argmax(accs),max(accs)*100))

        # --------- Training Phase Two ---------
        print('\nTraining Phase Two: close the plasticity in Recurrent layer...')
        # Load the RM-Sorn model with the best performance.
        with gzip.open(os.path.join(filepath,'%d-sorn.pickle'
                                             %(np.argmax(accs))),'rb') as f:
            sorn = pickle.load(f)
        sorn.W_ee.c.eta_stdp = 0
        sorn.W_ei.c.eta_istdp = 0
        sorn.W_ee.c.sp_prob = 0
        sorn.W_ee.c.no_prune = True
        c.eta_ip_e = 0
        c.noise_sig = 0
        sorn.simulation(c.steps_train)
        filename = os.path.join(filepath,'rm-sorn.pickle')
        sorn.quicksave()

        # --------- Testing Phase ---------
        print('\nTesting Phase: close all plasticity in SORN...')
        sorn.update = False
        sorn.display = True
        sorn.simulation(c.steps_test)

