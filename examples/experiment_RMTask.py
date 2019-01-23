from __future__ import division
import utils

from common.sources import CountingSource,TrialSource,NoSource
from common.experiments import AbstractExperiment
from common.rmsorn import RMSorn
from examples.plot_result import get_ConnFraction

from matplotlib import pyplot as plt
import os
import numpy as np
import gzip
import pickle
import csv

n_middle = 4
n_middles = [4,8,12,16,20,24,28,32]

class Experiment_test(AbstractExperiment):
    def start(self):
        super().start()
        c = self.params.c

        word1 = 'A'
        word2 = 'D'
        for i in range(n_middle):
            word1 += 'B'
            word2 += 'E'
        word1 += 'C'
        word2 += 'F'
        m_trans = np.ones((2,2)) * 0.5
        self.inputsource = CountingSource([word1,word2],m_trans,
                                          c.N_u_e,c.N_u_i,avoid=False)
        return self.inputsource

    def reset(self,sorn):
        super().reset(sorn)
        c = self.params.c
        sorn.__init__(c,self.inputsource)

    def run(self,sorn):
        super().run(sorn)
        c = self.params.c
        accs1 = []
        accs2 = []
        filepath = os.path.abspath(os.path.join(os.getcwd(),"..","rm_models"))

        # --------- Training Phase One ----------
        print('\nTraining Phase One: %d models are generated...'
                                %int(round(c.steps_train/c.interval_train)))
        # Compute the expectation and std deviation
        filename1 = os.path.abspath(os.path.join("..", "art_models/overall-fixed.csv"))
        filename2 = os.path.abspath(os.path.join("..", "art_models/onlylast-fixed.csv"))
        filename3 = os.path.abspath(os.path.join("..", "art_models/fraction.csv"))
        for n_middle in n_middles:
            best1 = []
            best2 = []
            frac = []
            # Create the input Source
            word1 = 'A'
            word2 = 'D'
            for i in range(n_middle):
                word1 += 'B'
                word2 += 'E'
            word1 += 'C'
            word2 += 'F'
            m_trans = np.ones((2, 2)) * 0.5
            self.inputsource = CountingSource([word1, word2], m_trans,
                                              c.N_u_e, c.N_u_i, avoid=False)
            for i in range(10):
                nums = int(round(c.steps_train / c.interval_train))
                accs1 = []
                accs2 = []
                fraction = []
                self.reset(sorn)
                for i in range(nums):
                    _1,_2 = sorn.simulation(c.interval_train)
                    sorn.update = False
                    sorn.display = True
                    acc1,acc2 = sorn.simulation(c.interval_test)
                    accs1.append(acc1)
                    accs2.append(acc2)
                    fraction.append(get_ConnFraction(sorn.W_ee.M))
                    sorn.update = True
                    sorn.display = False
                best1.append(max(accs1))
                best2.append(max(accs2))
                frac.append(fraction[np.argmax(accs1)])
            with open(filename1,'a+',newline='') as f:
                csv.writer(f).writerow(best1)
            with open(filename2,'a+',newline='') as f:
                csv.writer(f).writerow(best2)
            with open(filename3,'a+',newline='') as f:
                csv.writer(f).writerow(frac)
            print("The mean expectation of RM-SORN is %.4f and %.4f.\n" % (np.mean(best1),np.mean(best2)))
            print("The std deviation of RM-SORN is %.4f and %.4f.\n" % (np.std(best1),np.std(best2)))