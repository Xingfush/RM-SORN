from __future__ import division
import utils
import numpy as np
import os
import csv

from common.sources import CountingSource,TrialSource
from common.experiments import AbstractExperiment

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
        m_trans = np.ones((2,2))*0.5

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
        # Compute the expectation and std deviation
        filename1 = os.path.abspath(os.path.join("..","art_models/overall-sorn.csv"))
        filename2 = os.path.abspath(os.path.join("..", "art_models/onlylast-sorn.csv"))
        for n_middle in n_middles:
            acc1 = []
            acc2 = []
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

            for i in range(100):
                self.reset(sorn)
                #----- Input with plasticity
                print("\nInput plasticity period:")
                ans1 = sorn.simulation(c.steps_plastic)
                #----- Input without plasticity - train
                print("\nInput training period:")
                # Turn off plasticity
                sorn.W_ee.c.eta_stdp = 0
                sorn.W_ei.c.eta_istdp = 0
                sorn.W_ee.c.sp_prob = 0
                c.eta_ip = 0
                # turn off noise
                c.noise_sig = 0
                ans2 = sorn.simulation(c.steps_readouttrain)
                #----- Input without plasticity - test performance
                print('\nInput test period:')
                ans3 = sorn.simulation(c.steps_readouttest)
                # Compute the accuracy of SORN.
                y_read_train = np.zeros((6, len(ans2['C']) + 1))
                y_read_test = np.zeros((6, len(ans3['C']) + 1))
                for i, y in enumerate(ans2['C']):
                    y_read_train[y, i] = 1
                for i, y in enumerate(ans3['C']):
                    y_read_test[y, i] = 1
                y_read_train = y_read_train[:, :5000]
                y_read_test = y_read_test[:, :5000]
                target = np.argmax(y_read_test, axis=0)
                # States variables
                X_train = (ans2['R_x'] >= 0) + 0.
                X_test = (ans3['R_x'] >= 0) + 0.
                # Typical Calculation of Reservior Computing Readout
                X_train_pinv = np.linalg.pinv(X_train)
                W_trained = np.dot(y_read_train, X_train_pinv.T)

                y_predicted = np.dot(W_trained, X_test.T)
                prediction = np.argmax(y_predicted, axis=0)
                except_first = np.where((target != 0) & (target != 3))[0]
                only_last = np.where((target==2) | (target==5))[0]
                # Reduced performance(i.e. reduce the random initial char of word)
                y_test_red = target[except_first]
                y_pred_red = prediction[except_first]
                y_test_only = target[only_last]
                y_pred_only = prediction[only_last]
                perf_red = (y_test_red == y_pred_red).sum() / float(len(y_pred_red))
                perf_only = (y_test_only == y_pred_only).sum() / float(len(y_pred_only)+1)
                print("The testing accuracy of Counting Task is: %0.2f%% and %0.2f%%\n"
                                        % (perf_red*100, perf_only*100))
                acc1.append(perf_red)
                acc2.append(perf_only)
            with open(filename1,'a+',newline='') as f:
                csv.writer(f).writerow(acc1)
            with open(filename2,'a+',newline='') as f:
                csv.writer(f).writerow(acc2)
            print("The mean expectation of SORN is %.4f and %.4f.\n" % (np.mean(acc1),np.mean(acc2)))
            print("The std deviation of SORN is %.4f and %.4f.\n" % (np.std(acc1),np.std(acc2)))
            # print("The best performance of SORN is:" , acc1)
            # print("The best performance of SORN is:", acc2)

        # #----- Input with plasticity
        # print("\nInput plasticity period:")
        # ans1 = sorn.simulation(c.steps_plastic)
        # #----- Input without plasticity - train
        # print("\nInput training period:")
        # # Turn off plasticity
        # sorn.W_ee.c.eta_stdp = 0
        # sorn.W_ei.c.eta_istdp = 0
        # sorn.W_ee.c.sp_prob = 0
        # c.eta_ip = 0
        # # turn off noise
        # c.noise_sig = 0
        #
        # ans2 = sorn.simulation(c.steps_readouttrain)
        # #----- Input without plasticity - test performance
        # print('\nInput test period:')
        # ans3 = sorn.simulation(c.steps_readouttest)
        #
        # return {'plastic':ans1,'training':ans2,'testing':ans3}
