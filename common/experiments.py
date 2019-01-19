from __future__ import division
import utils
import numpy as np

class AbstractExperiment():
    """An experiment encapsulates everything experiment-specific for a simulation."""
    def __init__(self,params):
        self.params = params

    def start(self):
        """This is called initially to prepare the experiment.
           It should return a Sorn and a list of initialized stats."""
        self.ff_inhibition_broad = self.params.c.ff_inhibition_broad
        self.eta_ip = self.params.c.eta_ip

    def reset(self,sorn):
        sorn.update = self.params.c.with_plasticity
        if not self.params.c.with_plasticity:
            sorn.c.eta_ip = 0
        else:
            sorn.c.eat_ip = self.eta_ip
        sorn.c.ff_inhibition_broad = self.ff_inhibition_broad

    def run(self,sorn):
        """It should return a dictionary of objects to pickle (name,obj) including the sorn."""
        pass

    def plot_single(self,path,filename):
        pass




