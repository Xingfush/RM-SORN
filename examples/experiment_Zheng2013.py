from __future__ import division
import utils

from common.sources import NoSource
from common.experiments import AbstractExperiment

class Experiment_test(AbstractExperiment):
    def start(self):
        super().start()
        c = self.params.c
        self.inputsource = NoSource()
        return self.inputsource

    def reset(self,sorn):
        super().reset(sorn)
        c = self.params.c
        sorn.__init__(c,self.inputsource)

    def run(self,sorn):
        super().run(sorn)
        c = self.params.c
        sorn.simulation(c.steps_plastic)
        return {'source_plastic':self.inputsource}

