# RM-SORN
This code contains the necessary files to simulate the RM-SORN model proposed in paper *A Reward-Modulated Self-Organizing Recurrent
Network based on Neural Plasticity Mechanisms*, which is under review in *International Joint Conference of Neural Network (IJCNN)*.
Our RM-SORN model is based on previous SORN and, the network implementation (souce code) is based on the SORN implementation by [Hartmann et al.](https://github.com/chrhartm/SORN).
## Code organization
This code is set up in a modular manner.
* `common` folder contains code common to all experiments ,including `rmsorn`, `synapses`, `sources`, `experiment` and `experiments`.
* `utils` folder contains code for `Bunch` (a container to replace dict in RM-SORN) and other tool scripts, including `backup`, `datalog`, `pca` and `plotting`.
* `examples` folder contains code for specific experiment parameters and process, implemented by `param_xxxx` and `experiment_xxxxTask`, respectively.
Notably, the default parameters script is included in `common` folder, user can cover the default params via setting `param_xxxx` in `examples`. 
