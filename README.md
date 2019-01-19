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
## Getting started
The tasks for RM-SORN here is the same as in our paper, including: counting task, Motion prediction and Motion generation.
To run experiments, navigate to the `common` folder and run `python test_single.py param_xxxx`.
If you are planning to define extra task, please follow the  format setting in `experiments` in `common` to accomplish `experiment_xxxxTask`, as well as a `param_xxxx` file.
In particular, `experiment_RMTask` is utilized to run the simulation on varied task difficult $n$, 10 networks per task.
## Params for network definition and training
Detailed parameters of network definition and training for experiments in our paper, including Counting task, Motion predition and Motion generation.
Table 1. Default Parameter for network structure

| N_e | N_i | N_u_e | N_u_i | eta_stdp | eta_ip | sp_prob | sp_initial | noise_sig | T_e_max | T_e_min | T_i_max | T_i_min |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 200 | 40 | 10 | 0 | 0.001 | 0.001 | 0.1 | 0.001 | 0 | 1.0 | 0 | 0.5 | 0 |
