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
Detailed parameters of network definition and training for experiments in our paper, including Counting task, Motion predition and Motion generation. If somewhere specified unclear, turn to the parameter scripts.

Table 1. Default Parameter for network structure

| N_e | N_i | N_u_e | N_u_i | eta_stdp | eta_ip | sp_prob | sp_initial | noise_sig | T_e_max | T_e_min | T_i_max | T_i_min |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 200 | 0.2* N_e | 0.05* N_e | 0 | 0.001 | 0.001 | 0.1 (0.4 for N_e = 400) | 0.001 | 0 | 1.0 | 0 | 0.5 | 0 |

Table 2. Default Parameter for RM-SORN training

| steps_train | steps_test | interval_train | interval_test |
| :-: | :-: | :-: | :-: |
| 20000 | 10000 | 100 | 500 |

Table 3. Network parameters for Counting task

| N_e | N_o | N_i | N_u_e | N_u_i | T_o_max | T_o_min | T_e_max | T_e_min | T_i_max | T_i_min | eta_stdp (W_ee) | eta_stdp (W_oe) | eta_ip_e | eta_ip_o | h_ip_e | h_ip_o | punishment | recurrent_reward | window_size |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 200 | 6 | 0.2* N_e | 0.1* N_e | 0 | 0.5 | 0 | 1.0 | 0 | 1.0 | 0 | 0.005| 0.001 | 0.001 | 0.005 | 0.1 | [0.05,0.4,0.05,0.05,0.4,0.05] | True | False | 0|

| N_e | N_o | N_i | N_u_e | N_u_i | T_o_max | T_o_min | T_e_max | T_e_min | T_i_max | T_i_min | eta_stdp (W_ee) | eta_stdp (W_oe) | eta_ip_e | eta_ip_o | h_ip_e | h_ip_o | punishment | recurrent_reward | window_size |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 400 | 6 | 0.2* N_e | 0.1* N_e | 0 | 0.5 | 0 | 1.0 | 0 | 1.0 | 0 | 0.005| 0.001 | 0.002 | 0.001 | 0.1 | [0.05,0.4,0.05,0.05,0.4,0.05] | True | False | 0|

Table 4. Network parameters for Motion predition

| N_e | N_o | N_i | N_u_e | N_u_i | T_o_max | T_o_min | T_e_max | T_e_min | T_i_max | T_i_min | eta_stdp (W_ee) | eta_stdp (W_oe) | eta_ip_e | eta_ip_o | h_ip_e | h_ip_o | punishment | recurrent_reward | window_size |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 200 | n | 0.2* N_e | 0.15* N_e | 0 | 1.0 | 0 | 0.5 | 0 | 1.0 | 0 | 0.001| 0.005 | 0.002 | 0.005 | 0.2 | 1/n | True | False | 20|

Table 4. Network parameters for Motion generation

| N_e | N_o | N_i | N_u_e | N_u_i | T_o_max | T_o_min | T_e_max | T_e_min | T_i_max | T_i_min | eta_stdp (W_ee) | eta_stdp (W_oe) | eta_ip_e | eta_ip_o | h_ip_e | h_ip_o | punishment | recurrent_reward | window_size |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 200 | n | 0.2* N_e | 0.05* N_e | 0 | 0.5 | 0 | 1.0 | 0 | 1.0 | 0 | 0.01| 0.01 | 0.001 | 0.005 | 0.1 | 1/n | True | False | 10|

## Some results for three tasks

