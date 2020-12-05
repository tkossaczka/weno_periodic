from define_WENO_Network import WENONetwork
import torch
from torch import optim
import pandas as pd
import random
from define_problem_transport_eq import transport_equation
from define_problem_Buckley_Leverett import Buckley_Leverett
from define_problem_Burgers_equation import Burgers_equation
import numpy as np
import matplotlib.pyplot as plt
from initial_jump_generator import init_jump
import os, sys

torch.set_default_dtype(torch.float64)

# TRAIN NETWORK
train_model = WENONetwork()

# DROP PROBLEM FOR TRAINING
problem_main = Buckley_Leverett
#problem_class = transport_equation

def exact_overflows_loss(u, u_ex):
    u_max = torch.Tensor([1.0])
    u_min = torch.Tensor([0.0])
    overflows = torch.sum(torch.abs(torch.min(u, u_min)-u_min) + torch.max(u, u_max)-u_max )
    error = train_model.compute_error(u, u_ex)
    loss = error + overflows
    return loss

def validation_problems(j):
    params_vld = []
    params_vld.append({'T': 0.4, 'e': 1e-13, 'L': -1, 'R': 1, 'C': 0.25})
    params_vld.append({'T': 0.4, 'e': 1e-13, 'L': -1, 'R': 1, 'C': 0.4})
    params_vld.append({'T': 0.4, 'e': 1e-13, 'L': -1, 'R': 1, 'C': 0.5})
    params_vld.append({'T': 0.4, 'e': 1e-13, 'L': -1, 'R': 1, 'C': 0.6})
    params_vld.append({'T': 0.4, 'e': 1e-13, 'L': -1, 'R': 1, 'C': 0.7})
    params_vld.append({'T': 0.4, 'e': 1e-13, 'L': -1, 'R': 1, 'C': 0.8})
    params_vld.append({'T': 0.4, 'e': 1e-13, 'L': -1, 'R': 1, 'C': 0.9})
    return params_vld[j]
u_ex_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex64_0")
u_ex_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex64_1")
u_ex_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex64_2")
u_ex_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex64_3")
u_ex_4 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex64_4")
u_ex_5 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex64_5")
u_ex_6 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex64_6")
u_ex_whole_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex_0")
u_ex_whole_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex_1")
u_ex_whole_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex_2")
u_ex_whole_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex_3")
u_ex_whole_4 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex_4")
u_ex_whole_5 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex_5")
u_ex_whole_6 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex_6")
u_exs_whole = [u_ex_whole_0, u_ex_whole_1, u_ex_whole_2, u_ex_whole_3, u_ex_whole_4, u_ex_whole_5, u_ex_whole_6]
u_exs = [u_ex_whole_0[0:512 + 1:4, 0:2240 + 1:16], u_ex_whole_1[0:512 + 1:4, 0:2240 + 1:16], u_ex_whole_2[0:512 + 1:4, 0:2240 + 1:16], u_ex_whole_3[0:512 + 1:4, 0:2240 + 1:16], u_ex_whole_4[0:512 + 1:4, 0:2240 + 1:16], u_ex_whole_5[0:512 + 1:4, 0:2240 + 1:16], u_ex_whole_6[0:512 + 1:4, 0:2240 + 1:16]]

all_loss_test = []
for i in range(30):
    print(i)
    train_model = torch.load('C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Models/Model_30/{}.pt'.format(i))
    loss_test = []
    for j in range(7):
        print(j)
        single_problem_loss_test = []
        params = validation_problems(j)
        my_problem = problem_main(ic_numb=6, space_steps=64 * 2, time_steps=None, params=params)
        u_init, nn = train_model.init_run_weno(my_problem, vectorized=False, just_one_time_step=False)
        u_test = u_init
        for k in range(nn):
            uu_test = train_model.run_weno(my_problem, u_test, mweno=True, mapped=False, vectorized=False, trainable=True, k=k)
            u_test[:, k + 1] = uu_test
        for k in range(nn + 1):
            single_problem_loss_test.append(exact_overflows_loss(u_test[:, k], u_exs[j][:, k]).detach().numpy().max())
        loss_test.append(single_problem_loss_test)
    all_loss_test.append(loss_test)

all_loss_test = np.array(all_loss_test) #shape (training_steps, num_valid_problems, time_steps)
plt.plot(all_loss_test[:,:,-1])