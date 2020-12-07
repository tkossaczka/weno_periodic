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
train_model = WENONetwork()
problem = Buckley_Leverett

def exact_overflows_loss(u, u_ex):
    u_max = torch.Tensor([1.0])
    u_min = torch.Tensor([0.0])
    overflows = torch.sum(torch.abs(torch.min(u, u_min)-u_min) + torch.max(u, u_max)-u_max )
    error = train_model.compute_error(u, u_ex)
    loss = error + overflows
    return loss

df=pd.read_csv("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/parameters.txt")

all_loss_test = []
for i in range(30):
    print(i)
    sample_id = 100
    train_model = torch.load('C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Models/Model_31/{}.pt'.format(i))
    loss_test = []
    for j in range(8):
        u_ex = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Validation_set/u_exact128_{}.npy".format(sample_id))
        #u_ex = u_ex[0:512 + 1:4, 0:2240 + 1:16]
        u_ex = torch.Tensor(u_ex)
        C = float(df[df.sample_id == sample_id]["C"])
        print(C)
        params = {'T': 0.4, 'e': 1e-13, 'L': -1, 'R': 1, 'C': C}
        print(j)
        single_problem_loss_test = []
        problem_main = problem(ic_numb=6, space_steps=128, time_steps=None, params=params)
        params = problem_main.get_params()
        print(params)
        u_init, nn = train_model.init_run_weno(problem_main, vectorized=False, just_one_time_step=False)
        u_test = u_init
        for k in range(nn):
            uu_test = train_model.run_weno(problem_main, u_test, mweno=True, mapped=False, vectorized=False, trainable=True, k=k)
            u_test[:, k + 1] = uu_test
        for k in range(nn + 1):
            single_problem_loss_test.append(exact_overflows_loss(u_test[:, k], u_ex[:, k]).detach().numpy().max())
        loss_test.append(single_problem_loss_test)
        sample_id=sample_id+1
    all_loss_test.append(loss_test)

all_loss_test = np.array(all_loss_test) #shape (training_steps, num_valid_problems, time_steps)
plt.plot(all_loss_test[:,:,-1])

