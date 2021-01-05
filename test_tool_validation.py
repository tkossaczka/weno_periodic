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
df1=pd.read_csv("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Validation_set/parameters.txt")

# err_nt_max_vec = np.zeros(8)
# err_nt_mean_vec = np.zeros(8)
# err_t_max_vec = np.zeros(8)
# err_t_mean_vec = np.zeros(8)
# err_nt_JS_max_vec = np.zeros(8)
# err_nt_JS_mean_vec = np.zeros(8)

all_loss_test = []
for i in range(60):
    print(i)
    # sample_id = 50
    train_model = torch.load('C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Models/Model_59/{}.pt'.format(i))
    loss_test = []
    for j in [61,62,67,75,76,85,99]:
        sample_id = j
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
        # u_nt, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False)
        # for k in range(nn):
        #     u_nt = train_model.run_weno(problem_main, u_nt, mweno=True, mapped=False, vectorized=True, trainable=False, k=k)
        # u_nt = u_nt.detach().numpy()
        for k in range(nn + 1):
            single_problem_loss_test.append(exact_overflows_loss(u_test[:, k], u_ex[:, k]).detach().numpy().max())
        loss_test.append(single_problem_loss_test)
        #sample_id=sample_id+1
        # u_t = u_test[:,-1].detach().numpy()
        # error_nt_max = np.max(np.abs(u_nt - u_ex[:, -1].detach().numpy()))
        # error_t_max = np.max(np.abs(u_t - u_ex[:, -1].detach().numpy()))
        # error_nt_mean = np.mean((u_nt - u_ex[:, -1].detach().numpy()) ** 2)
        # error_t_mean = np.mean((u_t - u_ex[:, -1].detach().numpy()) ** 2)
        # err_nt_max_vec[j] = error_nt_max
        # err_t_max_vec[j] = error_t_max
        # err_nt_mean_vec[j] = error_nt_mean
        # err_t_mean_vec[j] = error_t_mean
        # _, x, t = problem_main.transformation(u_t)
        # plt.figure(j)
        # plt.plot(x, u_nt, color='blue', marker='o')
        # plt.plot(x, u_t, marker='o', color='green')
        # plt.plot(x, u_ex[:, -1], color="black")
    all_loss_test.append(loss_test)

all_loss_test = np.array(all_loss_test) #shape (training_steps, num_valid_problems, time_steps)

plt.plot(all_loss_test[:,:,-1])
a_labels = (df1["C"])
#plt.legend(['0.62','0.18','0.88', '0.77', '0.52', '0.3', '0.94'])
plt.xlabel('number of simulations')
plt.ylabel('LOSS')
plt.savefig("foo.pdf", bbox_inches='tight')

# err_mat = np.zeros((4,8))
# err_mat[0,:] = err_nt_max_vec
# err_mat[1,:] = err_t_max_vec
# err_mat[2,:] = err_nt_mean_vec
# err_mat[3,:] = err_t_mean_vec
