import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from initial_jump_generator import init_jump
from define_WENO_Network import WENONetwork
from define_problem_transport_eq import transport_equation
from define_problem_Buckley_Leverett import Buckley_Leverett
from define_problem_Burgers_equation import Burgers_equation
from initial_condition_switch import init_cond

torch.set_default_dtype(torch.float64)

#problem = transport_equation
problem = Buckley_Leverett
#problem = Burgers_equation

def exact_overflows_loss(u, u_ex):
    u_max = torch.Tensor([1.0])
    u_min = torch.Tensor([0.0])
    overflows = torch.sum(torch.abs(torch.min(u, u_min)-u_min) + torch.max(u, u_max)-u_max )
    error = train_model.compute_error(u, u_ex)
    loss = error + overflows
    return loss

rng = 10

err_nt_max_vec = np.zeros(rng)
err_nt_mean_vec = np.zeros(rng)
err_t_max_vec = np.zeros(rng)
err_t_mean_vec = np.zeros(rng)
all_loss_test = []

df=pd.read_csv("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Test_set/parameters.txt")

for i in range(10):
    train_model = torch.load('C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Models/Model_00/{}'.format(i))
    loss_test = []
    for j in range(400,410):
        single_problem_loss_test = []
        print(j)
        u_ex = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Test_set/u_exact60_{}.npy".format(j))
        u_ex = torch.Tensor(u_ex)
        # width = float(df[df.sample_id == j]["width"])
        # height = float(df[df.sample_id == j]["height"])
        C = float(df[df.sample_id==j]["C"])
        params = {'T': 0.4, 'e': 1e-13, 'L': -1, 'R': 1, 'C': C}
        my_problem = problem(ic_numb=6, space_steps=64, time_steps=None, params=params)
        #my_problem.initial_condition, _, _, _, _ = init_jump(my_problem.x, numb=1, xmid=1, height=height, width=width)
        #my_problem.initial_condition = torch.Tensor(my_problem.initial_condition)
        u_t, nn = train_model.init_run_weno(my_problem, vectorized=False, just_one_time_step=False)
        for k in range(nn):
            uu_t = train_model.run_weno(my_problem, u_t, mweno=True,mapped=False,vectorized=False,trainable=False,k=k)
            u_t[:,k+1]=uu_t
        _, x, t = my_problem.transformation(u_t)
        for k in range(nn + 1):
            single_problem_loss_test.append(exact_overflows_loss(u_t[:,k], u_ex[:, k]).detach().numpy().max())
        loss_test.append(single_problem_loss_test)
    all_loss_test.append(loss_test)

all_loss_test = np.array(all_loss_test)
plt.plot(all_loss_test[:,:,-1])
