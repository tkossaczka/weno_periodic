import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from initial_condition_Burgers import init_cond_B
from initial_jump_generator import init_jump
from define_WENO_Network import WENONetwork
from define_problem_transport_eq import transport_equation
from define_problem_Buckley_Leverett import Buckley_Leverett
from define_problem_Burgers_equation import Burgers_equation

torch.set_default_dtype(torch.float64)

#problem = transport_equation
#problem = Buckley_Leverett
problem = Burgers_equation

rng = 15

err_nt_max_vec = np.zeros(rng)
err_nt_mean_vec = np.zeros(rng)
err_t_max_vec = np.zeros(rng)
err_t_mean_vec = np.zeros(rng)

# df=pd.read_csv("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1/Test_set/parameters.txt")
df=pd.read_csv("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1/parameters.txt")

# train_model = torch.load('C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Models/Model_07/16')
#train_model = torch.load('C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Old_models/Models_BL_on_ic_numb_6/model3')
train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Models/Model_02/49")


ll = 160
for j in range(rng):
    sample_id = ll
    #u_ex = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1/Test_set/u_exact60_{}.npy".format(sample_id))
    u_ex = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1/Test_set/u_exact64_{}.npy".format(sample_id))
    u_ex = torch.Tensor(u_ex)
    print(j)
    # width = float(df[df.sample_id == sample_id]["width"])
    # height = float(df[df.sample_id == sample_id]["height"])
    # C = float(df[df.sample_id == sample_id]["C"])
    # params = {'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': C}
    ic_id = float(df[df.sample_id == sample_id]["ic_id"])
    kk = float(df[df.sample_id == sample_id]["k"])
    params = None
    my_problem = problem(ic_numb=ic_id, space_steps=64, time_steps=None, params=params)
    my_problem.initial_condition, _, _, _, _, _ = init_cond_B(ic_id, my_problem.x, kk)
    #my_problem.initial_condition, _, _, _, _ = init_jump(my_problem.x, numb=1, xmid=1, height=height, width=width)
    my_problem.initial_condition = torch.Tensor(my_problem.initial_condition)
    u_t, nn = train_model.init_run_weno(my_problem, vectorized=True, just_one_time_step=False)
    u_nt = u_t
    for k in range(nn):
        u_nt = train_model.run_weno(my_problem, u_nt, mweno=True, mapped=False, vectorized=True, trainable=False, k=k)
    for k in range(nn):
        u_t = train_model.run_weno(my_problem, u_t, mweno=True, mapped=False, vectorized=True, trainable=True, k=k)
    u_t = u_t.detach().numpy()
    u_nt = u_nt.detach().numpy()
    error_nt_max = np.max(np.abs(u_nt - u_ex[:,-1].detach().numpy()))
    error_t_max = np.max(np.abs(u_t - u_ex[:,-1].detach().numpy()))
    error_nt_mean = np.mean((u_nt - u_ex[:,-1].detach().numpy()) ** 2)
    error_t_mean = np.mean((u_t - u_ex[:,-1].detach().numpy()) ** 2)
    err_nt_max_vec[j] = error_nt_max
    err_t_max_vec[j] = error_t_max
    err_nt_mean_vec[j] = error_nt_mean
    err_t_mean_vec[j] = error_t_mean
    _, x, t = my_problem.transformation(u_nt)
    plt.figure(j + 1)
    plt.plot(x, u_nt, color='blue', marker='o')
    plt.plot(x, u_t, marker='o', color='green')
    plt.plot(x, u_ex[:,-1])
    ll = ll + 1

err_mat = np.zeros((4,rng))
err_mat[0,:] = err_nt_max_vec
err_mat[1,:] = err_t_max_vec
err_mat[2,:] = err_nt_mean_vec
err_mat[3,:] = err_t_mean_vec

