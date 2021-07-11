import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from initial_condition_Burgers import init_cond_B
from define_WENO_Network import WENONetwork
from define_problem_transport_eq import transport_equation
from define_problem_Buckley_Leverett import Buckley_Leverett
from define_problem_Burgers_equation import Burgers_equation

torch.set_default_dtype(torch.float64)
train_model = WENONetwork()

#problem = transport_equation
problem = Buckley_Leverett
# problem = Burgers_equation

train_model = torch.load('C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Models/Model_65/46.pt')  # 30/10 good

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
u_ex_0_w = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex_0")
u_ex_1_w = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex_1")
u_ex_2_w = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex_2")
u_ex_3_w = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex_3")
u_ex_4_w = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex_4")
u_ex_5_w = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex_5")
u_ex_6_w = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex_6")
divider_space = 4
divider_time = 16
u_exs_256 = [u_ex_0_w[0:1024 + 1:divider_space, 0:8960 + 1:divider_time], u_ex_1_w[0:1024 + 1:divider_space, 0:8960 + 1:divider_time],
         u_ex_2_w[0:1024 + 1:divider_space, 0:8960 + 1:divider_time], u_ex_3_w[0:1024 + 1:divider_space, 0:8960 + 1:divider_time],
         u_ex_4_w[0:1024 + 1:divider_space, 0:8960 + 1:divider_time], u_ex_5_w[0:1024 + 1:divider_space, 0:8960 + 1:divider_time],
         u_ex_6_w[0:1024 + 1:divider_space, 0:8960 + 1:divider_time]]
divider_space = 4*2
divider_time = 16*4
u_exs_128 = [u_ex_0_w[0:1024 + 1:divider_space, 0:8960 + 1:divider_time], u_ex_1_w[0:1024 + 1:divider_space, 0:8960 + 1:divider_time],
         u_ex_2_w[0:1024 + 1:divider_space, 0:8960 + 1:divider_time], u_ex_3_w[0:1024 + 1:divider_space, 0:8960 + 1:divider_time],
         u_ex_4_w[0:1024 + 1:divider_space, 0:8960 + 1:divider_time], u_ex_5_w[0:1024 + 1:divider_space, 0:8960 + 1:divider_time],
         u_ex_6_w[0:1024 + 1:divider_space, 0:8960 + 1:divider_time]]
divider_space = 4*2*2
divider_time = 16*4*4
u_exs_64 = [u_ex_0_w[0:1024 + 1:divider_space, 0:8960 + 1:divider_time], u_ex_1_w[0:1024 + 1:divider_space, 0:8960 + 1:divider_time],
         u_ex_2_w[0:1024 + 1:divider_space, 0:8960 + 1:divider_time], u_ex_3_w[0:1024 + 1:divider_space, 0:8960 + 1:divider_time],
         u_ex_4_w[0:1024 + 1:divider_space, 0:8960 + 1:divider_time], u_ex_5_w[0:1024 + 1:divider_space, 0:8960 + 1:divider_time],
         u_ex_6_w[0:1024 + 1:divider_space, 0:8960 + 1:divider_time]]
u_exs_whole = [u_ex_0_w, u_ex_1_w, u_ex_2_w, u_ex_3_w, u_ex_4_w, u_ex_5_w, u_ex_6_w]
rng=7

err_nt_max_vec = np.zeros(rng)
err_nt_mean_vec = np.zeros(rng)
err_t_mean_vec = np.zeros(rng)
err_t_max_vec = np.zeros(rng)
time_nt = np.zeros(rng)
time_t = np.zeros(rng)

for j in range(rng):
    params = validation_problems(j)
    sp_steps = 64*2
    u_exact_adjusted = u_exs_128[j][:, -1]
    my_problem = problem(ic_numb=6, space_steps=sp_steps , time_steps=None, params=params)
    u_nt, nn = train_model.init_run_weno(my_problem, vectorized=True, just_one_time_step=False)
    start_time = time.time()
    for k in range(nn):
        u_nt = train_model.run_weno(my_problem, u_nt, mweno=True, mapped=False, vectorized=True, trainable=False, k=k)
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
    u_nt = u_nt.detach().numpy()
    error_nt_max = np.max(np.abs(u_nt - u_exact_adjusted.detach().numpy()))
    error_nt_mean = np.sqrt(2 / sp_steps) * (np.sqrt(np.sum((u_nt - u_exact_adjusted.detach().numpy()) ** 2)))
    err_nt_max_vec[j] = error_nt_max
    err_nt_mean_vec[j] = error_nt_mean
    time_nt[j] = end_time - start_time

# for j in range(rng):
#     print(j)
#     params = validation_problems(j)
#     my_problem = problem(ic_numb=6, space_steps=64 * 2 , time_steps=None, params=params)
#     u_t, nn = train_model.init_run_weno(my_problem, vectorized=True, just_one_time_step=False)
#     start_time = time.time()
#     for k in range(nn):
#         u_t = train_model.run_weno(my_problem, u_t, mweno=True, mapped=False, vectorized=True, trainable=True, k=k)
#     end_time = time.time()
#     print("--- %s seconds ---" % (end_time - start_time))
#     u_t = u_t.detach().numpy()
#     u_exact_adjusted = u_exs_128[j][:, -1]
#     sp_steps = my_problem.space_steps
#     error_t_max = np.max(np.abs(u_t - u_exact_adjusted.detach().numpy()))
#     error_t_mean = np.sqrt(2/sp_steps)*(np.sqrt(np.sum((u_t-u_exact_adjusted.detach().numpy())**2)))
#     err_t_max_vec[j] = error_t_max
#     err_t_mean_vec[j] = error_t_mean
#     time_t[j] = end_time - start_time
