import torch
import numpy as np
import matplotlib.pyplot as plt
from define_WENO_Network import WENONetwork
from define_problem_transport_eq import transport_equation
from define_problem_Buckley_Leverett import Buckley_Leverett
from define_problem_Burgers_equation import Burgers_equation

torch.set_default_dtype(torch.float64)

train_model = torch.load('Models_on_generated_ex_sol/Model_10_60_36_0/0')

#problem = transport_equation
problem = Buckley_Leverett
#problem = Burgers_equation

rng = 7

def validation_problems(j):
    params_vld = []
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 0.25})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 0.5})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 0.7})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 1.0})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 1.3})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 1.6})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 2})
    return params_vld[j]

u_ex_0 = torch.load("Models_BL_on_ic_jump/u_ex_0")
u_ex_1 = torch.load("Models_BL_on_ic_jump/u_ex_1")
u_ex_2 = torch.load("Models_BL_on_ic_jump/u_ex_2")
u_ex_3 = torch.load("Models_BL_on_ic_jump/u_ex_3")
u_ex_4 = torch.load("Models_BL_on_ic_jump/u_ex_4")
u_ex_5 = torch.load("Models_BL_on_ic_jump/u_ex_5")
u_ex_6 = torch.load("Models_BL_on_ic_jump/u_ex_6")
u_exs = [u_ex_0, u_ex_1, u_ex_2, u_ex_3, u_ex_4, u_ex_5, u_ex_6]

u_ex_0_fine = torch.load("Models_BL_on_ic_jump/u_ex_0_fine")
u_ex_1_fine = torch.load("Models_BL_on_ic_jump/u_ex_1_fine")
u_ex_2_fine = torch.load("Models_BL_on_ic_jump/u_ex_2_fine")
u_ex_3_fine = torch.load("Models_BL_on_ic_jump/u_ex_3_fine")
u_ex_4_fine = torch.load("Models_BL_on_ic_jump/u_ex_4_fine")
u_ex_5_fine = torch.load("Models_BL_on_ic_jump/u_ex_5_fine")
u_ex_6_fine = torch.load("Models_BL_on_ic_jump/u_ex_6_fine")
u_exs_fine = [u_ex_0_fine, u_ex_1_fine, u_ex_2_fine, u_ex_3_fine, u_ex_4_fine, u_ex_5_fine, u_ex_6_fine]

C_vec = np.zeros(rng)
err_nt_max_vec = np.zeros(rng)
err_nt_mean_vec = np.zeros(rng)
err_t_max_vec = np.zeros(rng)
err_t_mean_vec = np.zeros(rng)
#u_nt_matr = np.zeros((101,rng))
#u_t_matr = np.zeros((101,rng))

for j in range(rng):
    print(j)
    params = validation_problems(j)
    my_problem = problem(ic_numb=6, space_steps=60, time_steps=None, params=params)
    #params=my_problem.params
    C_vec[j] = my_problem.params["C"]
    u_nt, nn = train_model.init_run_weno(my_problem, vectorized=True, just_one_time_step=False)
    for k in range(nn):
        u_nt = train_model.run_weno(my_problem, u_nt, mweno=True, mapped=False, vectorized=True, trainable=False, k=k)
    u_nt=u_nt.detach().numpy()
    u_t, nn = train_model.init_run_weno(my_problem, vectorized=True, just_one_time_step=False)
    for k in range(nn):
        u_t = train_model.run_weno(my_problem, u_t, mweno=True, mapped=False, vectorized=True, trainable=True, k=k)
    u_t = u_t.detach().numpy()
    _, x, t = my_problem.transformation(u_nt)
    time_steps = t.shape[0]
    # problem_ex = problem(ic_numb=2, space_steps=60 * 2 * 2 * 2, time_steps=None, params=params)
    # _, u_exact_adjusted = train_model.compute_exact_end(Buckley_Leverett, problem_ex, 60, time_steps, just_one_time_step=False, trainable=False)
    u_exact_adjusted = u_exs[j][:,-1]
    error_nt_max = np.max(np.abs(u_nt-u_exact_adjusted.detach().numpy()))
    error_t_max = np.max(np.abs(u_t-u_exact_adjusted.detach().numpy()))
    error_nt_mean = np.mean((u_nt-u_exact_adjusted.detach().numpy())**2)
    error_t_mean = np.mean((u_t-u_exact_adjusted.detach().numpy())**2)
    err_nt_max_vec[j] = error_nt_max
    err_t_max_vec[j] = error_t_max
    err_nt_mean_vec[j] = error_nt_mean
    err_t_mean_vec[j] = error_t_mean
    #u_nt_matr[:,j] = u_nt
    #u_t_matr[:,j] = u_t
    plt.figure(j+1)
    plt.plot(x, u_nt, color='blue', marker='o')
    plt.plot(x, u_t, marker='o', color='green')
    plt.plot(x, u_exact_adjusted)

err_mat = np.zeros((4,7))
err_mat[0,:] = err_nt_max_vec
err_mat[1,:] = err_t_max_vec
err_mat[2,:] = err_nt_mean_vec
err_mat[3,:] = err_t_mean_vec


# params = validation_problems(0)
# problem_ex_test = problem(ic_numb=6, space_steps=60 * 2 * 2 * 2, time_steps=None, params=params)
# _, u_ex_test = train_model.compute_exact(Buckley_Leverett, problem_ex_test, 60, 36, just_one_time_step=False, trainable=False)
# torch.save(u_ex_test, "u_ex_0_fine")



