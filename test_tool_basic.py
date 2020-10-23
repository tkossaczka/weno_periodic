import torch
import numpy as np
import matplotlib.pyplot as plt
from initial_condition_Burgers import init_cond_B
from define_WENO_Network import WENONetwork
from define_problem_transport_eq import transport_equation
from define_problem_Buckley_Leverett import Buckley_Leverett
from define_problem_Burgers_equation import Burgers_equation

torch.set_default_dtype(torch.float64)

train_model = WENONetwork()

train_model = torch.load('C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Models/Model_10/16')
#train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Models/Model_04/0")

#problem = transport_equation
problem = Buckley_Leverett
#problem = Burgers_equation

rng = 7
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
u_exs = [u_ex_0, u_ex_1, u_ex_2, u_ex_3, u_ex_4, u_ex_5, u_ex_6]

# rng = 8
# def validation_problems(j):
#     params_vld = []
#     params_vld.append({'ic_id': 1, 'k': 30})
#     params_vld.append({'ic_id': 1, 'k': 15})
#     params_vld.append({'ic_id': 2, 'k': 0.5})
#     params_vld.append({'ic_id': 2, 'k': 1})
#     params_vld.append({'ic_id': 2, 'k': 1.5})
#     params_vld.append({'ic_id': 3, 'k': 0.5})
#     params_vld.append({'ic_id': 3, 'k': 1})
#     params_vld.append({'ic_id': 3, 'k': 1.5})
#     return params_vld[j]
# u_ex_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_2/Basic_test_set/u_ex64_0")
# u_ex_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_2/Basic_test_set/u_ex64_1")
# u_ex_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_2/Basic_test_set/u_ex64_2")
# u_ex_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_2/Basic_test_set/u_ex64_3")
# u_ex_4 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_2/Basic_test_set/u_ex64_4")
# u_ex_5 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_2/Basic_test_set/u_ex64_5")
# u_ex_6 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_2/Basic_test_set/u_ex64_6")
# u_ex_7 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_2/Basic_test_set/u_ex64_7")
# u_exs = [u_ex_0, u_ex_1, u_ex_2, u_ex_3, u_ex_4, u_ex_5, u_ex_6, u_ex_7]


err_nt_max_vec = np.zeros(rng)
err_nt_mean_vec = np.zeros(rng)
err_t_max_vec = np.zeros(rng)
err_t_mean_vec = np.zeros(rng)


for j in range(rng):
    print(j)
    # Burger
    # params_vld = validation_problems(j)
    # ic_id_test = params_vld['ic_id']
    # kkkk = params_vld['k']
    # params = None
    # my_problem = problem(ic_numb=1, space_steps=64, time_steps=None, params=params)
    # my_problem.initial_condition, _, _, _, _, _ = init_cond_B(ic_id_test, my_problem.x, kkkk)
    # my_problem.initial_condition = torch.Tensor(my_problem.initial_condition)
    # Buckley
    params = validation_problems(j)
    my_problem = problem(ic_numb=6, space_steps=64, time_steps=None, params=params)
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
    plt.figure(j+1)
    plt.plot(x, u_nt, color='blue', marker='o')
    plt.plot(x, u_t, marker='o', color='green')
    plt.plot(x, u_exact_adjusted)

err_mat = np.zeros((4,rng))
err_mat[0,:] = err_nt_max_vec
err_mat[1,:] = err_t_max_vec
err_mat[2,:] = err_nt_mean_vec
err_mat[3,:] = err_t_mean_vec


# params = validation_problems(0)
# problem_ex_test = problem(ic_numb=6, space_steps=60 * 2 * 2 * 2, time_steps=None, params=params)
# _, u_ex_test = train_model.compute_exact(Buckley_Leverett, problem_ex_test, 60, 36, just_one_time_step=False, trainable=False)
# torch.save(u_ex_test, "u_ex_0_fine")

# def validation_problems(j):
#     params_vld = []
#     params_vld.append({'ic_id': 1, 'k': 30})
#     params_vld.append({'ic_id': 1, 'k': 15})
#     params_vld.append({'ic_id': 2, 'k': 0.5})
#     params_vld.append({'ic_id': 2, 'k': 1})
#     params_vld.append({'ic_id': 2, 'k': 1.5})
#     params_vld.append({'ic_id': 3, 'k': 0.5})
#     params_vld.append({'ic_id': 3, 'k': 1})
#     params_vld.append({'ic_id': 3, 'k': 1.5})
#     return params_vld[j]
#
# #params = None
# params = {'T': 0.4, 'e': 1e-13, 'L': -1, 'R': 1, 'C': 0.9}
# problem_ex_test = problem(ic_numb=6, space_steps=512 , time_steps=None, params=params)
# #problem_ex_test.initial_condition, _, _, _, _, _ = init_cond_B(3, problem_ex_test.x, 1.5)
# #problem_ex_test.initial_condition = torch.Tensor(problem_ex_test.initial_condition)
# u_ex, u_ex64 = train_model.compute_exact(Buckley_Leverett, problem_ex_test, 64, 35, just_one_time_step=False, trainable=False)
# torch.save(u_ex64, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex64_6")
# torch.save(u_ex, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex_6")



