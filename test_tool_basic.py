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


#problem = transport_equation
# problem = Buckley_Leverett
problem = Burgers_equation

if problem == Buckley_Leverett:
    train_model = torch.load('C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Models/Model_51/0.pt') #30/10 good
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
    u_ex_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex128_0")
    u_ex_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex128_1")
    u_ex_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex128_2")
    u_ex_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex128_3")
    u_ex_4 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex128_4")
    u_ex_5 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex128_5")
    u_ex_6 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex128_6")
    u_ex_0_w = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex_0")
    u_ex_1_w = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex_1")
    u_ex_2_w = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex_2")
    u_ex_3_w = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex_3")
    u_ex_4_w = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex_4")
    u_ex_5_w = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex_5")
    u_ex_6_w = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex_6")
    # divider_space = 4*2
    # divider_time = 16*4
    # u_exs = [u_ex_whole_0[0:512 + 1:divider_space, 0:2240 + 1:divider_time], u_ex_whole_1[0:512 + 1:divider_space, 0:2240 + 1:divider_time],
    #          u_ex_whole_2[0:512 + 1:divider_space, 0:2240 + 1:divider_time], u_ex_whole_3[0:512 + 1:divider_space, 0:2240 + 1:divider_time],
    #          u_ex_whole_4[0:512 + 1:divider_space, 0:2240 + 1:divider_time], u_ex_whole_5[0:512 + 1:divider_space, 0:2240 + 1:divider_time],
    #          u_ex_whole_6[0:512 + 1:divider_space, 0:2240 + 1:divider_time]]
    u_exs = [u_ex_0, u_ex_1, u_ex_2, u_ex_3, u_ex_4, u_ex_5, u_ex_6]
    u_exs_whole = [u_ex_0_w, u_ex_1_w, u_ex_2_w, u_ex_3_w, u_ex_4_w, u_ex_5_w, u_ex_6_w]
elif problem == Burgers_equation:
    train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Models/Model_61/47.pt")
    rng = 14
    def validation_problems(j):
        params_vld = []
        params_vld.append({'ic_id': 1, 'k': 30})
        params_vld.append({'ic_id': 1, 'k': 15})
        params_vld.append({'ic_id': 2, 'k': 0.5})
        params_vld.append({'ic_id': 2, 'k': 1})
        params_vld.append({'ic_id': 2, 'k': 1.5})
        params_vld.append({'ic_id': 3, 'k': 0.5})
        params_vld.append({'ic_id': 3, 'k': 1})
        params_vld.append({'ic_id': 3, 'k': 1.5})
        params_vld.append({'ic_id': 3, 'k': 3})
        params_vld.append({'ic_id': 3, 'k': 4})
        params_vld.append({'ic_id': 3, 'k': 5})
        params_vld.append({'ic_id': 3, 'k': 2})
        params_vld.append({'ic_id': 3, 'k': 3.5})
        params_vld.append({'ic_id': 3, 'k': 4.5})
        return params_vld[j]
    u_ex_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex128_0")
    u_ex_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex128_1")
    u_ex_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex128_2")
    u_ex_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex128_3")
    u_ex_4 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex128_4")
    u_ex_5 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex128_5")
    u_ex_6 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex128_6")
    u_ex_7 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex128_7")
    u_ex_8 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex128_8")
    u_ex_9 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex128_9")
    u_ex_10 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex128_10")
    u_ex_11 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex128_11")
    u_ex_12 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex128_12")
    u_ex_13 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex128_13")
    u_ex_whole_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex_0")
    u_ex_whole_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex_1")
    u_ex_whole_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex_2")
    u_ex_whole_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex_3")
    u_ex_whole_4 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex_4")
    u_ex_whole_5 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex_5")
    u_ex_whole_6 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex_6")
    u_ex_whole_7 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex_7")
    u_ex_whole_8 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex_8")
    u_ex_whole_9 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex_9")
    u_ex_whole_10 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex_10")
    u_ex_whole_11 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex_11")
    u_ex_whole_12 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex_12")
    u_ex_whole_13 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex_13")
    # u_exs = [u_ex_whole_0[0:512 + 1:4, 0:1600 + 1:16], u_ex_whole_1[0:512 + 1:4, 0:1600 + 1:16],
    #          u_ex_whole_2[0:512 + 1:4, 0:1600 + 1:16], u_ex_whole_3[0:512 + 1:4, 0:1600 + 1:16],
    #          u_ex_whole_4[0:512 + 1:4, 0:1600 + 1:16], u_ex_whole_5[0:512 + 1:4, 0:1600 + 1:16],
    #          u_ex_whole_6[0:512 + 1:4, 0:1600 + 1:16], u_ex_whole_7[0:512 + 1:4, 0:1600 + 1:16]]
    u_exs = [u_ex_0, u_ex_1, u_ex_2, u_ex_3, u_ex_4, u_ex_5, u_ex_6, u_ex_7,u_ex_8, u_ex_9,u_ex_10,u_ex_11,u_ex_12,u_ex_13]
    u_exs_whole = [u_ex_whole_0, u_ex_whole_1, u_ex_whole_2, u_ex_whole_3, u_ex_whole_4, u_ex_whole_5, u_ex_whole_6, u_ex_whole_7,u_ex_whole_8, u_ex_whole_9,u_ex_whole_10,u_ex_whole_11,u_ex_whole_12,u_ex_whole_13]


err_nt_max_vec = np.zeros(rng)
err_nt_mean_vec = np.zeros(rng)
err_t_max_vec = np.zeros(rng)
err_t_mean_vec = np.zeros(rng)
err_nt_JS_max_vec = np.zeros(rng)
err_nt_JS_mean_vec = np.zeros(rng)

for j in range(rng):
    print(j)
    if problem == Buckley_Leverett:
        params = validation_problems(j)
        #params['T'] = 0.2
        my_problem = problem(ic_numb=6, space_steps=64*2, time_steps=None, params=params)
        C = params['C']
    elif problem == Burgers_equation:
        params_vld = validation_problems(j)
        ic_id_test = params_vld['ic_id']
        kkkk = params_vld['k']
        params = None
        my_problem = problem(ic_numb=1, space_steps=64*2, time_steps=None, params=params)
        my_problem.initial_condition, _, _, _, _, _ = init_cond_B(ic_id_test, my_problem.x, kkkk)
        my_problem.initial_condition = torch.Tensor(my_problem.initial_condition)
    u_nt, nn = train_model.init_run_weno(my_problem, vectorized=True, just_one_time_step=False)
    for k in range(nn):
        u_nt = train_model.run_weno(my_problem, u_nt, mweno=True, mapped=False, vectorized=True, trainable=False, k=k)
    u_nt=u_nt.detach().numpy()
    u_nt_JS, nn = train_model.init_run_weno(my_problem, vectorized=True, just_one_time_step=False)
    for k in range(nn):
        u_nt_JS = train_model.run_weno(my_problem, u_nt_JS, mweno=False, mapped=False, vectorized=True, trainable=False, k=k)
    u_nt_JS=u_nt_JS.detach().numpy()
    u_t, nn = train_model.init_run_weno(my_problem, vectorized=True, just_one_time_step=False)
    for k in range(nn):
        u_t = train_model.run_weno(my_problem, u_t, mweno=True, mapped=False, vectorized=True, trainable=True, k=k)
    u_t = u_t.detach().numpy()
    _, x, t = my_problem.transformation(u_nt)
    time_steps = t.shape[0]
    h = my_problem.h
    x_ex = np.linspace(0, 2 - h, 1024)
    # x_ex = np.linspace(-1, 1 - h, 1024)
    # problem_ex = problem(ic_numb=2, space_steps=60 * 2 * 2 * 2, time_steps=None, params=params)
    # _, u_exact_adjusted = train_model.compute_exact_end(Buckley_Leverett, problem_ex, 60, time_steps, just_one_time_step=False, trainable=False)
    u_exact_adjusted = u_exs[j][:,-1] #[0:128 + 1:2, 0:100 + 1:4]
    u_exact = u_exs_whole[j][:,-1]
    error_nt_max = np.max(np.abs(u_nt-u_exact_adjusted.detach().numpy()))
    error_nt_JS_max = np.max(np.abs(u_nt_JS-u_exact_adjusted.detach().numpy()))
    error_t_max = np.max(np.abs(u_t-u_exact_adjusted.detach().numpy()))
    error_nt_mean = np.sqrt(2 / 128) * (np.sqrt(np.sum((u_nt - u_exact_adjusted.detach().numpy()) ** 2)))
    error_nt_JS_mean = np.sqrt(2 / 128) * (np.sqrt(np.sum((u_nt_JS - u_exact_adjusted.detach().numpy()) ** 2)))
    error_t_mean = np.sqrt(2 / 128) * (np.sqrt(np.sum((u_t - u_exact_adjusted.detach().numpy()) ** 2)))
    err_nt_max_vec[j] = error_nt_max
    err_nt_JS_max_vec[j] = error_nt_JS_max
    err_t_max_vec[j] = error_t_max
    err_nt_mean_vec[j] = error_nt_mean
    err_nt_JS_mean_vec[j] = error_nt_JS_mean
    err_t_mean_vec[j] = error_t_mean
    plt.figure(j+1)
    plt.plot(x, u_nt, color='blue', marker='o')
    plt.plot(x, u_nt_JS, color='green', marker='o')
    plt.plot(x, u_t, marker='o', color='red')
    plt.plot(x_ex, u_exact, color='black')

err_mat = np.zeros((6,rng))
err_mat[0,:] = err_nt_JS_max_vec
err_mat[1,:] = err_nt_max_vec
err_mat[2,:] = err_t_max_vec
err_mat[3,:] = err_nt_JS_mean_vec
err_mat[4,:] = err_nt_mean_vec
err_mat[5,:] = err_t_mean_vec

# err_mat = np.zeros((rng,6))
# err_mat[:,0] = err_nt_JS_max_vec
# err_mat[:,3] = err_nt_JS_mean_vec
# err_mat[:,1] = err_nt_max_vec
# err_mat[:,4] = err_nt_mean_vec
# err_mat[:,2] = err_t_max_vec
# err_mat[:,5] = err_t_mean_vec
# err_mat=err_mat.T
#
# import pandas as pd
# pd.DataFrame(err_mat).to_csv("err_mat.csv")
# pd.DataFrame(err_mat).to_latex()

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

# def validation_problems(j):
#     params_vld = []
#     params_vld.append({'T': 0.4, 'e': 1e-13, 'L': -1, 'R': 1, 'C': 0.25})
#     params_vld.append({'T': 0.4, 'e': 1e-13, 'L': -1, 'R': 1, 'C': 0.4})
#     params_vld.append({'T': 0.4, 'e': 1e-13, 'L': -1, 'R': 1, 'C': 0.5})
#     params_vld.append({'T': 0.4, 'e': 1e-13, 'L': -1, 'R': 1, 'C': 0.6})
#     params_vld.append({'T': 0.4, 'e': 1e-13, 'L': -1, 'R': 1, 'C': 0.7})
#     params_vld.append({'T': 0.4, 'e': 1e-13, 'L': -1, 'R': 1, 'C': 0.8})
#     params_vld.append({'T': 0.4, 'e': 1e-13, 'L': -1, 'R': 1, 'C': 0.9})
#     return params_vld[j]
#
# params = None
# # params = {'T': 0.4, 'e': 1e-13, 'L': -1, 'R': 1, 'C': 0.7}
# problem_ex_test = problem(ic_numb=2, space_steps=64 * 2 * 2 * 2 * 2 , time_steps=None, params=params)
# problem_ex_test.initial_condition, _, _, _, _, _ = init_cond_B(3, problem_ex_test.x, 4.5)
# problem_ex_test.initial_condition = torch.Tensor(problem_ex_test.initial_condition)
# u_ex, u_ex128 = train_model.compute_exact(Burgers_equation, problem_ex_test, 64*2, 25*4, just_one_time_step=False, trainable=False)
# # u_ex, u_ex128 = train_model.compute_exact(Buckley_Leverett, problem_ex_test, 64*2, 35*4, just_one_time_step=False, trainable=False)
# # torch.save(u_ex, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex_4")
# # torch.save(u_ex128, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex128_4")
# torch.save(u_ex, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex_13")
# torch.save(u_ex128, "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Basic_test_set/u_ex128_13")

# # #Burgers: 64,25 Buckley: 64,35

# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# fig, ax = plt.subplots()
# ax.plot(x, u_nt, color='blue') #, marker='o')
# ax.plot(x, u_nt_JS, color='green') #, marker='o')
# ax.plot(x, u_t, color='red')
# ax.plot(x_ex, u_exact, color='black')
# #axins = zoomed_inset_axes(ax, 1.5, loc=1)  # zoom = 6
# axins = inset_axes(ax, width=0.75, height=2, loc=1)
# axins.plot(x, u_nt, color='blue')
# axins.plot(x, u_nt_JS, color='green')
# axins.plot(x, u_t, color='red')
# axins.plot(x_ex, u_exact, color='black')
# axins.set_xlim(0.48, 0.55)  # Limit the region for zoom
# axins.set_ylim(0, 0.6)
# plt.xticks(visible=False)  # Not present ticks
# plt.yticks(visible=False)
# axins2 = inset_axes(ax, width=0.75, height=2, loc=2)
# axins2.plot(x, u_nt, color='blue')
# axins2.plot(x, u_nt_JS, color='green')
# axins2.plot(x, u_t, color='red')
# axins2.plot(x_ex, u_exact, color='black')
# axins2.set_xlim(-0.15, 0)  # Limit the region for zoom
# axins2.set_ylim(0.1, 0.4)
# plt.xticks(visible=False)  # Not present ticks
# plt.yticks(visible=False)
# #
# ## draw a bbox of the region of the inset axes in the parent axes and
# ## connecting lines between the bbox and the inset axes area
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
# mark_inset(ax, axins2, loc1=1, loc2=3, fc="none", ec="0.5")
# plt.draw()
# plt.show()
#
