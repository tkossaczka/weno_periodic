from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import torch
import numpy as np
import matplotlib.pyplot as plt
from initial_condition_Burgers import init_cond_B
from define_WENO_Network import WENONetwork
from define_problem_transport_eq import transport_equation
from define_problem_Buckley_Leverett import Buckley_Leverett
from define_problem_Burgers_equation import Burgers_equation

torch.set_default_dtype(torch.float64)
#train_model = WENONetwork()

#problem = transport_equation
problem = Buckley_Leverett
#problem = Burgers_equation

if problem == Buckley_Leverett:
    train_model = torch.load('C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Models/Model_30/19.pt')
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
    u_ex_whole_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex_0")
    u_ex_whole_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex_1")
    u_ex_whole_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex_2")
    u_ex_whole_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex_3")
    u_ex_whole_4 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex_4")
    u_ex_whole_5 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex_5")
    u_ex_whole_6 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_2/Basic_test_set/u_ex_6")
    u_exs_whole = [u_ex_whole_0, u_ex_whole_1, u_ex_whole_2, u_ex_whole_3, u_ex_whole_4, u_ex_whole_5, u_ex_whole_6]
    divider_space = 4
    divider_time = 16
    u_exs = [u_ex_whole_0[0:512 + 1:divider_space, 0:2240 + 1:divider_time], u_ex_whole_1[0:512 + 1:divider_space, 0:2240 + 1:divider_time],
             u_ex_whole_2[0:512 + 1:divider_space, 0:2240 + 1:divider_time], u_ex_whole_3[0:512 + 1:divider_space, 0:2240 + 1:divider_time],
             u_ex_whole_4[0:512 + 1:divider_space, 0:2240 + 1:divider_time], u_ex_whole_5[0:512 + 1:divider_space, 0:2240 + 1:divider_time],
             u_ex_whole_6[0:512 + 1:divider_space, 0:2240 + 1:divider_time]]

err_nt_max_vec = np.zeros(rng)
err_nt_mean_vec = np.zeros(rng)
err_t_max_vec = np.zeros(rng)
err_t_mean_vec = np.zeros(rng)
err_nt_JS_max_vec = np.zeros(rng)
err_nt_JS_mean_vec = np.zeros(rng)

for j in range(2,3):
    print(j)
    if problem == Buckley_Leverett:
        params = validation_problems(j)
        my_problem = problem(ic_numb=6, space_steps=64*2, time_steps=None, params=params)
    elif problem == Burgers_equation:
        params_vld = validation_problems(j)
        ic_id_test = params_vld['ic_id']
        kkkk = params_vld['k']
        params = None
        my_problem = problem(ic_numb=1, space_steps=64, time_steps=None, params=params)
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
    x_ex = np.linspace(-1, 1 - h, 512)
    # problem_ex = problem(ic_numb=2, space_steps=60 * 2 * 2 * 2, time_steps=None, params=params)
    # _, u_exact_adjusted = train_model.compute_exact_end(Buckley_Leverett, problem_ex, 60, time_steps, just_one_time_step=False, trainable=False)
    u_exact_adjusted = u_exs[j][:,-1]
    u_exact = u_exs_whole[j][:,-1]
    error_nt_max = np.max(np.abs(u_nt-u_exact_adjusted.detach().numpy()))
    error_nt_JS_max = np.max(np.abs(u_nt_JS-u_exact_adjusted.detach().numpy()))
    error_t_max = np.max(np.abs(u_t-u_exact_adjusted.detach().numpy()))
    error_nt_mean = np.mean((u_nt-u_exact_adjusted.detach().numpy())**2)
    error_nt_JS_mean = np.mean((u_nt_JS-u_exact_adjusted.detach().numpy())**2)
    error_t_mean = np.mean((u_t-u_exact_adjusted.detach().numpy())**2)
    err_nt_max_vec[j] = error_nt_max
    err_nt_JS_max_vec[j] = error_nt_JS_max
    err_t_max_vec[j] = error_t_max
    err_nt_mean_vec[j] = error_nt_mean
    err_nt_JS_mean_vec[j] = error_nt_JS_mean
    err_t_mean_vec[j] = error_t_mean
    # plt.figure(j+1)
    # plt.plot(x, u_nt, color='blue', marker='o')
    # plt.plot(x, u_nt_JS, color='green', marker='o')
    # plt.plot(x, u_t, marker='o', color='red')
    # plt.plot(x_ex, u_exact, color='black')

fig, ax = plt.subplots()
ax.plot(x, u_nt_JS, color='blue') #, marker='o')
ax.plot(x, u_nt, color='green') #, marker='o')
ax.plot(x, u_t, color='red')
ax.plot(x_ex, u_exact, color='black')
ax.legend(('WENO-JS', 'WENO-Z', 'WENO-ML', 'ref. sol.'), loc=(0.55, 0.75))
ax.set_xlabel('x')
ax.set_ylabel('u')
#axins = zoomed_inset_axes(ax, 1.5, loc=1)  # zoom = 6
axins = inset_axes(ax, width=1, height=2, loc=1)
axins.plot(x, u_nt_JS, color='blue')
axins.plot(x, u_nt, color='green')
axins.plot(x, u_t, color='red')
axins.plot(x_ex, u_exact, color='black')
axins.set_xlim(0.5, 0.6)  # Limit the region for zoom
axins.set_ylim(0, 0.55)
plt.xticks(visible=False)  # Not present ticks
plt.yticks(visible=False)
axins2 = inset_axes(ax, width=1, height=2, loc=2)
axins2.plot(x, u_nt, color='blue')
axins2.plot(x, u_nt_JS, color='green')
axins2.plot(x, u_t, color='red')
axins2.plot(x_ex, u_exact, color='black')
axins2.set_xlim(-0.2, -0.05)  # Limit the region for zoom
axins2.set_ylim(0.12, 0.3)
plt.xticks(visible=False)  # Not present ticks
plt.yticks(visible=False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
mark_inset(ax, axins2, loc1=1, loc2=3, fc="none", ec="0.5")
plt.draw()
plt.show()
plt.savefig("foo.pdf", bbox_inches='tight')