import torch
import numpy as np
import matplotlib.pyplot as plt
from initial_condition_Burgers import init_cond_B
from define_WENO_Network import WENONetwork
from define_problem_transport_eq import transport_equation
from define_problem_Buckley_Leverett import Buckley_Leverett
from define_problem_Burgers_equation import Burgers_equation
import pandas as pd

torch.set_default_dtype(torch.float64)
train_model = WENONetwork()


#problem = transport_equation
# problem = Buckley_Leverett
problem = Burgers_equation

train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Models/Model_47/40.pt")
df=pd.read_csv("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Validation_set/parameters.txt")

rng = 13
err_nt_max_vec = np.zeros(rng)
err_nt_mean_vec = np.zeros(rng)
err_t_max_vec = np.zeros(rng)
err_t_mean_vec = np.zeros(rng)
err_nt_JS_max_vec = np.zeros(rng)
err_nt_JS_mean_vec = np.zeros(rng)
ii=0

# for j in [150,106,113,16,105,140,122,32,101,118,122,107,100]:
for j in [34]:
    print(j)
    sample_id = j
    u_ex = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Validation_set/u_exact128_{}.npy".format(sample_id))
    u_ex_whole = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_1024/Validation_set/u_exact_{}.npy".format(sample_id))
    u_ex = torch.Tensor(u_ex)
    ic_id = float(df[df.sample_id == sample_id]["ic_id"])
    kkk = float(df[df.sample_id == sample_id]["k"])
    problem_main = problem(ic_numb=ic_id, space_steps=64 * 2, time_steps=None, params=None)
    params = problem_main.get_params()
    ts = problem_main.time_steps
    problem_main.initial_condition, _, _, _, _, _ = init_cond_B(ic_id, problem_main.x, kkk)
    problem_main.initial_condition = torch.Tensor(problem_main.initial_condition)
    u_nt, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False)
    for k in range(nn):
        u_nt = train_model.run_weno(problem_main, u_nt, mweno=True, mapped=False, vectorized=True, trainable=False, k=k)
    u_nt=u_nt.detach().numpy()
    u_nt_JS, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False)
    for k in range(nn):
        u_nt_JS = train_model.run_weno(problem_main, u_nt_JS, mweno=False, mapped=False, vectorized=True, trainable=False, k=k)
    u_nt_JS=u_nt_JS.detach().numpy()
    u_t, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False)
    for k in range(nn):
        u_t = train_model.run_weno(problem_main, u_t, mweno=True, mapped=False, vectorized=True, trainable=True, k=k)
    u_t = u_t.detach().numpy()
    _, x, t = problem_main.transformation(u_nt)
    time_steps = t.shape[0]
    h = problem_main.h
    # x_ex = np.linspace(0, 2 - h, 1024)
    x_ex = np.linspace(0,2 - 2/1024, 1024)
    # problem_ex = problem(ic_numb=2, space_steps=60 * 2 * 2 * 2, time_steps=None, params=params)
    # _, u_exact_adjusted = train_model.compute_exact_end(Buckley_Leverett, problem_ex, 60, time_steps, just_one_time_step=False, trainable=False)
    u_exact_adjusted = u_ex[:,-1]
    u_exact = u_ex_whole[:,-1]
    error_nt_max = np.max(np.abs(u_nt-u_exact_adjusted.detach().numpy()))
    error_nt_JS_max = np.max(np.abs(u_nt_JS-u_exact_adjusted.detach().numpy()))
    error_t_max = np.max(np.abs(u_t-u_exact_adjusted.detach().numpy()))
    error_nt_mean = (1/128)*(np.sqrt(np.sum((u_nt-u_exact_adjusted.detach().numpy())**2)))
    error_nt_JS_mean = (1/128)*(np.sqrt(np.sum((u_nt_JS-u_exact_adjusted.detach().numpy())**2)))
    error_t_mean = (1/128)*(np.sqrt(np.sum((u_t-u_exact_adjusted.detach().numpy())**2)))
    err_nt_max_vec[ii] = error_nt_max
    err_nt_JS_max_vec[ii] = error_nt_JS_max
    err_t_max_vec[ii] = error_t_max
    err_nt_mean_vec[ii] = error_nt_mean
    err_nt_JS_mean_vec[ii] = error_nt_JS_mean
    err_t_mean_vec[ii] = error_t_mean
    plt.figure(j+1)
    plt.plot(x, u_nt, color='blue', marker='o')
    plt.plot(x, u_nt_JS, color='green', marker='o')
    plt.plot(x, u_t, marker='o', color='red')
    plt.plot(x_ex, u_exact, color='black')
    ii = ii + 1

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

# # #Burgers: 64,25 Buckley: 64,35

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# # problem_32
# fig, ax = plt.subplots()
# ax.plot(x, u_nt, color='blue') #, marker='o')
# ax.plot(x, u_nt_JS, color='green') #, marker='o')
# ax.plot(x, u_t, color='red')
# ax.plot(x_ex, u_exact, color='black')
# ax.legend(('WENO-JS', 'WENO-Z', 'WENO-DS', 'ref. sol.'), loc=(0.01,0.74))
# ax.set_xlabel('x')
# ax.set_ylabel('u')
# #axins = zoomed_inset_axes(ax, 1.5, loc=1)  # zoom = 6
# axins = inset_axes(ax, width=1, height=1, loc=1)
# axins.plot(x, u_nt, color='blue')
# axins.plot(x, u_nt_JS, color='green')
# axins.plot(x, u_t, color='red')
# axins.plot(x_ex, u_exact, color='black')
# axins.set_xlim(0.95, 1)  # Limit the region for zoom
# axins.set_ylim(1.75, 1.88)
# plt.xticks(visible=False)  # Not present ticks
# plt.yticks(visible=False)
# axins2 = inset_axes(ax, width=1, height=1, loc=3)
# axins2.plot(x, u_nt, color='blue')
# axins2.plot(x, u_nt_JS, color='green')
# axins2.plot(x, u_t, color='red')
# axins2.plot(x_ex, u_exact, color='black')
# axins2.set_xlim(0.98,1.04)  # Limit the region for zoom
# axins2.set_ylim(-1.9,-1.7)
# plt.xticks(visible=False)  # Not present ticks
# plt.yticks(visible=False)
# # draw a bbox of the region of the inset axes in the parent axes and
# # connecting lines between the bbox and the inset axes area
# mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")
# mark_inset(ax, axins2, loc1=1, loc2=4, fc="none", ec="0.5")
# plt.draw()
# plt.show()
# plt.savefig("foo.pdf", bbox_inches='tight')

# # problem_34
# fig, ax = plt.subplots()
# ax.plot(x, u_nt, color='blue') #, marker='o')
# ax.plot(x, u_nt_JS, color='green') #, marker='o')
# ax.plot(x, u_t, color='red')
# ax.plot(x_ex, u_exact, color='black')
# ax.legend(('WENO-JS', 'WENO-Z', 'WENO-DS', 'ref. sol.'), loc=(0.01,0.74))
# ax.set_xlabel('x')
# ax.set_ylabel('u')
# #axins = zoomed_inset_axes(ax, 1.5, loc=1)  # zoom = 6
# axins = inset_axes(ax, width=1, height=1, loc=1)
# axins.plot(x, u_nt, color='blue')
# axins.plot(x, u_nt_JS, color='green')
# axins.plot(x, u_t, color='red')
# axins.plot(x_ex, u_exact, color='black')
# axins.set_xlim(0.95, 1)  # Limit the region for zoom
# axins.set_ylim(1.5, 1.62)
# plt.xticks(visible=False)  # Not present ticks
# plt.yticks(visible=False)
# axins2 = inset_axes(ax, width=1, height=1, loc=3)
# axins2.plot(x, u_nt, color='blue')
# axins2.plot(x, u_nt_JS, color='green')
# axins2.plot(x, u_t, color='red')
# axins2.plot(x_ex, u_exact, color='black')
# axins2.set_xlim(0.98,1.04)  # Limit the region for zoom
# axins2.set_ylim(-1.62,-1.52)
# plt.xticks(visible=False)  # Not present ticks
# plt.yticks(visible=False)
# # draw a bbox of the region of the inset axes in the parent axes and
# # connecting lines between the bbox and the inset axes area
# mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")
# mark_inset(ax, axins2, loc1=1, loc2=4, fc="none", ec="0.5")
# plt.draw()
# plt.show()
# plt.savefig("foo.pdf", bbox_inches='tight')

# # # problem_39
# fig, ax = plt.subplots()
# ax.plot(x, u_nt, color='blue') #, marker='o')
# ax.plot(x, u_nt_JS, color='green') #, marker='o')
# ax.plot(x, u_t, color='red')
# ax.plot(x_ex, u_exact, color='black')
# ax.legend(('WENO-JS', 'WENO-Z', 'WENO-DS', 'ref. sol.'), loc=(0.7,0.1))
# ax.set_xlabel('x')
# ax.set_ylabel('u')
# #axins = zoomed_inset_axes(ax, 1.5, loc=1)  # zoom = 6
# axins = inset_axes(ax, width=1, height=1, loc=9)
# axins.plot(x, u_nt, color='blue')
# axins.plot(x, u_nt_JS, color='green')
# axins.plot(x, u_t, color='red')
# axins.plot(x_ex, u_exact, color='black')
# axins.set_xlim(0.15, 0.3)  # Limit the region for zoom
# axins.set_ylim(1.45, 1.68)
# plt.xticks(visible=False)  # Not present ticks
# plt.yticks(visible=False)
# axins2 = inset_axes(ax, width=1, height=1, loc=10)
# axins2.plot(x, u_nt, color='blue')
# axins2.plot(x, u_nt_JS, color='green')
# axins2.plot(x, u_t, color='red')
# axins2.plot(x_ex, u_exact, color='black')
# axins2.set_xlim(0.24,0.29)  # Limit the region for zoom
# axins2.set_ylim(-0.01,0.06)
# plt.xticks(visible=False)  # Not present ticks
# plt.yticks(visible=False)
# #vdraw a bbox of the region of the inset axes in the parent axes and
# # connecting lines between the bbox and the inset axes area
# mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")
# mark_inset(ax, axins2, loc1=2, loc2=3, fc="none", ec="0.5")
# plt.draw()
# plt.show()
# plt.savefig("foo.pdf", bbox_inches='tight')

# # problem_16
# fig, ax = plt.subplots()
# ax.plot(x, u_nt, color='blue') #, marker='o')
# ax.plot(x, u_nt_JS, color='green') #, marker='o')
# ax.plot(x, u_t, color='red')
# ax.plot(x_ex, u_exact, color='black')
# ax.legend(('WENO-JS', 'WENO-Z', 'WENO-DS', 'ref. sol.'), loc=(0.01,0.74))
# ax.set_xlabel('x')
# ax.set_ylabel('u')
# #axins = zoomed_inset_axes(ax, 1.5, loc=1)  # zoom = 6
# axins = inset_axes(ax, width=1, height=1, loc=1)
# axins.plot(x, u_nt, color='blue')
# axins.plot(x, u_nt_JS, color='green')
# axins.plot(x, u_t, color='red')
# axins.plot(x_ex, u_exact, color='black')
# axins.set_xlim(1.24, 1.31)  # Limit the region for zoom
# axins.set_ylim(0.94,1.01)
# plt.xticks(visible=False)  # Not present ticks
# plt.yticks(visible=False)
# ## draw a bbox of the region of the inset axes in the parent axes and
# ## connecting lines between the bbox and the inset axes area
# mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")
# plt.draw()
# plt.show()
# plt.savefig("foo.pdf", bbox_inches='tight')
#
