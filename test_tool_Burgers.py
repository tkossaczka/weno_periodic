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

problem = Burgers_equation

train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Models/Model_78/29.pt")


u_ex_whole = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Test_set_additional/u_exact_IN_15.npy")
u_ex = u_ex_whole[0:1024 + 1: 8]

space_steps = 128 #64*2 #*2 # 1024
problem_main = problem(ic_numb=2, space_steps=space_steps, time_steps=None, params=None)
params = problem_main.get_params()
ts = problem_main.time_steps
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
error_nt_max = np.max(np.abs(u_nt-u_ex))
error_nt_JS_max = np.max(np.abs(u_nt_JS-u_ex))
error_t_max = np.max(np.abs(u_t-u_ex))
error_nt_mean = np.sqrt(2 / space_steps) * (np.sqrt(np.sum((u_nt - u_ex) ** 2)))
error_nt_JS_mean = np.sqrt(2 / space_steps) * (np.sqrt(np.sum((u_nt_JS - u_ex) ** 2)))
error_t_mean = np.sqrt(2 / space_steps) * (np.sqrt(np.sum((u_t - u_ex) ** 2)))
plt.plot(x, u_nt, color='blue', marker='o')
plt.plot(x, u_nt_JS, color='green', marker='o')
plt.plot(x, u_t, marker='o', color='red')
#plt.plot(x, u_ex, color='black')

# np.save("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Test_set_additional/u_exact_IN_17.npy", u_nt)


# # IN_15
# fig, ax = plt.subplots()
# ax.plot(x, u_nt_JS, color='blue') #, marker='o')
# ax.plot(x, u_nt, color='green') #, marker='o')
# ax.plot(x, u_t, color='red', marker='x')
# ax.plot(x, u_ex, color='black')
# ax.legend(('WENO-JS', 'WENO-Z', 'WENO-DS', 'ref. sol.'), loc=1)
# ax.set_xlabel('x')
# ax.set_ylabel('u')
# plt.savefig("BE_IN_15.pdf", bbox_inches='tight')

# # IN_14
# h = problem_main.h
# x_ex = np.linspace(-1, 1 - h, 1024)
# fig, ax = plt.subplots()
# ax.plot(x, u_nt_JS, color='blue') #, marker='o')
# ax.plot(x, u_nt, color='green') #, marker='o')
# ax.plot(x, u_t, color='red', marker='x')
# ax.plot(x, u_ex, color='black')
# ax.legend(('WENO-JS', 'WENO-Z', 'WENO-DS', 'ref. sol.'), loc=4)
# ax.set_xlabel('x')
# ax.set_ylabel('u')
# plt.savefig("BE_IN_14.pdf", bbox_inches='tight')

# # IN_13
# fig, ax = plt.subplots()
# ax.plot(x, u_nt_JS, color='blue') #, marker='o')
# ax.plot(x, u_nt, color='green') #, marker='o')
# ax.plot(x, u_t, color='red', marker='x')
# ax.plot(x, u_ex, color='black')
# ax.legend(('WENO-JS', 'WENO-Z', 'WENO-DS', 'ref. sol.'), loc=2)
# ax.set_xlabel('x')
# ax.set_ylabel('u')
# plt.savefig("BE_IN_13.pdf", bbox_inches='tight')

# # IN_12
# h = problem_main.h
# x_ex = np.linspace(0, 2 - h, 1024)
# fig, ax = plt.subplots()
# ax.plot(x, u_nt_JS, color='blue') #, marker='o')
# ax.plot(x, u_nt, color='green') #, marker='o')
# ax.plot(x, u_t, color='red', marker='x')
# ax.plot(x_ex, u_ex_whole, color='black')
# ax.legend(('WENO-JS', 'WENO-Z', 'WENO-DS', 'ref. sol.'), loc=4)
# ax.set_xlabel('x')
# ax.set_ylabel('u')
# plt.savefig("BE_IN_12.pdf", bbox_inches='tight')

# # IN_2
# fig, ax = plt.subplots()
# ax.plot(x, u_nt_JS, color='blue') #, marker='o')
# ax.plot(x, u_nt, color='green') #, marker='o')
# ax.plot(x, u_t, color='red', marker='x')
# ax.plot(x, u_ex, color='black')
# ax.legend(('WENO-JS', 'WENO-Z', 'WENO-DS', 'ref. sol.'), loc=1)
# ax.set_xlabel('x')
# ax.set_ylabel('u')
# plt.savefig("BE_IN_2.pdf", bbox_inches='tight')

# # IN_3
# fig, ax = plt.subplots()
# ax.plot(x, u_nt_JS, color='blue') #, marker='o')
# ax.plot(x, u_nt, color='green') #, marker='o')
# ax.plot(x, u_t, color='red', marker='x')
# ax.plot(x, u_ex, color='black')
# ax.legend(('WENO-JS', 'WENO-Z', 'WENO-DS', 'ref. sol.'), loc=4)
# ax.set_xlabel('x')
# ax.set_ylabel('u')
# plt.savefig("BE_IN_3.pdf", bbox_inches='tight')

# # IN_10
# fig, ax = plt.subplots()
# ax.plot(x, u_nt_JS, color='blue') #, marker='o')
# ax.plot(x, u_nt, color='green') #, marker='o')
# ax.plot(x, u_t, color='red', marker='x')
# ax.plot(x, u_ex, color='black')
# ax.legend(('WENO-JS', 'WENO-Z', 'WENO-DS', 'ref. sol.'), loc=4)
# ax.set_xlabel('x')
# ax.set_ylabel('u')
# plt.savefig("BE_IN_10.pdf", bbox_inches='tight')

# # IN_4
# fig, ax = plt.subplots()
# ax.plot(x, u_nt_JS, color='blue') #, marker='o')
# ax.plot(x, u_nt, color='green') #, marker='o')
# ax.plot(x, u_t, color='red', marker='x')
# ax.plot(x, u_ex, color='black')
# ax.legend(('WENO-JS', 'WENO-Z', 'WENO-DS', 'ref. sol.'), loc=4)
# ax.set_xlabel('x')
# ax.set_ylabel('u')
# plt.savefig("BE_IN_4.pdf", bbox_inches='tight')

# # IN_7
# fig, ax = plt.subplots()
# ax.plot(x, u_nt_JS, color='blue') #, marker='o')
# ax.plot(x, u_nt, color='green') #, marker='o')
# ax.plot(x, u_t, color='red', marker='x')
# ax.plot(x, u_ex, color='black')
# ax.legend(('WENO-JS', 'WENO-Z', 'WENO-DS', 'ref. sol.'), loc=4)
# ax.set_xlabel('x')
# ax.set_ylabel('u')
# plt.savefig("BE_IN_7.pdf", bbox_inches='tight')