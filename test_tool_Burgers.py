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


# u_ex_whole = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Test_set_additional/u_exact_IN_4.npy")
# u_ex = u_ex_whole[0:1024 + 1: 8]

space_steps = 1024 #64*2 #*2 # 1024
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
# error_nt_max = np.max(np.abs(u_nt-u_ex))
# error_nt_JS_max = np.max(np.abs(u_nt_JS-u_ex))
# error_t_max = np.max(np.abs(u_t-u_ex))
# error_nt_mean = np.sqrt(2 / space_steps) * (np.sqrt(np.sum((u_nt - u_ex) ** 2)))
# error_nt_JS_mean = np.sqrt(2 / space_steps) * (np.sqrt(np.sum((u_nt_JS - u_ex) ** 2)))
# error_t_mean = np.sqrt(2 / space_steps) * (np.sqrt(np.sum((u_t - u_ex) ** 2)))
plt.plot(x, u_nt, color='blue', marker='o')
plt.plot(x, u_nt_JS, color='green', marker='o')
plt.plot(x, u_t, marker='o', color='red')
# plt.plot(x, u_ex, color='black')

# np.save("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Test_set_additional/u_exact_IN_2.npy", u_nt)

