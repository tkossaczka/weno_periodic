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

train_model = torch.load('C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Models/Model_46/28.pt')  # 30/10 good
params = {'T': 0.4, 'e': 1e-13, 'L': -1, 'R': 1, 'C': 0.25}

my_problem = problem(ic_numb=6, space_steps=64 * 2*2, time_steps=None, params=params)
C = params['C']
u_ex_0_w = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1024/Basic_test_set/u_ex_0")
u_ex = u_ex_0_w[0:1024 + 1:4, 0:8960 + 1:16]

u_nt, nn = train_model.init_run_weno(my_problem, vectorized=True, just_one_time_step=False)
start_time = time.time()
for k in range(nn):
    u_nt = train_model.run_weno(my_problem, u_nt, mweno=True, mapped=False, vectorized=True, trainable=False, k=k)
print("--- %s seconds ---" % (time.time() - start_time))
u_nt = u_nt.detach().numpy()
error_nt_mean = np.mean((u_nt - u_ex.detach().numpy()[:,-1]) ** 2)

# u_t, nn = train_model.init_run_weno(my_problem, vectorized=True, just_one_time_step=False)
# start_time = time.time()
# for k in range(nn):
#     u_t = train_model.run_weno(my_problem, u_t, mweno=True, mapped=False, vectorized=True, trainable=True, k=k)
# print("--- %s seconds ---" % (time.time() - start_time))
# u_t = u_t.detach().numpy()
