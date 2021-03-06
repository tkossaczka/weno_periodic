from define_WENO_Network import WENONetwork
import torch
import numpy as np
import matplotlib.pyplot as plt
# from define_WENO_Network import WENONetwork
from define_problem_transport_eq import transport_equation
from define_problem_Buckley_Leverett import Buckley_Leverett
from initial_jump_generator import init_jump

torch.set_default_dtype(torch.float64)

train_model = WENONetwork()

params = None
#params = {'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 0.42693827394864636} # model18!!!! so good
#params = {'T': 1, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 3.6184183991789673}
#params = {'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 1.3235713669969233}
#params = {'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 0.05981614981451925}
#params = {'T': 0.4, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 0.25}
#params = {'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 0.44885801631216066}
#params = {'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 1.630911434140728}
#params = {'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 0.4336634929945137}
#params = {'T': 0.4, 'e': 1e-13, 'L': 1, 'R': 1, 'C': 0.2}
#params = {'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 0.8092717532100926}   {'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 2.0868858027855617}  #100 space steps, good

#problem = transport_equation
problem = Buckley_Leverett

params = {'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 0.5}
my_problem = problem(ic_numb=0,space_steps=120, time_steps=None, params = params)
params = my_problem.params
my_problem.initial_condition, _, _, _, _ = init_jump(my_problem.x, numb=1, xmid=1, height=0.3, width=0.1)
my_problem.initial_condition = torch.Tensor(my_problem.initial_condition)
#V_t, S_t, tt_t = train_model.full_WENO(my_problem, trainable=True, plot=False, vectorized=False)
V_nt, S_nt, tt_nt = train_model.full_WENO(my_problem, trainable=False, plot=True, vectorized=False)
time_steps = tt_nt.shape[0]

# problem_ex = problem(ic_numb=6, space_steps=60*2, time_steps=None, params = params)
# _, u_exact_adjusted = train_model.compute_exact_end(Buckley_Leverett, problem_ex, 60, time_steps, just_one_time_step = False, trainable= False)
# error_nt = np.max(np.abs(V_nt[:, -1]-u_exact_adjusted.detach().numpy()))
# error_t = np.max(np.abs(V_t[:, -1]-u_exact_adjusted.detach().numpy()))
# error_nt_mean = np.mean((V_nt[:, -1]-u_exact_adjusted.detach().numpy())**2)
# error_t_mean = np.mean((V_t[:, -1]-u_exact_adjusted.detach().numpy())**2)
# plt.plot(S_nt, V_nt[:, -1], color='blue', marker='o')
# plt.plot(S_t, V_t[:,-1], marker='o', color='green')
# plt.plot(S_nt, u_exact_adjusted, color='red')


#u_exact = my_problem.exact()
# plt.figure(3)
# plt.plot(S_t,V_t[:,-1],S_t,u_exact[:,-1])
#V_t_last = V_t[:,-1]
#error_t = np.mean((u_exact[:,-1] - V_t_last)**2)
#error_t = my_problem.err(V_t_last,-1)

# plt.figure(4)
# plt.plot(S_nt,V_nt[:,-1],S_nt,u_exact[:,-1])
#V_nt_last = V_nt[:,-1]
#error_nt = np.mean((u_exact[:,-1] - V_nt_last)**2)
# error_nt = my_problem.err(V_nt_last,-1)

# plt.figure(2)
# for k in range(0,len(V_nt[0])):
#     plt.plot(S_nt,V_nt[:,k])
#
# plt.figure(3)
# for k in range(0,len(V_t[0])):
#     plt.plot(S_t,V_t[:,k])

# plt.plot(S_nt,V_nt[:,-1])
# plt.plot(S_t,V_t[:,-1])
