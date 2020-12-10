import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from define_WENO_Network import WENONetwork
from define_problem_transport_eq import transport_equation
from define_problem_Buckley_Leverett import Buckley_Leverett
from define_problem_Burgers_equation import Burgers_equation

train_model = WENONetwork()
train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Models/Model_42/12.pt")
#train_model = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Models/Model_18/46")

torch.set_default_dtype(torch.float64)

params=None
#params =  {'T': 0.4, 'e': 1e-13, 'L': -1, 'R': 1, 'C': 0.25}

#problem = transport_equation
problem = Buckley_Leverett
#problem = Burgers_equation

my_problem = problem(ic_numb=6,space_steps=64*2, time_steps=None, params = params)
params = my_problem.get_params()
#problem_ex = problem(space_steps=100*2*2, time_steps=40*4*4, params = params)

# #WHOLE SOLUTION
# u, nn = train_model.init_run_weno(problem_main, vectorized=False, just_one_time_step=False)
# for k in range(nn):
#     uu = train_model.run_weno(problem_main, u, mweno=True,mapped=False,vectorized=False,trainable=False,k=k)
#     u[:,k+1]=uu
# u=u.detach().numpy()
# _,x,t = problem_main.transformation(u)
# n=u.shape[1]
# plt.plot(x,u[:,0],x,u[:,-1])
# X, Y = np.meshgrid(x, t, indexing="ij")
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, u)


# JUST LAST TIME STEP
u_nt, nn = train_model.init_run_weno(my_problem, vectorized=True, just_one_time_step=False)
for k in range(nn):
    u_nt = train_model.run_weno(my_problem, u_nt, mweno=True, mapped=False, vectorized=True, trainable=False, k=k)
u_nt=u_nt.detach().numpy()
u_t, nn = train_model.init_run_weno(my_problem, vectorized=True, just_one_time_step=False)
for k in range(nn):
    u_t = train_model.run_weno(my_problem, u_t, mweno=True, mapped=False, vectorized=True, trainable=True, k=k)
u_t = u_t.detach().numpy()
_, x, t = my_problem.transformation(u_t)
plt.plot(x, u_nt, color='blue', marker='o')
plt.plot(x, u_t, marker='o', color='green')


# problem_ex = problem(ic_numb=6, space_steps=50*2*2, time_steps=None, params=params)
# _, u_ex = train_model.compute_exact(Buckley_Leverett, problem_ex, 50, 25, just_one_time_step=False, trainable=False)

# plt.figure(2)
# plt.plot(x,u_ex[:,0],x,u_ex[:,-1])

# plt.figure(3)
# plt.plot(x,u[:,-1], x, u_ex[:,-1])

# plt.figure(4)
# X, Y = np.meshgrid(x, t, indexing="ij")
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, uu)

# plt.figure(5)
# X, Y = np.meshgrid(x, t, indexing="ij")
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, uex)

#u_exact, u_exact_adjusted = train_model.compute_exact(Buckley_Leverett, problem_ex, 100, 40, just_one_time_step = True, trainable= False)

#error = train_model.compute_error(problem, u, u_exact_adjusted, trainable=False)

#plt.plot(x, uu[:, -1], x, u_exact_adjusted[:,-1])

#plt.plot(x, uu[:, 1], x, u_exact_adjusted[:,1])

# for k in range(0,len(u_exact_adjusted[0])):
#     plt.plot(x,u_exact_adjusted[:,k])
#
# for k in range(0,len(u_exact[0])):
#     plt.plot(u_exact[:,k])

#u_exact = problem_main.exact()
# plt.plot(x,uu[:,-1],x,u_exact)
#
# u_last = u[:,-1]
# error = my_problem.err(u_last)
#
# u_whole_exact=my_problem.whole_exact()
# error_whole= uu-u_whole_exact


