import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from define_WENO_Network import WENONetwork
from define_problem_transport_eq import transport_equation
from define_problem_Buckley_Leverett import Buckley_Leverett

train_model = WENONetwork()
train_model = torch.load('model3')

torch.set_default_dtype(torch.float64)

#params=None
params =  {'T': 0.4, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 0.25}
#params = {'T': 0.4, 'e': 1e-13, 'L': 1, 'R': 1, 'C': 5.467189905555848}

#problem = transport_equation
problem = Buckley_Leverett

problem_main = problem(ic_numb=6,space_steps=100, time_steps=None, params = params)
params = problem_main.get_params()
#problem_ex = problem(space_steps=100*2*2, time_steps=40*4*4, params = params)
#problem_ex = problem(space_steps=100*2*2*2*2*2*2*2, time_steps=40*4*4*4*4*4*4*4, params = params)

u, nn = train_model.init_run_weno(problem_main, vectorized=False, just_one_time_step=False)
for k in range(nn):
    uu = train_model.run_weno(problem_main, u, mweno=True,mapped=False,vectorized=False,trainable=True,k=k)
    u[:,k+1]=uu


u=u.detach().numpy()
_,x,t = problem_main.transformation(u)
n=u.shape[1]
plt.plot(x,u[:,0],x,u[:,-1])
# plt.plot(x,uu[:,0],x,uu[:,int(np.ceil(n/5))],x,uu[:,int(np.ceil(3*n/5))],x,u[:,-1])
#params = problem_main.get_params()


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
#
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


