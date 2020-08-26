import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from define_WENO_Network import WENONetwork
from define_problem_transport_eq import transport_equation

train_model = WENONetwork()
train_model = torch.load('model')

torch.set_default_dtype(torch.float64)

params=None
#params =  {'T': 0.4, 'e': 1e-13, 'L': 1, 'R': 1, 'C': 1.0060472785249628}
#params = {'T': 0.4, 'e': 1e-13, 'L': 1, 'R': 1, 'C': 5.467189905555848}
#params = {'sigma': 0.3, 'rate': 0.02, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5, 'psi':20}
#params = {'sigma': 0.3, 'rate': 0.25, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -1.5, 'xr': 2, 'psi':30}
#params = {'sigma': 0.3, 'rate': 0.1, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5}

#problem = PME
#problem = heat_equation
problem = transport_equation
#problem= Digital_option
#problem= Call_option
#problem = Call_option_GS
#problem = Digital_option_GS
#problem = Buckley_Leverett

problem_main = problem(space_steps=50, time_steps=None, params = params)
params = problem_main.get_params()
#problem_ex = problem(space_steps=100*2*2, time_steps=40*4*4, params = params)
#problem_ex = problem(space_steps=100*2*2*2*2*2*2*2, time_steps=40*4*4*4*4*4*4*4, params = params)
u = train_model.run_weno(problem_main, vectorized=False, trainable = False, just_one_time_step = False)
uu=u.detach().numpy()
_,x,t = problem_main.transformation(u)
#plt.plot(x, uu[:, -1])
n=uu.shape[1]
plt.plot(x,uu[:,0],x,uu[:,-1])
# plt.plot(x,uu[:,0],x,uu[:,int(np.ceil(n/5))],x,uu[:,int(np.ceil(3*n/5))],x,u[:,-1])
#params = problem_main.get_params()

plt.plot(x,uu[:,0],x,uu[:,int(np.ceil(n/5))],x,uu[:,int(np.ceil(2*n/5))],x,uu[:,int(np.ceil(3*n/5))],x,uu[:,-1])

plt.figure(2)
X, Y = np.meshgrid(x, t, indexing="ij")
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, uu)

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


# h = (np.abs(-4) + np.abs(6)) / 100
# xx = np.linspace(-4, 6-h, 100)
# xmid = 0.5 * (xx[-1]  + xx[0])
# u0 = np.zeros((xx.shape[0]))
# for n in range(-2,3):
#     plus = np.exp(-20 * (xx - xmid - n * 2) ** 2)
#     u0 = u0 + plus
# plt.plot(xx,u0)
#

h = (np.abs(-10) + np.abs(2)) / 1200
xx = np.linspace(-10, 2-h, 1200)
time = np.linspace(0, 10, 2000)
xmid = 1 # 0.5 * (x[-1] + x[0])
uex = np.zeros((xx.shape[0],time.shape[0]))
plus = np.zeros((xx.shape[0],time.shape[0]))

narr = np.array([-5,-4,-3,-2,-1,0])
for k in range(0, time.shape[0]):
    for j in range(0, xx.shape[0]):
        if (xx[j] >=  time[k] + (1+narr[0]*2) and xx[j] < time[k] + (2+narr[0]*2)) or \
                (xx[j] >=  time[k] + (1+narr[1]*2) and xx[j] < time[k] + (2+narr[1]*2)) or \
                (xx[j] >= time[k] + (1 + narr[2] * 2) and xx[j] < time[k] + (2 + narr[2] * 2)) or \
                (xx[j] >= time[k] + (1 + narr[3] * 2) and xx[j] < time[k] + (2 + narr[3] * 2)) or \
                (xx[j] >= time[k] + (1 + narr[4] * 2) and xx[j] < time[k] + (2 + narr[4] * 2)) or \
                (xx[j] >= time[k] + (1 + narr[5] * 2) and xx[j] < time[k] + (2 + narr[5] * 2)) :
                uex[j, k] = 1

for n in range(-5,3):
    for k in range(0, time.shape[0]):
        for j in range(0, xx.shape[0]):
            plus[j,k] = np.exp(-20 * (xx[j] - time[k] - xmid - n * 2) ** 2)
    uex = uex + plus

plt.figure(3)
X, Y = np.meshgrid(xx[1000:], time, indexing="ij")
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, uex[1000:,:])

# plt.plot(xx[1000:],uex[1000:,0],xx[1000:],uex[1000:,int(np.ceil(n/5))],xx[1000:],uex[1000:,int(np.ceil(2*n/5))],xx[1000:],uex[1000:,int(np.ceil(3*n/5))],xx[1000:],uex[1000:,-1])

plt.plot(xx[1000:],uex[1000:,0],xx[1000:],uex[1000:,400],xx[1000:],uex[1000:,800],xx[1000:],uex[1000:,1200],xx[1000:],uex[1000:,1600],xx[1000:],uex[1000:,-1])
plt.plot(xx[1000:],uex[1000:,0],xx[1000:],uex[1000:,-1])

X, Y = np.meshgrid(xx, time, indexing="ij")
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, uex)