import numpy as np
from scipy import signal
from transport_exact_solution import transport_exact_solution

# for du/dt = -du/dx

def exact_sol(numb,x,t):
    #Lx = x[-1] - x[0]
    xmid = 1

    if numb==1: # Gaussian wave
        u0 = np.exp(-20 * (x - xmid)**2)
        uex = np.zeros((x.shape[0], t.shape[0]))
        for i in range(0, t.shape[0]):
            uex[:, i] = transport_exact_solution(initial_cond=u0, time=t[i], dx=x[1] - x[0], transport_coef=1)
    elif numb==2: # Sinusoidal wave
        uex = np.zeros((x.shape[0],t.shape[0]))
        for k in range(0, t.shape[0]):
            for j in range(0, x.shape[0]):
                uex[j, k] = np.sin(np.pi*(x[j]-t[k]))
    elif numb==3: # Shifted sinusoidal wave
        uex = np.zeros((x.shape[0], t.shape[0]))
        for k in range(0, t.shape[0]):
            for j in range(0, x.shape[0]):
                uex[j, k] = 0.5 - np.sin(np.pi * (x[j] - t[k]))
    elif numb==4: # Riemann problem
        u0 = np.zeros(x.shape[0])
        u0[x >= xmid] = 1
        uex = np.zeros((x.shape[0], t.shape[0]))
        for i in range(0, t.shape[0]):
            uex[:, i] = transport_exact_solution(initial_cond=u0, time=t[i], dx=x[1] - x[0], transport_coef=1)
    elif numb==5: # Signal square
        uex = np.zeros((x.shape[0], t.shape[0]))
        for k in range(0, t.shape[0]):
            for j in range(0, x.shape[0]):
                uex[j, k] = signal.square(2 * np.pi * (x[j] - t[k]))
    elif numb==6: # Square Jump
        u0 = np.zeros(x.shape[0])
        for k in range(0, x.shape[0]):
            if x[k] >= xmid - 0.3 and x[k] <= xmid + 0.3:
                u0[k] = 1
        uex = np.zeros((x.shape[0],t.shape[0]))
        for i in range(0,t.shape[0]):
            uex[:,i] = transport_exact_solution(initial_cond=u0, time=t[i], dx=x[1]-x[0], transport_coef=1)
    elif numb==7: # Oleg's trapezoidal left
        u0 = np.zeros(x.shape[0])
        for k in range(0, x.shape[0]):
            if x[k] >= xmid - 0.3 and x[k] <= xmid + 0.3:
                u0[k] = np.exp(x[k])
        uex = np.zeros((x.shape[0], t.shape[0]))
        for i in range(0, t.shape[0]):
            uex[:, i] = transport_exact_solution(initial_cond=u0, time=t[i], dx=x[1] - x[0], transport_coef=1)
    elif numb==8: # Oleg's trapezoidal right
        u0 = np.zeros(x.shape[0])
        for k in range(0, x.shape[0]):
            if x[k] >= xmid - 0.3 and x[k] <= xmid + 0.3:
                u0[k] = np.exp(-x[k])
        uex = np.zeros((x.shape[0], t.shape[0]))
        for i in range(0, t.shape[0]):
            uex[:, i] = transport_exact_solution(initial_cond=u0, time=t[i], dx=x[1] - x[0], transport_coef=1)

    return uex