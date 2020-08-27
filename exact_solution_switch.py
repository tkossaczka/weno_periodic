import numpy as np
from scipy import signal
from transport_exact_solution import transport_exact_solution

# for du/dt = -du/dx

class init_cond_switch():
    def __init__(self, x):
        self.Lx = x[-1] - x[0]
        self.xmid = 1

    def case_1(self, x, t): # Gaussian wave
        u0 = np.exp(-20 * (x - self.xmid)**2)
        uex = np.zeros((x.shape[0], t.shape[0]))
        for i in range(0, t.shape[0]):
            uex[:, i] = transport_exact_solution(initial_cond=u0, time=t[i], dx=x[1] - x[0], transport_coef=1)
        return uex
        return  uex

    def case_2(self,x, t): # Sinusoidal wave
        uex = np.zeros((x.shape[0],t.shape[0]))
        for k in range(0, t.shape[0]):
            for j in range(0, x.shape[0]):
                uex[j, k] = np.sin(np.pi*(x[j]-t[k]))
        return uex

    def case_3(self,x,t): # Shifted sinusoidal wave
        uex = np.zeros((x.shape[0], t.shape[0]))
        for k in range(0, t.shape[0]):
            for j in range(0, x.shape[0]):
                uex[j, k] = 0.5 - np.sin(np.pi * (x[j] - t[k]))
        return uex

    # def case_4(self,x): # Hyperbolic Tangent
    #     mu = 0.02
    #     return  0.5 * (1 - np.tanh(x / (4 * mu)))

    def case_4(self,x, t): # Riemann problem
        u0 = np.zeros(x.shape[0])
        u0[x >= self.xmid] = 1
        uex = np.zeros((x.shape[0], t.shape[0]))
        for i in range(0, t.shape[0]):
            uex[:, i] = transport_exact_solution(initial_cond=u0, time=t[i], dx=x[1] - x[0], transport_coef=1)
        return uex
        return uex

    def case_5(self,x, t): # Signal square
        uex = np.zeros((x.shape[0], t.shape[0]))
        for k in range(0, t.shape[0]):
            for j in range(0, x.shape[0]):
                uex[j, k] = signal.square(2 * np.pi * (x[j] - t[k]))
        return uex

    def case_6(self,x, t): # Square Jump
        u0 = np.zeros(x.shape[0])
        for k in range(0, x.shape[0]):
            if x[k] >= self.xmid - 0.3 and x[k] <= self.xmid + 0.3:
                u0[k] = 1
        uex = np.zeros((x.shape[0],t.shape[0]))
        for i in range(0,t.shape[0]):
            uex[:,i] = transport_exact_solution(initial_cond=u0, time=t[i], dx=x[1]-x[0], transport_coef=1)
        return uex

    def case_7(self,x,t): # Oleg's trapezoidal left
        u0 = np.zeros(x.shape[0])
        for k in range(0, x.shape[0]):
            if x[k] >= self.xmid - 0.3 and x[k] <= self.xmid + 0.3:
                u0[k] = np.exp(x[k])
        uex = np.zeros((x.shape[0], t.shape[0]))
        for i in range(0, t.shape[0]):
            uex[:, i] = transport_exact_solution(initial_cond=u0, time=t[i], dx=x[1] - x[0], transport_coef=1)
        return uex

    def case_8(self,x,t): # Oleg's trapezoidal right
        u0 = np.zeros(x.shape[0])
        for k in range(0, x.shape[0]):
            if x[k] >= self.xmid - 0.3 and x[k] <= self.xmid + 0.3:
                u0[k] = np.exp(-x[k])
        uex = np.zeros((x.shape[0], t.shape[0]))
        for i in range(0, t.shape[0]):
            uex[:, i] = transport_exact_solution(initial_cond=u0, time=t[i], dx=x[1] - x[0], transport_coef=1)
        return uex