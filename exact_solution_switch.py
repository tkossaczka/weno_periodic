import numpy as np
from scipy import signal

# for du/dt = -du/dx

class init_cond_switch():
    def __init__(self, x):
        self.Lx = x[-1] - x[0]
        self.xmid = 0.5 * (x[-1]  + x[0])

    def case_1(self, x, t): # Gaussian wave
        uex = np.zeros((x.shape[0],t.shape[0]))
        for k in range(0, t.shape[0]):
            for j in range(0, x.shape[0]):
                uex[j,k] =  np.exp(-20 * (x[j] - t[k] - self.xmid)**2)
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
        uex = np.zeros((x.shape[0], t.shape[0]))
        for k in range(0, t.shape[0]):
            for j in range(0, x.shape[0]):
                if (x[j] >= self.xmid + t[k]):
                    uex[j, k] = 1
        return uex

    def case_5(self,x, t): # Signal square
        uex = np.zeros((x.shape[0], t.shape[0]))
        for k in range(0, t.shape[0]):
            for j in range(0, x.shape[0]):
                uex[j, k] = signal.square(2 * np.pi * (x[j] - t[k]))
        return uex

    def case_6(self,x, t): # Square Jump
        uex = np.zeros((x.shape[0], t.shape[0]))
        for k in range(0, t.shape[0]):
            for j in range(0, x.shape[0]):
                if x[j] >= self.xmid - 0.3 + t[k] and x[j] <= self.xmid + 0.3 + t[k] :
                    uex[j,k] = 1
        return uex

    def case_7(self,x,t): # Oleg's trapezoidal left
        uex = np.zeros((x.shape[0], t.shape[0]))
        for k in range(0, t.shape[0]):
            for j in range(0, x.shape[0]):
                if x[j] >= self.xmid - 0.3 + t[k] and x[j] <= self.xmid + 0.3 + t[k]:
                    uex[j,k] = np.exp(x[j]- t[k])
        return uex

    def case_8(self,x,t): # Oleg's trapezoidal right
        uex = np.zeros((x.shape[0], t.shape[0]))
        for k in range(0, t.shape[0] ):
            for j in range(0, x.shape[0]):
                if x[j] >= self.xmid - 0.3 + t[k] and x[j] <= self.xmid + 0.3 + t[k]:
                    uex[j,k] = np.exp(-x[j])
        return uex