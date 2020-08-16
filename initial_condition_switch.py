import numpy as np
from scipy import signal

class init_cond_switch():
    def __init__(self, x):
        self.Lx = x[-1] - x[0]
        self.xmid = 0.5 * (x[-1]  + x[0])

    def case_1(self, x): # Gaussian wave for
        return  np.exp(-20 * (x - self.xmid)**2)

    def case_2(self,x): # Sinusoidal wave
        return np.sin(np.pi*x)

    def case_3(self,x): # Shifted sinusoidal wave
        return 0.5 - np.sin(np.pi*x)

    def case_4(self,x): # Hyperbolic Tangent
        mu = 0.02
        return  0.5 * (1 - np.tanh(x / (4 * mu)))

    def case_5(self,x): # Riemann problem
        u0 = np.zeros(x.shape[0])
        u0[x >= self.xmid] = 1
        return u0

    def case_6(self,x): # Signal square
        return signal.square(2 * np.pi * x)

    def case_7(self,x): # Square Jump
        u0 = np.zeros(x.shape[0])
        for k in range(0, x.shape[0]):
            if x[k] >= self.xmid - 0.3 and x[k] <= self.xmid + 0.3:
                u0[k] = 1
            else:
                u0[k] = 0
        return u0

    def case_8(self,x): # Oleg's trapezoidal left
        u0 = np.zeros(x.shape[0])
        for k in range(0, x.shape[0] ):
            if x[k] >= self.xmid-0.3 and x[k] <= self.xmid + 0.3:
                u0[k] = np.exp(x[k])
            else:
                u0[k] = 0
        return u0

    def case_9(self,x): # Oleg's trapezoidal right
        u0 = np.zeros(x.shape[0])
        for k in range(0, x.shape[0] ):
            if x[k] >= self.xmid-0.3 and x[k] <= self.xmid + 0.3:
                u0[k] = np.exp(-x[k])
            else:
                u0[k] = 0
        return u0