import numpy as np
from scipy import signal

def init_cond(ic_numb, x):
    #Lx = x[-1] - x[0]
    xmid = 0
    numb = 1
    height = 1
    width = 0.3

    if ic_numb==1: # Gaussian wave for
        u0 = np.exp(-20 * (x - xmid)**2)
    elif ic_numb==2: # Sinusoidal wave
        u0 = np.sin(np.pi*x)
    elif ic_numb==3: # Shifted sinusoidal wave
        u0 = 0.5 - np.sin(np.pi*x)
    elif ic_numb==4: # Riemann problem
        u0 = np.zeros(x.shape[0])
        u0[x <= xmid] = 1
    elif ic_numb==5: # Signal square
        u0 = signal.square(2 * np.pi * x)
    elif ic_numb==6: # Square Jump
        u0 = np.zeros(x.shape[0])
        for k in range(0, x.shape[0]):
            if x[k] >= -0.5 and x[k] <= xmid:
                u0[k] = 1
    elif ic_numb==7: # Oleg's trapezoidal left
        u0 = np.zeros(x.shape[0])
        for k in range(0, x.shape[0] ):
            if x[k] >= xmid-0.3 and x[k] <= xmid + 0.3:
                u0[k] = np.exp(x[k])
    elif ic_numb==8: # Oleg's trapezoidal right
        u0 = np.zeros(x.shape[0])
        for k in range(0, x.shape[0] ):
            if x[k] >= xmid-0.3 and x[k] <= xmid + 0.3:
                u0[k] = np.exp(-x[k])

    return u0, numb, xmid, height, width
