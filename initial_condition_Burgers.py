import numpy as np
import random

def init_cond_B(ic_numb, x, k = None):
    #Lx = x[-1] - x[0]
    xmid = 1
    numb = 1
    height = 1
    width = 0.3

    if ic_numb == 1: # Gaussian wave for
        if k == None:
            k = random.uniform(10, 30)
        u0 = np.exp(-k * (x - xmid)**2)
    elif ic_numb == 2: # Sinusoidal wave
        if k == None:
            k = 1.5 #random.uniform(1, 2)
        u0 = k*np.sin(np.pi*x)
    elif ic_numb == 3:  # Riemann problem
        if k == None:
            k = random.uniform(1, 2)
        u0 = np.zeros(x.shape[0])
        u0[x >= xmid] = k*1
    elif ic_numb == 4:
        u0 = 1 +  np.sin(4 * np.pi *x)
    elif ic_numb == 5:
        u0 = 3 +  np.sin(2 * np.pi *x)
    elif ic_numb == 6:
        u0 = 0.5 +  np.sin(6 * np.pi *x)


    return u0, numb, xmid, height, width, k