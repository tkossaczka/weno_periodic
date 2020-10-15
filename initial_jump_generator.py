import numpy as np
import random
import matplotlib.pyplot as plt

def init_jump(x):
    numb = random.choice(range(1, 3))
    if numb==1:
        xmid1 = 1
        height = np.abs(0.7 * np.random.randn()) + 0.1
        width = random.choice(np.linspace(0.1, 0.4, 11))
        u0 = np.zeros(x.shape[0])
        for k in range(0, x.shape[0]):
            if x[k] >= xmid1 - width and x[k] <= xmid1 + width:
                u0[k] = height
        height = [height]
        xmid = [xmid1]
        width = [width]
    elif numb==2:
        xmid1 = 0.5
        xmid2 = 1.5
        height1 = np.abs(0.7 * np.random.randn()) + 0.1
        width1 = random.choice(np.linspace(0.1, 0.4, 11))
        height2 = np.abs(0.7 * np.random.randn()) + 0.1
        width2 = random.choice(np.linspace(0.1, 0.4, 11))
        u01 = np.zeros(x.shape[0])
        u02 = np.zeros(x.shape[0])
        for k in range(0, x.shape[0]):
            if x[k] >= xmid1 - width1 and x[k] <= xmid1 + width1:
                u01[k] = height1
            if x[k] >= xmid2 - width2 and x[k] <= xmid2 + width2:
                u02[k] = height2
        u0 = u01 + u02
        height = [height1, height2]
        xmid = [xmid1,xmid2]
        width = [width1, width2]

    #plt.plot(u0)

    return u0, xmid, height, width