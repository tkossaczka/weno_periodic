import numpy as np
import random
import matplotlib.pyplot as plt

def init_jump(x, numb=None, xmid=None, height=None, width=None):
    if xmid == None:
        xmid = 1
        height = 1 #np.minimum(np.abs(0.7 * np.random.randn()) + 0.1,1.5)
        width = random.choice(np.linspace(0.2, 0.4, 110))
    u0 = np.zeros(x.shape[0])
    for k in range(0, x.shape[0]):
        if x[k] >= xmid - width and x[k] <= xmid + width:
            u0[k] = height

    # if numb == None:
    #     #numb = random.choice(range(1, 3))
    #     numb = 1
    # if numb==1:
    #     if xmid == None:
    #         xmid1 = 1
    #         height = np.minimum(np.abs(0.7 * np.random.randn()) + 0.1,1.5)
    #         width = random.choice(np.linspace(0.1, 0.4, 11))
    #         height = [height]
    #         xmid = [xmid1]
    #         width = [width]
    #     u0 = np.zeros(x.shape[0])
    #     for k in range(0, x.shape[0]):
    #         if x[k] >= xmid[0] - width[0] and x[k] <= xmid[0] + width[0]:
    #             u0[k] = height[0]
    # elif numb==2:
    #     if xmid == None:
    #         xmid1 = 0.5
    #         xmid2 = 1.5
    #         height1 = np.abs(0.7 * np.random.randn()) + 0.1
    #         width1 = random.choice(np.linspace(0.1, 0.4, 11))
    #         height2 = np.abs(0.7 * np.random.randn()) + 0.1
    #         width2 = random.choice(np.linspace(0.1, 0.4, 11))
    #         height = [height1, height2]
    #         xmid = [xmid1, xmid2]
    #         width = [width1, width2]
    #     u01 = np.zeros(x.shape[0])
    #     u02 = np.zeros(x.shape[0])
    #     for k in range(0, x.shape[0]):
    #         if x[k] >= xmid[0] - width[0] and x[k] <= xmid[0] + width[0]:
    #             u01[k] = height[0]
    #         if x[k] >= xmid[1] - width[1] and x[k] <= xmid[1] + width[1]:
    #             u02[k] = height[1]
    #     u0 = u01 + u02
    #
    # # plt.plot(u0)

    return u0, numb, xmid, height, width