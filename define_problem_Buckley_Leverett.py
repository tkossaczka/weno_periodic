import numpy as np
import torch
import random
from initial_condition_switch import init_cond
from exact_solution_switch import exact_sol
from initial_jump_generator import init_jump

class Buckley_Leverett():
    def __init__(self, ic_numb, space_steps, time_steps=None, params=None, w5_minus='Lax-Friedrichs'):
        """
        Atributes needed to be initialized to make WENO network functional
        space_steps, time_steps, initial_condition, boundary_condition, x, time, h, n
        """
        self.ic_numb = ic_numb
        self.params = params
        if params is None:
            self.init_params()
        self.space_steps = space_steps
        self.time_steps = time_steps
        n, self.t, self.h, self.x, self.time = self.__compute_n_t_h_x_time()
        if time_steps is None:
            self.time_steps = n
        self.initial_condition, self.numb, self.xmid, self.height, self.width = self.__compute_initial_condition()
        #self.boundary_condition = self.__compute_boundary_condition()
        self.w5_minus = w5_minus

    def init_params(self):
        params = dict()
        params["T"] = 0.5
        params["e"] = 10 ** (-13)
        params["L"] = 0
        params["R"] = 2
        params["C"] = random.uniform(0.1, 0.95)
        self.params = params

    def get_params(self):
        return self.params

    def __compute_n_t_h_x_time(self):
        T = self.params["T"]
        L= self.params["L"]
        R= self.params["R"]
        m = self.space_steps
        h = (np.abs(L) + np.abs(R)) / m
        # n = np.ceil(0.5*T/(h**2))
        # n = np.ceil(T / ((2/3) * h**(5/3)))
        # n = int(n)
        # n=50
        n = np.ceil(0.08*T/(h**2))
        #n = np.ceil(0.25 * T / (h ** (5/3)))
        n = int(n)
        t = T / n
        x = np.linspace(L, R-h, m )
        time = np.linspace(0, T, n + 1)
        return n, t, h, x, time

    def __compute_initial_condition(self):
        #m = self.space_steps
        x = self.x
        ic_numb = self.ic_numb
        if ic_numb == 0:
            u_init, numb, xmid, height, width = init_jump(x)
            u_init = torch.Tensor(u_init)
        else:
            u_init, numb, xmid, height, width = init_cond(ic_numb, x)
            u_init = torch.Tensor(u_init)
        # for k in range(0, m + 1):
        #     if x[k] > -0.5 and x[k]<0:
        #         u_init[k] = 1
        #     else:
        #         u_init[k] = 0
        # for k in range(0, m + 1):
        #     if x[k] > -1/np.sqrt(2)-0.4 and x[k] < -1/np.sqrt(2)+0.4:
        #         u_init[k] = 1
        #     elif x[k] > 1/np.sqrt(2)-0.4 and x[k] < 1/np.sqrt(2)+0.4:
        #         u_init[k] = -1
        #     else:
        #         u_init[k] = 0
        # for k in range(0, m + 1):
        #     if x[k] >= 0 and x[k] < 1-1/np.sqrt(2):
        #         u_init[k] = 0
        #     else:
        #         u_init[k] = 1
        return u_init, numb, xmid, height, width

    # def __compute_boundary_condition(self):
    #     n = self.time_steps
    #
    #     u_bc_l = torch.zeros((3, n+1))
    #     u_bc_r = torch.zeros((3, n + 1))
    #
    #     u1_bc_l = torch.zeros((3, n+1))
    #     u1_bc_r = torch.zeros((3, n+1))
    #
    #     u2_bc_l = torch.zeros((3, n+1))
    #     u2_bc_r = torch.zeros((3, n+1))
    #
    #     return u_bc_l, u_bc_r, u1_bc_l, u1_bc_r, u2_bc_l, u2_bc_r

    def der_2(self):
        term_2 = 0
        return term_2

    def der_1(self):
        term_1 = 1
        return term_1

    def der_0(self):
        term_0 = 0
        return term_0

    def der_const(self):
        term_const=0
        return term_const

    def funct_diffusion(self,u):
        #m = self.space_steps
        #u_diff = torch.zeros(m+1)
        u_diff =  (2*u**2 - (4/3)*u**3) # *((u > 0) & (u<1))
        # u_diff = (u) * ((torch.abs(u) > 0.25))
        #u_diff = (u) * ((np.abs(u) > 0.25))
        # for k in range(0, m + 1):
        #     if u[k] < -0.25 or u[k]>0.25:
        #         u_diff[k] = u[k]
        #     else:
        #         u_diff[k] = 0
        return u_diff

    def funct_convection(self, u):
        C = self.params["C"]
        u_conv = (u**2)/(u**2+C*(1-u)**2)
        #u_conv = (4*u**2)/(4*u**2+(1-u)**2)
        #u_conv = (u**2)*(1-5*(1-u)**2)/(u**2+(1-u)**2)
        #u_conv = (u ** 2)  / (u ** 2 + (1 - u) ** 2)
        #u_conv = u**2
        return u_conv

    def funct_derivative(self,u):
        C = self.params["C"]
        u_der = 2*C*u*(1-u)/(u**2+C*(1-u)**2)**2
        #u_der =  8*u*(1-u)/(5*u**2-2*u+1)**2
        #u_der = (-20*u**5+50*u**4-60*u**3+38*u**2-8*u) / (2 * u ** 2 - 2 * u + 1) ** 2
        #u_der = 2 * u * (1 - u) / (2 * u ** 2 - 2 * u + 1) ** 2
        #u_der = 2*u
        return u_der

    def transformation(self, u):
        u = u
        t = self.time
        x = self.x
        return u, x, t



