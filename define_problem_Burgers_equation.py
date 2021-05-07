import numpy as np
import torch
from initial_condition_switch import init_cond
from exact_solution_switch import exact_sol
from initial_jump_generator import init_jump
from initial_condition_Burgers import init_cond_B

class Burgers_equation():
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
        self.initial_condition, self.numb, self.xmid, self.height, self.width, self.k = self.__compute_initial_condition()
        self.w5_minus = w5_minus

    def init_params(self):
        params = dict()
        params["T"] = 0.3
        params["e"] = 10 ** (-13)
        params["L"] = 0
        params["R"] = 2
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
        #n = np.ceil(0.08*T/(h**2))
        n=25*4 *4*4*4
        #n = np.ceil(0.25 * T / (h ** (5/3)))
        n = int(n)
        t = T / n
        x = np.linspace(L, R-h, m )
        time = np.linspace(0, T, n + 1)
        return n, t, h, x, time

    def __compute_initial_condition(self):
        x = self.x
        ic_numb = self.ic_numb
        if ic_numb == 0:
            u_init, numb, xmid, height, width = init_jump(x)
            u_init = torch.Tensor(u_init)
        else:
            u_init, numb, xmid, height, width, k = init_cond_B(ic_numb, x)
            u_init = torch.Tensor(u_init)
        return u_init, numb, xmid, height, width, k

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
        u_diff = 0
        return u_diff

    def funct_convection(self, u):
        # u_conv = (u**2)/(2)
        u_conv = (u**4)/16
        return u_conv

    def funct_derivative(self,u):
        u_der = u
        return u_der

    def transformation(self, u):
        u = u
        t = self.time
        x = self.x
        return u, x, t
