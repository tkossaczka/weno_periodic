import numpy as np
import torch
from initial_condition_switch import init_cond_switch
from exact_solution_switch import exact_cond_switch

class transport_equation():
    def __init__(self, space_steps, time_steps=None, params=None, w5_minus='Lax-Friedrichs'):
        """
        Atributes needed to be initialized to make WENO network functional
        space_steps, time_steps, initial_condition, boundary_condition, x, time, h, n
        """

        self.params = params
        if params is None:
            self.init_params()
        self.space_steps = space_steps
        self.time_steps = time_steps
        n, self.t, self.h, self.x, self.time = self.__compute_n_t_h_x_time()
        if time_steps is None:
            self.time_steps = n
        self.initial_condition = self.__compute_initial_condition()
        #self.boundary_condition = self.__compute_boundary_condition()
        self.w5_minus = w5_minus

    def init_params(self):
        params = dict()
        params["T"] = 10 #5 #1
        params["e"] = 10 ** (-13)
        params["L"] = 0 #0 # -1
        params["R"] = 2 #2 # 1
        self.params = params

    def get_params(self):
        return self.params

    def __compute_n_t_h_x_time(self):
        T = self.params["T"]
        L= self.params["L"]
        R = self.params["R"]
        m = self.space_steps
        h = (np.abs(L) + np.abs(R)) / m
        n = np.ceil(T / ((2/3) * h**(5/3)))  #0.4 pre sinus
        n = int(n)
        t = T / n
        x = np.linspace(L, R-h, m)
        time = np.linspace(0, T, n + 1)
        return n, t, h, x, time

    def __compute_initial_condition(self):
        #m = self.space_steps
        x = self.x
        IC_object = init_cond_switch(x)
        u_init = IC_object.case_1(x)
        u_init = torch.Tensor(u_init)

        # u_init = torch.zeros(m)
        # for k in range(0, m ):
        #     if x[k] > 1:
        #         u_init[k] = 1
        #     else:
        #         u_init[k] = 0
        # for k in range(0, m ):
        #     u_init[k] = np.sin(np.pi*x[k])
        return u_init

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

    def funct_convection(self, u):
        return u

    def funct_diffusion(self, u):
        return u

    def funct_derivative(self, u):
        u_der = u ** 0
        return u_der

    def exact(self):
        x = self.x
        t = self.time
        EC_object = exact_cond_switch(x)
        u_exact = EC_object.case_1(x,t)
        # m = self.space_steps
        # n,_, _,_,_ = self.__compute_n_t_h_x_time()
        # x, time = self.x, self.time
        # uex = np.zeros((m, n + 1))
        # for k in range(0, n + 1):
        #     for j in range(0, m):
        #         uex[j, k] = np.sin(np.pi*(x[j]-time[k]))
        # u_ex = uex[:, n]
        return u_exact

    # def err(self, u):
    #     uex = self.exact()
    #     t = self.time
    #     #u_last = u_last.detach().numpy()
    #     u = u.detach().numpy()
    #     error =np.zeros(t.shape[0])
    #     for i in range(0,t.shape[0]):
    #         error[i] = np.max(np.absolute(uex[:,i] - u[:,i]))
    #     #xmaxerr = np.max(xerr)
    #     return error

    def err(self, u, k):
        uex = self.exact()
        uex = uex[:,k]
        t = self.time
        #u_last = u_last.detach().numpy()
        u = u.detach().numpy()
        error =np.zeros(t.shape[0])
        error = np.max(np.absolute(uex - u))
        #xmaxerr = np.max(xerr)
        return error

    def transformation(self, u):
        u = u
        t = self.time
        x = self.x
        return u, x, t