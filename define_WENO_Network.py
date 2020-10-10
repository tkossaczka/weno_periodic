import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import torch
from torch import nn, optim # Sets of preset layers and optimizers
from scipy.stats import norm
import torch.nn.functional as F # Sets of functions such as ReLU
from torchvision import datasets, transforms # Popular datasets, architectures and common

class WENONetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_nn_weno5 = self.get_inner_nn_weno5()
        self.inner_nn_weno6 = self.get_inner_nn_weno6()
        self.weno5_mult_bias, self.weno6_mult_bias = self.get_multiplicator_biases()

    def get_inner_nn_weno5(self):
        net = nn.Sequential(
            nn.Conv1d(1, 20, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
            nn.Conv1d(20, 40, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
            nn.Conv1d(40, 80, kernel_size=1, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(80, 40, kernel_size=1, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(40, 20, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv1d(20, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())
        return net

    def get_inner_nn_weno6(self):
        net = nn.Sequential(
            nn.Conv1d(1, 20, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
            nn.Conv1d(20, 40, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
            nn.Conv1d(40, 80, kernel_size=1, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(80, 40, kernel_size=1, stride=1, padding=0),
            nn.ELU(),
            nn.Conv1d(40, 20, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv1d(20, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())
        return net

    def get_multiplicator_biases(self):
        # first for weno 5, second for weno 6
        return 0.1, 0.1

    def WENO5(self, uu, e, w5_minus, mweno=True, mapped=False, trainable=True):
        uu_left = uu
        uu_right = torch.roll(uu, -1)

        def get_fluxes(uu):
            uummm = torch.roll(uu, 3)
            uumm = torch.roll(uu, 2)
            uum = torch.roll(uu, 1)
            uup = torch.roll(uu, -1)
            uupp = torch.roll(uu, -2)
            if w5_minus is True:
                flux0 = (11 * uu - 7 * uup + 2 * uupp) / 6
                flux1 = (2 * uum + 5 * uu - uup) / 6
                flux2 = (-uumm + 5 * uum + 2 * uu) / 6
            else:
                flux0 = (2 * uummm - 7 * uumm + 11 * uum) / 6
                flux1 = (- uumm + 5 * uum + 2* uu) / 6
                flux2 = (2*uum + 5 * uu - uup) / 6
            return flux0, flux1, flux2

        fluxp0, fluxp1, fluxp2 = get_fluxes(uu_right)
        fluxn0, fluxn1, fluxn2 = get_fluxes(uu_left)

        def get_betas(uu):
            uummm = torch.roll(uu, 3)
            uumm = torch.roll(uu, 2)
            uum = torch.roll(uu, 1)
            uup = torch.roll(uu, -1)
            uupp = torch.roll(uu, -2)
            if w5_minus is True:
                beta0 = 13 / 12 * (uu - 2 * uup + uupp) ** 2 + 1 / 4 * (
                            3 * uu - 4 * uup + uupp) ** 2
                beta1 = 13 / 12 * (uum - 2 * uu + uup) ** 2 + 1 / 4 * (uum - uup) ** 2
                beta2 = 13 / 12 * (uumm - 2 * uum + uu) ** 2 + 1 / 4 * (
                            uumm - 4 * uum + 3 * uu) ** 2
            else:
                beta0 = 13 / 12 * (uummm - 2 * uumm + uum) ** 2 + 1 / 4 * (
                        uummm - 4 * uumm + 3*uum) ** 2
                beta1 = 13 / 12 * (uumm - 2 * uum + uu) ** 2 + 1 / 4 * (uumm - uu) ** 2
                beta2 = 13 / 12 * (uum - 2 * uu + uup) ** 2 + 1 / 4 * (
                        3*uum - 4 * uu + uup) ** 2
            return beta0, beta1, beta2

        betap0, betap1, betap2 = get_betas(uu_right)
        betan0, betan1, betan2 = get_betas(uu_left)

        if trainable:
            dif = self.__get_average_diff(uu)
            beta_multiplicators = self.inner_nn_weno5(dif[None, None, :])[0, 0, :] + self.weno5_mult_bias
            # beta_multiplicators_left = beta_multiplicators[:-1]
            # beta_multiplicators_right = beta_multiplicators[1:]

            betap_corrected_list = []
            betan_corrected_list = []
            for k, beta in enumerate([betap0, betap1, betap2]):
                shift = k -1
                betap_corrected_list.append(beta * torch.roll(beta_multiplicators, shifts=shift, dims=0))
            for k, beta in enumerate([betan0, betan1, betan2]):
                shift = k - 1
                betan_corrected_list.append(beta * torch.roll(beta_multiplicators, shifts=shift, dims=0))
            [betap0, betap1, betap2] = betap_corrected_list
            [betan0, betan1, betan2] = betan_corrected_list

        d0 = 1 / 10;
        d1 = 6 / 10;
        d2 = 3 / 10;

        def get_omegas_mweno(betas, ds):
            beta_range_square = (betas[2] - betas[0]) ** 2
            return [d / (e + beta) ** 2 * (beta_range_square + (e + beta) ** 2) for beta, d in zip(betas, ds)]

        def get_omegas_weno(betas, ds):
            return [d / (e + beta) ** 2 for beta, d in zip(betas, ds)]

        omegas_func_dict = {0: get_omegas_weno, 1: get_omegas_mweno}
        [omegap_0, omegap_1, omegap_2] = omegas_func_dict[int(mweno)]([betap0, betap1, betap2], [d0, d1, d2])
        [omegan_0, omegan_1, omegan_2] = omegas_func_dict[int(mweno)]([betan0, betan1, betan2], [d0, d1, d2])

        def normalize(tensor_list):
            sum_ = sum(tensor_list)  # note, that inbuilt sum applies __add__ iteratively therefore its overloaded-
            return [tensor / sum_ for tensor in tensor_list]

        [omegap0, omegap1, omegap2] = normalize([omegap_0, omegap_1, omegap_2])
        [omegan0, omegan1, omegan2] = normalize([omegan_0, omegan_1, omegan_2])
        #[omegap0, omegap1, omegap2] = [d0, d1, d2]
        #[omegan0, omegan1, omegan2] = [d0, d1, d2]

        if mapped:
            def get_alpha(omega, d):
                return (omega * (d + d ** 2 - 3 * d * omega + omega ** 2)) / (d ** 2 + omega * (1 - 2 * d))

            [alphap0, alphap1, alphap2] = [get_alpha(omega, d) for omega, d in zip([omegap0, omegap1, omegap2],
                                                                                   [d0, d1, d2])]
            [alphan0, alphan1, alphan2] = [get_alpha(omega, d) for omega, d in zip([omegan0, omegan1, omegan2],
                                                                                   [d0, d1, d2])]

            [omegap0, omegap1, omegap2] = normalize([alphap0, alphap1, alphap2])
            [omegan0, omegan1, omegan2] = normalize([alphan0, alphan1, alphan2])

        fluxp = omegap0 * fluxp0 + omegap1 * fluxp1 + omegap2 * fluxp2
        fluxn = omegan0 * fluxn0 + omegan1 * fluxn1 + omegan2 * fluxn2

        RHS = (fluxp - fluxn)
        return RHS

    def WENO6(self, uu, e, mweno=True, mapped=False, trainable=True):
        uu_left = uu
        uu_right = torch.roll(uu, -1)

        def get_fluxes(uu):
            uummm = torch.roll(uu, 3)
            uumm = torch.roll(uu, 2)
            uum = torch.roll(uu, 1)
            uup = torch.roll(uu, -1)
            uupp = torch.roll(uu, -2)
            flux0 = (uummm - 3 * uumm - 9 * uum + 11 * uu) / 12
            flux1 = (uumm - 15 * uum + 15 * uu - uup) / 12
            flux2 = (-11 * uum + 9 * uu + 3 * uup - uupp) / 12
            return flux0, flux1, flux2

        fluxp0, fluxp1, fluxp2 = get_fluxes(uu_right)
        fluxn0, fluxn1, fluxn2 = get_fluxes(uu_left)

        def get_betas(uu):
            uummm = torch.roll(uu, 3)
            uumm = torch.roll(uu, 2)
            uum = torch.roll(uu, 1)
            uup = torch.roll(uu, -1)
            uupp = torch.roll(uu, -2)
            beta0 = 13 / 12 * (uummm - 3 * uumm + 3 * uum - uu) ** 2 + 1 / 4 * (
                        uummm - 5 * uumm + 7 * uum - 3 * uu) ** 2
            beta1 = 13 / 12 * (uumm - 3 * uum + 3 * uu - uup) ** 2 + 1 / 4 * (
                        uumm - uum - uu + uup) ** 2
            beta2 = 13 / 12 * (uum - 3 * uu + 3 * uup - uupp) ** 2 + 1 / 4 * (
                        -3 * uum + 7 * uu - 5 * uup + uupp) ** 2
            return beta0, beta1, beta2

        betap0, betap1, betap2 = get_betas(uu_right)
        betan0, betan1, betan2 = get_betas(uu_left)

        if trainable:
            dif = self.__get_average_diff(uu)
            beta_multiplicators = self.inner_nn_weno6(dif[None, None, :])[0, 0, :] + self.weno6_mult_bias
            # beta_multiplicators_left = beta_multiplicators[:-1]
            # beta_multiplicators_right = beta_multiplicators[1:]

            betap_corrected_list = []
            betan_corrected_list = []
            for k, beta in enumerate([betap0, betap1, betap2]):
                shift = k -1
                betap_corrected_list.append(beta * (beta_multiplicators[3+shift:-3+shift]))
            for k, beta in enumerate([betan0, betan1, betan2]):
                shift = k - 1
                betan_corrected_list.append(beta * (beta_multiplicators[3+shift:-3+shift]))
            [betap0, betap1, betap2] = betap_corrected_list
            [betan0, betan1, betan2] = betan_corrected_list

        gamap0 = 1 / 21
        gamap1 = 19 / 21
        gamap2 = 1 / 21
        gaman0 = 4 / 27
        gaman1 = 19 / 27
        gaman2 = 4 / 27
        sigmap = 42 / 15
        sigman = 27 / 15

        def get_omegas_mweno(betas, gamas):
            beta_range_square = (betas[2] - betas[0]) ** 2
            return [gama / (e + beta) ** 2 * (beta_range_square + (e + beta) ** 2) for beta, gama in zip(betas, gamas)]

        def get_omegas_weno(betas, gamas):
            return [gama / (e + beta) ** 2 for beta, gama in zip(betas, gamas)]

        omegas_func_dict = {0: get_omegas_weno, 1: get_omegas_mweno}
        [omegapp_0, omegapp_1, omegapp_2] = omegas_func_dict[int(mweno)]([betap0, betap1, betap2],
                                                                         [gamap0, gamap1, gamap2])
        [omeganp_0, omeganp_1, omeganp_2] = omegas_func_dict[int(mweno)]([betap0, betap1, betap2],
                                                                         [gaman0, gaman1, gaman2])
        [omegapn_0, omegapn_1, omegapn_2] = omegas_func_dict[int(mweno)]([betan0, betan1, betan2],
                                                                         [gamap0, gamap1, gamap2])
        [omegann_0, omegann_1, omegann_2] = omegas_func_dict[int(mweno)]([betan0, betan1, betan2],
                                                                         [gaman0, gaman1, gaman2])

        def normalize(tensor_list):
            sum_ = sum(tensor_list)  # note, that inbuilt sum applies __add__ iteratively therefore its overloaded-
            return [tensor / sum_ for tensor in tensor_list]

        [omegapp0, omegapp1, omegapp2] = normalize([omegapp_0, omegapp_1, omegapp_2])
        [omeganp0, omeganp1, omeganp2] = normalize([omeganp_0, omeganp_1, omeganp_2])
        [omegapn0, omegapn1, omegapn2] = normalize([omegapn_0, omegapn_1, omegapn_2])
        [omegann0, omegann1, omegann2] = normalize([omegann_0, omegann_1, omegann_2])

        omegaps = [omegapp0, omegapp1, omegapp2, omegapn0, omegapn1, omegapn2]
        omegans = [omeganp0, omeganp1, omeganp2, omegann0, omegann1, omegann2]

        [omegap0, omegap1, omegap2, omegan0, omegan1, omegan2] = [sigmap * omegap - sigman * omegan
                                                                  for omegap, omegan in zip(omegaps, omegans)]

        if mapped:
            d0 = -2 / 15
            d1 = 19 / 15
            d2 = -2 / 15

            def get_alpha(omega, d):
                return (omega * (d + d ** 2 - 3 * d * omega + omega ** 2)) / (d ** 2 + omega * (1 - 2 * d))

            [alphap0, alphap1, alphap2] = [get_alpha(omega, d) for omega, d in zip([omegap0, omegap1, omegap2],
                                                                                   [d0, d1, d2])]
            [alphan0, alphan1, alphan2] = [get_alpha(omega, d) for omega, d in zip([omegan0, omegan1, omegan2],
                                                                                   [d0, d1, d2])]
            [omegap0, omegap1, omegap2] = normalize([alphap0, alphap1, alphap2])
            [omegan0, omegan1, omegan2] = normalize([alphan0, alphan1, alphan2])

        fluxp = omegap0 * fluxp0 + omegap1 * fluxp1 + omegap2 * fluxp2
        fluxn = omegan0 * fluxn0 + omegan1 * fluxn1 + omegan2 * fluxn2

        RHS = (fluxp - fluxn)

        return RHS

    def __get_average_diff(self, uu):
        dif = uu[1:] - uu[:-1]
        dif_left = torch.zeros_like(uu)
        dif_right = torch.zeros_like(uu)
        dif_left[:-1] = dif
        dif_left[-1] = dif[-1]
        dif_right[1:] = dif
        dif_right[0] = dif[0]
        dif_final = 0.5 * dif_left + 0.5 * dif_right
        return dif_final

    def init_run_weno(self, problem, vectorized, just_one_time_step):
        m = problem.space_steps
        n, t, h = problem.time_steps, problem.t, problem.h
        # x, time = problem.x, problem.time
        # w5_minus = problem.w5_minus

        if vectorized:
            u = problem.initial_condition
        else:
            u = torch.zeros((m, n + 1))
            u[:, 0] = problem.initial_condition

        if just_one_time_step is True:
            nn = 1
        else:
            nn = n
        return u, nn

    def run_weno(self, problem, u, mweno, mapped, vectorized, trainable, k):
        n, t, h = problem.time_steps, problem.t, problem.h
        e = problem.params['e']
        # term_2 = problem.der_2()
        term_1 = problem.der_1()
        # term_0 = problem.der_0()
        # term_const = problem.der_const()

        u = torch.Tensor(u)
        if vectorized:
            uu = u
            #ll=1
        else:
            uu = u[:,k]
            #ll=k

        uu_conv = problem.funct_convection(uu)
        #uu_diff = problem.funct_diffusion(u[:, ll - 1])
        #RHSd = self.WENO6(uu_diff, e, mweno=mweno, mapped=mapped, trainable=trainable)
        max_der = torch.max(torch.abs(problem.funct_derivative(uu)))
        RHSc_p = self.WENO5(0.5*(uu_conv+max_der*uu), e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
        RHSc_n = self.WENO5(0.5*(uu_conv-max_der*uu), e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
        RHSc = RHSc_p + RHSc_n
        # u1 = u[:, ll - 1] + t * ((term_2 / h ** 2) * RHSd - (term_1 / h) * RHSc + term_0 * u[:, ll - 1])
        u1 = uu + t * ( - (term_1 / h) * RHSc )

        uu1_conv = problem.funct_convection(u1)
        #uu1_diff = problem.funct_diffusion(u1)
        #RHS1d = self.WENO6(uu1_diff, e, mweno=mweno, mapped=mapped, trainable=trainable)
        max_der = torch.max(torch.abs(problem.funct_derivative(u1)))
        RHS1c_p = self.WENO5(0.5*(uu1_conv+max_der*u1), e, w5_minus=False, mweno=mweno, mapped=mapped, trainable=trainable)
        RHS1c_n = self.WENO5(0.5*(uu1_conv-max_der*u1), e, w5_minus=True, mweno=mweno, mapped=mapped, trainable=trainable)
        RHS1c = RHS1c_p + RHS1c_n
        # u2 = 0.75*u[:,ll-1]+0.25*u1+0.25*t*((term_2/h ** 2)*RHS1d-(term_1/h)*RHS1c+term_0*u1)
        u2 = 0.75*uu+0.25*u1+0.25*t*(-(term_1/h)*RHS1c)

        uu2_conv = problem.funct_convection(u2)
        #uu2_diff = problem.funct_diffusion(u2)
        #RHS2d = self.WENO6(uu2_diff, e, mweno=mweno, mapped=mapped, trainable=trainable)
        max_der = torch.max(torch.abs(problem.funct_derivative(u2)))
        RHS2c_p = self.WENO5(0.5 * (uu2_conv + max_der * u2), e, w5_minus=False, mweno=mweno, mapped=mapped,
                             trainable=trainable)
        RHS2c_n = self.WENO5(0.5 * (uu2_conv - max_der * u2), e, w5_minus=True, mweno=mweno, mapped=mapped,
                             trainable=trainable)
        RHS2c = RHS2c_p + RHS2c_n
        # if vectorized:
        #     u[:, 0] = (1 / 3) * u[:, ll - 1] + (2 / 3) * u2 + (2 / 3) * t * (
        #             (term_2 / h ** 2) * RHS2d - (term_1 / h) * RHS2c + term_0 * u2)
        # else:
        #     u[:, l] = (1 / 3) * u[:, ll - 1] + (2 / 3) * u2+ (2 / 3) * t * (
        #             (term_2 / h ** 2) * RHS2d - (term_1 / h) * RHS2c + term_0 * u2)
        if vectorized:
            u_ret = (1 / 3) * uu + (2 / 3) * u2 + (2 / 3) * t * (- (term_1 / h) * RHS2c)
        else:
            u[:, k+1] = (1 / 3) * uu + (2 / 3) * u2+ (2 / 3) * t * (- (term_1 / h) * RHS2c)
            u_ret = u[:,k+1]

        return u_ret

    def forward(self, problem, u_ret, k):
        u = self.run_weno(problem,u_ret,mweno=True,mapped=False,vectorized=True,trainable=True,k=k)
        V,_,_ = problem.transformation(u)
        return V

    def get_axes(self, problem, u):
        _, S, t = problem.transformation(u)
        return S, t

    def full_WENO(self, problem, trainable, plot=True, vectorized=False):
        u, nn = self.init_run_weno(problem, vectorized=vectorized, just_one_time_step=False)
        for k in range(nn):
            uu = self.run_weno(problem, u, mweno=True, mapped=False,vectorized=vectorized,trainable=trainable,k=k)
            u[:,k+1]=uu
        V, S, tt = problem.transformation(u)
        V = V.detach().numpy()
        if plot:
            X, Y = np.meshgrid(S, tt, indexing="ij")
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, V, cmap=cm.viridis)
        return V, S, tt

    def compare_wenos(self, problem):
        u_init, nn = self.init_run_weno(problem, vectorized=True, just_one_time_step=True)
        u_trained = self.run_weno(problem, u_init, mweno=True, mapped= False, vectorized=True, trainable=True, k=1)
        V_trained, S, tt = problem.transformation(u_trained)
        u_classic = self.run_weno(problem, u_init, mweno=True, mapped= False, vectorized=True, trainable=False, k=1)
        V_classic, S, tt = problem.transformation(u_classic)
        plt.plot(S, V_classic.detach().numpy(), S, V_trained.detach().numpy())

    def compute_exact_end(self, problem_class, problem, space_steps, time_steps, just_one_time_step, trainable):
        if hasattr(problem_class, 'exact'):
            print('nothing to do')
        else:
            u, nn = self.init_run_weno(problem, vectorized=True, just_one_time_step=just_one_time_step)
            for k in range(nn):
                u = self.run_weno(problem, u, mweno=True, mapped=False, vectorized=True, trainable=trainable, k=k)
        u_exact = u
        space_steps_exact = problem.space_steps
        time_steps_exact = problem.time_steps
        divider_space = space_steps_exact / space_steps
        divider_time = time_steps_exact / time_steps
        divider_space = int(divider_space)
        divider_time = int(divider_time)
        u_exact_adjusted = u_exact[0:space_steps_exact+1:divider_space] #,0:time_steps_exact+1:divider_time]
        return u_exact, u_exact_adjusted

    def compute_exact(self, problem_class, problem, space_steps, time_steps, just_one_time_step, trainable):
        if hasattr(problem_class, 'exact'):
            print('nothing to do')
        else:
            space_steps_exact = problem.space_steps
            time_steps_exact = problem.time_steps
            #u = np.zeros((space_steps_exact,time_steps_exact))
            u, nn = self.init_run_weno(problem, vectorized=False, just_one_time_step=just_one_time_step)
            for k in range(nn):
                u[:,k+1] = self.run_weno(problem, u, mweno=True, mapped=False, vectorized=False, trainable=trainable, k=k)
        u_exact = u
        divider_space = space_steps_exact / space_steps
        divider_time = time_steps_exact / time_steps
        divider_space = int(divider_space)
        divider_time = int(divider_time)
        u_exact_adjusted = u_exact[0:space_steps_exact+1:divider_space, 0:time_steps_exact+1:divider_time]
        return u_exact, u_exact_adjusted

    def compute_error(self, u, u_ex):
        u_last = u
        u_ex_last = u_ex
        xerr = (u_ex_last - u_last)**2
        #xerr =torch.abs(u_ex_last - u_last)
        #err = torch.max(xerr)
        err = torch.mean(xerr)
        #print(xerr)
        return err

    def order_compute(self, iterations, initial_space_steps, initial_time_steps, params, problem_class, trainable, ic_numb):
        problem = problem_class(ic_numb=ic_numb, space_steps=initial_space_steps, time_steps=initial_time_steps, params=params)
        vecerr = np.zeros((iterations))[:, None]
        order = np.zeros((iterations - 1))[:, None]
        if hasattr(problem_class,'exact'):
            u, nn = self.init_run_weno(problem, vectorized=True, just_one_time_step=False)
            for k in range(nn):
                u = self.run_weno(problem, u, mweno=True, mapped=False, vectorized=True, trainable=trainable, k=k)
            u_last = u
            xmaxerr = problem.err(u_last)
            vecerr[0] = xmaxerr
            print(problem.space_steps, problem.time_steps)
            for i in range(1, iterations):
                if initial_time_steps is None:
                    spec_time_steps = None
                else:
                    spec_time_steps = problem.time_steps*4
                problem = problem_class(ic_numb=ic_numb, space_steps=problem.space_steps * 2, time_steps=spec_time_steps, params=params)
                u, nn = self.init_run_weno(problem, vectorized=True, just_one_time_step=False)
                for k in range(nn):
                    u = self.run_weno(problem, u, mweno=True, mapped=False, vectorized=True, trainable=trainable, k=k)
                u_last = u
                xmaxerr = problem.err(u_last)
                vecerr[i] = xmaxerr
                order[i - 1] = np.log(vecerr[i - 1] / vecerr[i]) / np.log(2)
                print(problem.space_steps, problem.time_steps)
        else:
            u, nn = self.init_run_weno(problem, vectorized=True, just_one_time_step=False)
            for k in range(nn):
                u = self.run_weno(problem, u, mweno=True, mapped=False, vectorized=True, trainable=trainable, k=k)
            u_last = u
            u_last = u_last.detach().numpy()
            fine_space_steps = initial_space_steps*2*2*2*2*2
            if initial_time_steps is None:
                fine_time_steps = None
            else:
                fine_time_steps = initial_time_steps*4*4*4*4*4
            problem_fine = problem_class(ic_numb=ic_numb, space_steps=fine_space_steps, time_steps=fine_time_steps, params=params)
            m = problem.space_steps
            u_ex, nn = self.init_run_weno(problem_fine, vectorized=True, just_one_time_step=False)
            for k in range(nn):
                u_ex = self.run_weno(problem_fine, u_ex, mweno=True, mapped=False, vectorized=True, trainable=False, k=k)
            u_ex = u_ex.detach().numpy()
            divider = fine_space_steps/m
            divider = int(divider)
            u_ex_short = u_ex[0:fine_space_steps+1:divider]
            xerr = np.absolute(u_ex_short - u_last)
            xmaxerr = np.max(xerr)
            vecerr[0] = xmaxerr
            print(problem.space_steps, problem.time_steps)
            for i in range(1, iterations):
                if initial_time_steps is None:
                    spec_time_steps = None
                else:
                    spec_time_steps = problem.time_steps*4
                problem = problem_class(ic_numb=ic_numb, space_steps=problem.space_steps * 2, time_steps=spec_time_steps, params=params)
                m = problem.space_steps
                u, nn = self.init_run_weno(problem, vectorized=True, just_one_time_step=False)
                for k in range(nn):
                    u = self.run_weno(problem, u, mweno=True, mapped=False, vectorized=True, trainable=trainable, k=k)
                u_last = u
                u_last = u_last.detach().numpy()
                divider = fine_space_steps / m
                divider = int(divider)
                u_ex_short = u_ex[0:fine_space_steps+1:divider]
                xerr = np.absolute(u_ex_short - u_last)
                xmaxerr = np.max(xerr)
                vecerr[i] = xmaxerr
                order[i - 1] = np.log(vecerr[i - 1] / vecerr[i]) / np.log(2)
                print(problem.space_steps, problem.time_steps)
        return vecerr, order




