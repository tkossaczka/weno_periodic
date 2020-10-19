import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from initial_jump_generator import init_jump
from define_WENO_Network import WENONetwork
from define_problem_transport_eq import transport_equation
from define_problem_Buckley_Leverett import Buckley_Leverett
from define_problem_Burgers_equation import Burgers_equation

torch.set_default_dtype(torch.float64)

#problem = transport_equation
problem = Buckley_Leverett
#problem = Burgers_equation

def exact_overflows_loss(u, u_ex, height):
    u_max = torch.Tensor([height])
    u_min = torch.Tensor([0.0])
    overflows = torch.sum(torch.abs(torch.min(u, u_min)-u_min) + torch.max(u, u_max)-u_max )
    error = train_model.compute_error(u, u_ex)
    loss = error + overflows
    return loss

rng = 10

err_nt_max_vec = np.zeros(rng)
err_nt_mean_vec = np.zeros(rng)
err_t_max_vec = np.zeros(rng)
err_t_mean_vec = np.zeros(rng)
all_loss_test = []

df=pd.read_csv("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data/Test_set/parameters.txt")

for i in range(100):
    train_model = torch.load('C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Models/Model_00/{}'.format(i))
    loss_test = []
    for j in range(400,480):
        single_problem_loss_test = []
        u_ex = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data/Test_set/u_exact60_{}.npy".format(j))
        u_ex = torch.Tensor(u_ex)
        print(j)
        width = float(df[df.sample_id == j]["width"])
        height = float(df[df.sample_id == j]["height"])
        C = float(df[df.sample_id==j]["C"])
        params = {'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': C}
        my_problem = problem(ic_numb=0, space_steps=60, time_steps=None, params=params)
        my_problem.initial_condition, _, _, _, _ = init_jump(my_problem.x, numb=1, xmid=1, height=height, width=width)
        my_problem.initial_condition = torch.Tensor(my_problem.initial_condition)
        u_t, nn = train_model.init_run_weno(my_problem, vectorized=False, just_one_time_step=False)
        for k in range(nn):
            uu_t = train_model.run_weno(my_problem, u_t, mweno=True,mapped=False,vectorized=False,trainable=True,k=k)
            u_t[:,k+1]=uu_t
        for k in range(nn + 1):
            single_problem_loss_test.append(exact_overflows_loss(u_t[:,k], u_ex[:, k], height).detach().numpy().max())
        loss_test.append(single_problem_loss_test)
    all_loss_test.append(loss_test)

all_loss_test = np.array(all_loss_test)
plt.plot(all_loss_test[:,:,-1])




        # for k in range(nn):
        #     u_nt = train_model.run_weno(my_problem, u_nt, mweno=True, mapped=False, vectorized=True, trainable=False, k=k)
        # u_nt=u_nt.detach().numpy()
        # u_t, nn = train_model.init_run_weno(my_problem, vectorized=True, just_one_time_step=False)
        # for k in range(nn):
        #     u_t = train_model.run_weno(my_problem, u_t, mweno=True, mapped=False, vectorized=True, trainable=True, k=k)
        # u_t = u_t.detach().numpy()
        # _, x, t = my_problem.transformation(u_nt)
        # time_steps = t.shape[0]
        # # problem_ex = problem(ic_numb=2, space_steps=60 * 2 * 2 * 2, time_steps=None, params=params)
        # # _, u_exact_adjusted = train_model.compute_exact_end(Buckley_Leverett, problem_ex, 60, time_steps, just_one_time_step=False, trainable=False)
        # u_exact_adjusted = u_exs[j][:,-1]
        # error_nt_max = np.max(np.abs(u_nt-u_exact_adjusted.detach().numpy()))
        # error_t_max = np.max(np.abs(u_t-u_exact_adjusted.detach().numpy()))
        # error_nt_mean = np.mean((u_nt-u_exact_adjusted.detach().numpy())**2)
        # error_t_mean = np.mean((u_t-u_exact_adjusted.detach().numpy())**2)
        # err_nt_max_vec[j] = error_nt_max
        # err_t_max_vec[j] = error_t_max
        # err_nt_mean_vec[j] = error_nt_mean
        # err_t_mean_vec[j] = error_t_mean
        # plt.figure(j+1)
        # plt.plot(x, u_nt, color='blue', marker='o')
        # plt.plot(x, u_t, marker='o', color='green')
        # plt.plot(x, u_exact_adjusted)

# err_mat = np.zeros((4,7))
# err_mat[0,:] = err_nt_max_vec
# err_mat[1,:] = err_t_max_vec
# err_mat[2,:] = err_nt_mean_vec
# err_mat[3,:] = err_t_mean_vec


# params = validation_problems(0)
# problem_ex_test = problem(ic_numb=6, space_steps=60 * 2 * 2 * 2, time_steps=None, params=params)
# _, u_ex_test = train_model.compute_exact(Buckley_Leverett, problem_ex_test, 60, 36, just_one_time_step=False, trainable=False)
# torch.save(u_ex_test, "u_ex_0_fine")



