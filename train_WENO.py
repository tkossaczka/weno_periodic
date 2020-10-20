from define_WENO_Network import WENONetwork
import torch
from torch import optim
import pandas as pd
import random
from define_problem_transport_eq import transport_equation
from define_problem_Buckley_Leverett import Buckley_Leverett
from define_problem_Burgers_equation import Burgers_equation
import numpy as np
import matplotlib.pyplot as plt
from initial_jump_generator import init_jump

torch.set_default_dtype(torch.float64)

# TRAIN NETWORK
train_model = WENONetwork()

# DROP PROBLEM FOR TRAINING
problem_class = Buckley_Leverett
#problem_class = transport_equation

def monotonicity_loss(u):
    monotonicity = torch.sum(torch.max(u[:-1]-u[1:], torch.Tensor([0.0])))
    loss = monotonicity
    return loss

def exact_loss(u, u_ex):
    error = train_model.compute_error(u, u_ex)
    loss = error
    return loss

def exact_overflows_loss(u, u_ex, height):
    u_max = torch.Tensor([height])
    u_min = torch.Tensor([0.0])
    overflows = torch.sum(torch.abs(torch.min(u, u_min)-u_min) + torch.max(u, u_max)-u_max )
    error = train_model.compute_error(u, u_ex)
    loss = error + overflows
    return loss

def overflows_loss(u):
    u_max = torch.Tensor([1.0])
    u_min = torch.Tensor([0.0])
    overflows = torch.sum(torch.abs(torch.min(u, u_min)-u_min) + torch.max(u, u_max)-u_max )
    loss = overflows # peeks_left + peeks_right
    return loss

def validation_problems(j):
    params_vld = []
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 0.25})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 0.5})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 0.7})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 1.0})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 1.3})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 1.6})
    params_vld.append({'T': 0.5, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 2})
    return params_vld[j]


u_ex_0_fine = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data/Basic_test_set/u_ex_0_fine")
u_ex_1_fine = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data/Basic_test_set/u_ex_1_fine")
u_ex_2_fine = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data/Basic_test_set/u_ex_2_fine")
u_ex_3_fine = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data/Basic_test_set/u_ex_3_fine")
u_ex_4_fine = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data/Basic_test_set/u_ex_4_fine")
u_ex_5_fine = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data/Basic_test_set/u_ex_5_fine")
u_ex_6_fine = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data/Basic_test_set/u_ex_6_fine")
u_exs = [u_ex_0_fine, u_ex_1_fine, u_ex_2_fine, u_ex_3_fine, u_ex_4_fine, u_ex_5_fine, u_ex_6_fine]

# optimizer = optim.SGD(train_model.parameters(), lr=0.1)
optimizer = optim.Adam(train_model.parameters(), lr=1e-3)

it = 20
losses = []
all_loss_test = []
df=pd.read_csv("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1/parameters.txt")

for j in range(it):
    #sample_id=j
    sample_id = random.randint(0,100)
    u_ex = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Buckley_Leverett_Data_1/u_exact60_{}.npy".format(sample_id))
    u_ex = torch.Tensor(u_ex)
    width = float(df[df.sample_id==sample_id]["width"])
    height = float(df[df.sample_id==sample_id]["height"])
    C = float(df[df.sample_id==sample_id]["C"])
    problem_main = problem_class(ic_numb=0, space_steps=60, time_steps=36, params=None)
    problem_main.params["C"] = C
    params = problem_main.get_params()
    ts = problem_main.time_steps
    problem_main.initial_condition, _, _, _, _ = init_jump(problem_main.x, numb=1, xmid=1, height=height, width=width)
    problem_main.initial_condition = torch.Tensor(problem_main.initial_condition)
    V_init, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False)
    V_train = V_init
    print(j)
    print(params)
    print(width, height)
    single_problem_losses = []
    loss_test = []
    for k in range(nn):
        # Forward path
        V_train = train_model.forward(problem_main,V_train,k)
        # Train model:
        optimizer.zero_grad()  # Clear gradients
        # Calculate loss
        #loss = overflows_loss(V_train)
        loss = exact_overflows_loss(V_train, u_ex[:,k+1], height)
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize weights
        #g = train_model.parameters()
        #x = g.__next__()
        #print(x.detach().numpy().sum(axis=0))
        print(k, loss.data.numpy())
        single_problem_losses.append(loss.detach().numpy().max())
        V_train.detach_()
    losses.append(single_problem_losses)
    path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Buckley_Leverett_Test/Models/Model_08/{}".format(j)
    torch.save(train_model, path)
    # TEST IF LOSS IS DECREASING WITH THE NUMBER OF ITERATIONS INCREASING
    for kk in range(7):
        single_problem_loss_test = []
        params = validation_problems(kk)
        problem_test = problem_class(ic_numb=6, space_steps=60, time_steps=None, params=params)
        u_init, nn = train_model.init_run_weno(problem_test, vectorized=False, just_one_time_step=False)
        u_test = u_init
        for k in range(nn):
            uu_test = train_model.run_weno(problem_test, u_test, mweno=True,mapped=False,vectorized=False,trainable=True,k=k)
            u_test[:,k+1]=uu_test
        for k in range(nn+1):
            single_problem_loss_test.append(exact_overflows_loss(u_test[:,k], u_exs[kk][:, k], height=1).detach().numpy().max())
        loss_test.append(single_problem_loss_test)
    all_loss_test.append(loss_test)

losses = np.array(losses)
all_loss_test = np.array(all_loss_test) #shape (training_steps, num_valid_problems, time_steps)
# get (training_steps, num_valid_problems):
plt.plot(all_loss_test[:,:,-1])
# plt.plot(all_loss_test.sum(axis=2))
# plt.plot(all_loss_test.sum(axis=1).T)

# for k in range(it):
#     plt.plot(losses[k, :])
#
# plt.plot(losses.sum(axis=1))

#plt.plot(loss_test.sum(axis=1))

#plt.plot(S, V_train.detach().numpy())
#print("number of parameters:", sum(p.numel() for p in train_model.parameters()))
# g=train_model.parameters()
# g.__next__()

#torch.save(train_model, "model_10_60_36_8")
