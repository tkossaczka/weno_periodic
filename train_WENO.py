from define_WENO_Network import WENONetwork
import torch
from torch import optim
from define_problem_transport_eq import transport_equation
from define_problem_Buckley_Leverett import Buckley_Leverett
import numpy as np
import matplotlib.pyplot as plt
from initial_jump_generator import init_jump

torch.set_default_dtype(torch.float64)

# TRAIN NETWORK
train_model = WENONetwork()

# DROP PROBLEM FOR TRAINING
#params = None
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

def exact_overflows_loss(u, u_ex):
    u_max = torch.Tensor([1.0])
    u_min = torch.Tensor([0.0])
    overflows = torch.sum(torch.abs(torch.min(u, u_min)-u_min) + torch.max(u, u_max)-u_max )
    error = train_model.compute_error(u, u_ex)
    loss = error + overflows
    return loss

def overflows_loss(u): #, problem_class, params, problem_main):
    # peeks_left = torch.sum(torch.max(u_left[:-1]-u_left[1:], torch.Tensor([0.0])))
    # peeks_right = torch.sum(torch.abs(torch.min(u_right[:-1] - u_right[1:], torch.Tensor([0.0]))))
    # problem_ex = problem_class(space_steps=100*2*2*2*2*2*2*2, time_steps=50*4*4*4*4*4*4*4, params=params)
    # _, u_ex = train_model.compute_exact(problem_class, problem_ex, 100, 50,
    #                                                       just_one_time_step=True, trainable=False)
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

#optimizer = optim.SGD(train_model.parameters(), lr=0.1)
optimizer = optim.Adam(train_model.parameters())

u_ex_0 = torch.load("u_ex_0")
u_ex_1 = torch.load("u_ex_1")
u_ex_2 = torch.load("u_ex_2")
u_ex_3 = torch.load("u_ex_3")
u_ex_4 = torch.load("u_ex_4")
u_ex_5 = torch.load("u_ex_5")
u_ex_6 = torch.load("u_ex_6")
u_exs = [u_ex_0, u_ex_1, u_ex_2, u_ex_3, u_ex_4, u_ex_5, u_ex_6]

it = 10
losses = []
all_loss_test = []

for j in range(it):
    problem_main = problem_class(ic_numb=6, space_steps=60, time_steps=None, params=None)
    params = problem_main.get_params()
    ts = problem_main.time_steps
    problem_ex = problem_class(ic_numb=6, space_steps=60 * 2 * 2, time_steps=None, params=params)
    _, u_ex = train_model.compute_exact(Buckley_Leverett, problem_ex, 60, ts, just_one_time_step=False, trainable=False)
    V_init, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False)
    V_train = V_init
    print(j)
    print(params)
    single_problem_losses = []
    loss_test = []
    for k in range(nn):
        # Forward path
        V_train = train_model.forward(problem_main,V_train,k)
        # Train model:
        optimizer.zero_grad()  # Clear gradients
        # Calculate loss
        #loss = overflows_loss(V_train)
        loss = exact_overflows_loss(V_train, u_ex[:,k+1])
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize weights
        #g = train_model.parameters()
        #x = g.__next__()
        #print(x.detach().numpy().sum(axis=0))
        print(k, loss.data.numpy())
        single_problem_losses.append(loss.detach().numpy().max())
        V_train.detach_()
        #print(params)
    losses.append(single_problem_losses)
    iteration = j
    path = "model_10_60_36_14/{}".format(iteration)
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
            single_problem_loss_test.append(exact_overflows_loss(u_test[:,k], u_exs[kk][:, k]).detach().numpy().max())
        loss_test.append(single_problem_loss_test)
    all_loss_test.append(loss_test)

#loss_test = np.array(loss_test)
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
