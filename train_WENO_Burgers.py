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
from initial_condition_Burgers import init_cond_B

torch.set_default_dtype(torch.float64)

# TRAIN NETWORK
train_model = WENONetwork()

# DROP PROBLEM FOR TRAINING
#problem_class = Buckley_Leverett
#problem_class = transport_equation
problem_class = Burgers_equation

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
    params_vld.append({'ic_id': 1, 'k': 30})
    params_vld.append({'ic_id': 1, 'k': 15})
    params_vld.append({'ic_id': 2, 'k': 0.5})
    params_vld.append({'ic_id': 2, 'k': 1})
    params_vld.append({'ic_id': 2, 'k': 1.5})
    params_vld.append({'ic_id': 3, 'k': 0.5})
    params_vld.append({'ic_id': 3, 'k': 1})
    params_vld.append({'ic_id': 3, 'k': 1.5})
    return params_vld[j]

u_ex_0 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_3/Basic_test_set/u_ex64_0")
u_ex_1 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_3/Basic_test_set/u_ex64_1")
u_ex_2 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_3/Basic_test_set/u_ex64_2")
u_ex_3 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_3/Basic_test_set/u_ex64_3")
u_ex_4 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_3/Basic_test_set/u_ex64_4")
u_ex_5 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_3/Basic_test_set/u_ex64_5")
u_ex_6 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_3/Basic_test_set/u_ex64_6")
u_ex_7 = torch.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_3/Basic_test_set/u_ex64_7")
u_exs = [u_ex_0, u_ex_1, u_ex_2, u_ex_3, u_ex_4, u_ex_5, u_ex_6, u_ex_7]

#optimizer = optim.SGD(train_model.parameters(), lr=0.1)
optimizer = optim.Adam(train_model.parameters(), lr=0.001)

it = 100
losses = []
all_loss_test = []
df=pd.read_csv("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_3/parameters.txt")
df1 = df[df['ic_id'] == 3]
list_df = [df1['sample_id']]
index = 0

for j in range(it):
    # sample_id=j
    sample_id = random.randint(1,100)
    u_ex = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_3/u_exact64_{}.npy".format(sample_id))
    u_ex = torch.Tensor(u_ex)
    ic_id = float(df[df.sample_id==sample_id]["ic_id"])
    kkk = float(df[df.sample_id==sample_id]["k"])
    # if j not in df1["sample_id"]:
    #     j = j+1
    #     continue
    # sample_id = int(df1.loc[j,:]["sample_id"])
    # kkk = float(df1.loc[j,:]["k"])
    # ic_id = float(df1.loc[j,:]["ic_id"])
    u_ex = np.load("C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Burgers_Equation_Data_3/u_exact64_{}.npy".format(sample_id))
    u_ex = torch.Tensor(u_ex)
    problem_main = problem_class(ic_numb=ic_id, space_steps=64, time_steps=None, params=None)
    params = problem_main.get_params()
    ts = problem_main.time_steps
    problem_main.initial_condition, _, _, _, _, _ = init_cond_B(ic_id, problem_main.x, kkk)
    problem_main.initial_condition = torch.Tensor(problem_main.initial_condition)
    V_init, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False)
    V_train = V_init
    print(j)
    print(sample_id, ic_id, kkk)
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
        #loss = exact_overflows_loss(V_train, u_ex[:,k+1], height)
        loss = exact_loss(V_train, u_ex[:,k+1])
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize weights
        #g = train_model.parameters()
        #x = g.__next__()
        #print(x.detach().numpy().sum(axis=0))
        print(k, loss.data.numpy())
        single_problem_losses.append(loss.detach().numpy().max())
        V_train.detach_()
    losses.append(single_problem_losses)
    path = "C:/Users/Tatiana/Desktop/Research/Research_ML_WENO/Burgers_Equation_Test/Models/Model_14/{}".format(index)
    torch.save(train_model, path)
    index = index + 1
    # TEST IF LOSS IS DECREASING WITH THE NUMBER OF ITERATIONS INCREASING
    for kk in range(8):
        single_problem_loss_test = []
        params = None
        params_vld = validation_problems(kk)
        ic_id_test = params_vld['ic_id']
        kkkk = params_vld['k']
        problem_test = problem_class(ic_numb=ic_id_test, space_steps=64, time_steps=None, params=params)
        problem_test.initial_condition, _, _, _, _, _ = init_cond_B(ic_id_test, problem_test.x, kkkk)
        problem_test.initial_condition = torch.Tensor(problem_test.initial_condition)
        u_init, nn = train_model.init_run_weno(problem_test, vectorized=False, just_one_time_step=False)
        u_test = u_init
        for k in range(nn):
            uu_test = train_model.run_weno(problem_test, u_test, mweno=True,mapped=False,vectorized=False,trainable=True,k=k)
            u_test[:,k+1]=uu_test
        for k in range(nn+1):
            single_problem_loss_test.append(exact_loss(u_test[:,k], u_exs[kk][:, k]).detach().numpy().max())
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
