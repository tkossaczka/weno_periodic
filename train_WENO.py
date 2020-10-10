from define_WENO_Network import WENONetwork
import torch
from torch import optim
from define_problem_transport_eq import transport_equation
from define_problem_Buckley_Leverett import Buckley_Leverett

torch.set_default_dtype(torch.float64)

# TRAIN NETWORK
train_model = WENONetwork()

# DROP PROBLEM FOR TRAINING
#params = None
problem_class = Buckley_Leverett
#problem_class = transport_equation

def monotonicity_loss(u): #, problem_class, params, problem_main):
    monotonicity = torch.sum(torch.max(u[:-1]-u[1:], torch.Tensor([0.0])))
    loss = monotonicity
    return loss

def exact_overflows_loss(u, u_ex): #, problem_class, params, problem_main):  just_one_time_step=True, trainable=False)
    u_max = torch.max(u_ex)
    u_min = torch.min(u_ex)
    overflows = torch.sum(torch.abs(torch.min(u, u_min)-u_min) + torch.max(u, u_max)-u_max )
    error = train_model.compute_error(u, u_ex)
    loss = error + overflows
    return loss

def exact2_overflows_loss(u, u_ex): #, problem_class, params, problem_main): just_one_time_step=True, trainable=False)
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


# problem_ex = problem_class(ic_numb=6, space_steps=60*2*2, time_steps=10*4*4, params=params)
#     _, u_ex = train_model.compute_exact_end(Buckley_Leverett, problem_ex, 60, 10,
#                                             just_one_time_step=False, trainable=False)

#optimizer = optim.SGD(train_model.parameters(), lr=0.1)
optimizer = optim.Adam(train_model.parameters())

for j in range(10):
    problem_main = problem_class(ic_numb=6, space_steps=60, time_steps=10, params=None)
    params = problem_main.get_params()
    #ts = problem_main.time_steps
    problem_ex = problem_class(ic_numb=6, space_steps=60 * 2 * 2, time_steps=10*4*4, params=params)
    _, u_ex = train_model.compute_exact(Buckley_Leverett, problem_ex, 60, 10, just_one_time_step=False, trainable=False)
    V_init, nn = train_model.init_run_weno(problem_main, vectorized=True, just_one_time_step=False)
    V_train = V_init
    print(j)
    print(params)
    for k in range(nn):
        # Forward path
        #params = None
        #params = {'T': 0.4, 'e': 1e-13, 'L': 1, 'R': 1, 'C': 0.5}
        #my_problem = Buckley_Leverett(space_steps=200, time_steps=1,params=params)
        #problem_main = problem_class(space_steps=160, time_steps=None, params=params)
        V_train = train_model.forward(problem_main,V_train,k)
        # Train model:
        optimizer.zero_grad()  # Clear gradients
        # Calculate loss
        # params = problem_main.get_params()
        #loss = monotonicity_loss(V_train[:,1], problem_class, params, problem_main)  # Digital
        #loss = monotonicity_loss(V_train, problem_class, params=params)  # Buckley
        #loss = error_overflows_loss(V_train, u_exact[:,k])
        #loss = overflows_loss(V_train)
        loss = exact2_overflows_loss(V_train, u_ex[:,k+1])
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize weights
        #g = train_model.parameters()
        #x = g.__next__()
        #print(x.detach().numpy().sum(axis=0))
        print(k, loss.data.numpy())
        V_train.detach_()
        #print(params)


#plt.plot(S, V_train.detach().numpy())
#print("number of parameters:", sum(p.numel() for p in train_model.parameters()))
# g=train_model.parameters()
# g.__next__()

torch.save(train_model, "model18")


# model 16 trenovany na mean errore