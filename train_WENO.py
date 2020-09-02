from define_WENO_Network import WENONetwork
import torch
from torch import optim
from define_problem_transport_eq import transport_equation

torch.set_default_dtype(torch.float64)

# TRAIN NETWORK
train_model = WENONetwork()

# DROP PROBLEM FOR TRAINING
#params = None
#problem_class = Buckley_Leverett
problem_class = transport_equation

def monotonicity_loss(u, u_ex): #, problem_class, params, problem_main):
    # _, exact = problem_main.exact(first_step=True)
    # exact = torch.Tensor(exact)
    # error = torch.max(torch.abs(u-exact))
    # monotonicity = torch.sum(torch.max(u[:-1]-u[1:], torch.Tensor([0.0])))
    # loss = monotonicity #+ error

    # peeks_left = torch.sum(torch.max(u_left[:-1]-u_left[1:], torch.Tensor([0.0])))
    # peeks_right = torch.sum(torch.abs(torch.min(u_right[:-1] - u_right[1:], torch.Tensor([0.0]))))

    # overflows = torch.sum(torch.abs(torch.min(u, torch.Tensor([0.0])) +
    #                                 (torch.max(u, torch.Tensor([1.0]))-torch.Tensor([1.0])))) # *(torch.max(x, torch.Tensor([1.0])) != 1)))
    # problem_ex = problem_class(space_steps=100*2*2*2*2*2*2*2, time_steps=50*4*4*4*4*4*4*4, params=params)
    # _, u_ex = train_model.compute_exact(problem_class, problem_ex, 100, 50,
    #                                                       just_one_time_step=True, trainable=False)
    u_max = torch.max(u_ex)
    u_min = torch.min(u_ex)
    overflows = torch.sum(torch.abs(torch.min(u, u_min)-u_min) + torch.max(u, u_max)-u_max )
    error = train_model.compute_error(u, u_ex)
    loss = error + overflows # peeks_left + peeks_right
    return loss


#optimizer = optim.SGD(train_model.parameters(), lr=0.1)
optimizer = optim.Adam(train_model.parameters())


problem_main = problem_class(space_steps=100, time_steps=None, params=None)
V_init, nn = train_model.init_run_weno(problem_main,vectorized=True,just_one_time_step=False)
u_exact =  problem_main.exact()
u_exact = torch.Tensor(u_exact)

for j in range(10):
    V_train = V_init
    print(j)
    for k in range(nn+1):
        # Forward path
        #params = None
        #params = {'T': 0.4, 'e': 1e-13, 'L': 1, 'R': 1, 'C': 0.5}
        #params = {'sigma': 0.3, 'rate': 0.1, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5}
        #my_problem = Digital_option(space_steps=160, time_steps=1,params=params)
        #my_problem = heat_equation(space_steps=160, time_steps=1, params=None)
        #my_problem = Buckley_Leverett(space_steps=200, time_steps=1,params=params)

        #problem_main = problem_class(space_steps=160, time_steps=None, params=params)
        #problem_main = problem_class(space_steps=100, time_steps=50, params=params)

        V_train = train_model.forward(problem_main,V_train,k)
        # Train model:
        optimizer.zero_grad()  # Clear gradients
        # Calculate loss
        params = problem_main.get_params()
        #loss = monotonicity_loss(V_train[:,1], problem_class, params, problem_main)  # Digital
        #loss = monotonicity_loss(V_train, problem_class, params=params)  # Buckley
        loss = monotonicity_loss(V_train, u_exact[:,k])
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

torch.save(train_model, "model3")