import torch
from define_WENO_Network import WENONetwork
from define_problem_transport_eq import transport_equation
from define_problem_Buckley_Leverett import Buckley_Leverett

with torch.no_grad():
    train_model = torch.load('model3')
    problem = transport_equation
    #problem = Buckley_Leverett
    torch.set_default_dtype(torch.float64)

    params = None
    #params = {'T': 0.4, 'e': 1e-13, 'L': 0, 'R': 2, 'C': 0.25}
    #params = {'T': 0.2, 'e': 1e-13, 'L': 1, 'R': 1, 'C': 0.3}
    #params = {'sigma': 0.3, 'rate': 0.02, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5}
    #params = {'sigma': 0.3, 'rate': 0.25, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -1.5, 'xr': 2, 'psi': 30}
    # params = {'T': 1, 'e': 1e-13, 'L': 3.141592653589793}
    #err_trained, order_trained = train_model.order_compute(5, 10, None,  params, problem, trainable=True, ic_numb=6)
    err_not_trained, order_not_trained = train_model.order_compute(4,10,None, params, problem, trainable=False, ic_numb=2)