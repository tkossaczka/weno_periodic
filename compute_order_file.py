import torch
from define_WENO_Network import WENONetwork
from define_problem_transport_eq import transport_equation

with torch.no_grad():
    train_model = torch.load('model')
    problem = transport_equation
    #problem = heat_equation
    #problem = Call_option
    #problem = Digital_option
    #problem = PME
    #problem = Call_option_GS
    #problem = Digital_option_GS
    #problem = Buckley_Leverett
    torch.set_default_dtype(torch.float64)

    params = None
    #params = {'T': 0.4, 'e': 1e-13, 'L': 1, 'R': 1, 'C': 0.25}
    #params = {'T': 0.2, 'e': 1e-13, 'L': 1, 'R': 1, 'C': 0.3}
    #params = {'sigma': 0.3, 'rate': 0.02, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -6, 'xr': 1.5}
    #params = {'sigma': 0.3, 'rate': 0.25, 'E': 50, 'T': 1, 'e': 1e-13, 'xl': -1.5, 'xr': 2, 'psi': 30}
    # params = {'T': 1, 'e': 1e-13, 'L': 3.141592653589793}
    #err_trained, order_trained = train_model.order_compute(6, 10, None,  params, problem, trainable=True)
    err_not_trained, order_not_trained = train_model.order_compute(3,10,None, params, problem, trainable=False)