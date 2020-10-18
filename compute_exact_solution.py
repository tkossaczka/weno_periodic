import torch
import pandas as pd
import numpy as np
from define_WENO_Network import WENONetwork
from define_problem_transport_eq import transport_equation
from define_problem_Buckley_Leverett import Buckley_Leverett
from define_problem_Burgers_equation import Burgers_equation
import random

torch.set_default_dtype(torch.float64)

#problem = transport_equation
problem = Buckley_Leverett
#problem = Burgers_equation

train_model = WENONetwork()

parameters = []

for k in range(5):
    print(k)
    problem_ex = problem(ic_numb=0, space_steps=60 * 2  , time_steps=None, params=None)
    numb = problem_ex.numb
    width = problem_ex.width
    height = problem_ex.height
    xmid = problem_ex.xmid
    C = problem_ex.params["C"]
    u_exact, u_exact_60 = train_model.compute_exact(Buckley_Leverett, problem_ex, 60, 36, just_one_time_step=False, trainable=False)
    u_exact = u_exact.detach().numpy()
    u_exact_60 = u_exact_60.detach().numpy()
    np.save("Exact_Solutions_1/u_exact_{}_{}".format(k, str(random.random())),u_exact)
    np.save("Exact_Solutions_1/u_exact_60_{}_{}".format(k, str(random.random())), u_exact_60)
    parameters.append([width, height, C])

#pd.DataFrame(parameters)

df=pd.DataFrame(parameters, columns=["width", "height", "C"])

df.to_csv("Exact_Solutions_1/parameters.csv",index=False)
# df2=pd.read_csv("Exact_Solutions/parameters.csv")

# a=[[1,2],[3,4],[5,5]]
# pd.DataFrame(a)
# df=pd.DataFrame(a)
# df=pd.DataFrame(a, columns=["height", "width"])
# df.to_csv("lol.csv",index=False)
# df2=pd.read_csv("lol.csv")
# df2.loc[2]
# df2.loc[2,"height"]

