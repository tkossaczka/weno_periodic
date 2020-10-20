import torch
import pandas as pd
import numpy as np
from define_WENO_Network import WENONetwork
from define_problem_transport_eq import transport_equation
from define_problem_Buckley_Leverett import Buckley_Leverett
from define_problem_Burgers_equation import Burgers_equation
import random
import os, sys, argparse

torch.set_default_dtype(torch.float64)

#problem = transport_equation
#problem = Buckley_Leverett
problem = Burgers_equation

train_model = WENONetwork()
parameters = []

def save_problem_and_solution(save_path, sample_id):
    print("{},".format(sample_id))
    ic_id = random.randint(1,3)
    problem_ex = problem(ic_numb=ic_id, space_steps=64 * 2 , time_steps=None, params=None)
    width = problem_ex.width
    height = problem_ex.height
    k = problem_ex.k
    #C = problem_ex.params["C"]
    u_exact, u_exact_64 = train_model.compute_exact(Burgers_equation, problem_ex, 64, 27, just_one_time_step=False, trainable=False)
    u_exact = u_exact.detach().numpy()
    u_exact_64 = u_exact_64.detach().numpy()

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    np.save(os.path.join(save_path, "u_exact_{}".format(sample_id)), u_exact)
    np.save(os.path.join(save_path, "u_exact64_{}".format(sample_id)), u_exact_64)

    # if not os.path.exists(os.path.join(save_path, "parameters.txt")):
    #     with open(os.path.join(save_path, "parameters.txt"), "a") as f:
    #         f.write("{},{},{},{}\n".format("sample_id","width","height","C"))
    # with open(os.path.join(save_path, "parameters.txt"), "a") as f:
    #     f.write("{},{},{},{}\n".format(sample_id, width, height, C))

    if not os.path.exists(os.path.join(save_path, "parameters.txt")):
        with open(os.path.join(save_path, "parameters.txt"), "a") as f:
            f.write("{},{},{}\n".format("sample_id","ic_id","k"))
    with open(os.path.join(save_path, "parameters.txt"), "a") as f:
        f.write("{},{},{}\n".format(sample_id, ic_id, k))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate exact solutions with given sample number for filename')
    parser.add_argument('save_path', default='', help='sample number for filename')
    parser.add_argument('sample_number', default='0', help='sample number for filename')

    args = parser.parse_args()
    save_problem_and_solution(args.save_path, args.sample_number)

    # usage example: seq 0 15 | xargs -i{} -P8 python compute_exact_solution.py C:\Users\Tatiana\Desktop\Research\Research_ML_WENO\WENO_general_periodic\Buckley_Leverett_Data {}
    # seq 0 15 | xargs -i{} -P8 python compute_exact_solution.py C:\Users\Tatiana\Desktop\Research\Research_ML_W ENO\Burgers_Equation_Test\Burgers_Equation_Data {}
