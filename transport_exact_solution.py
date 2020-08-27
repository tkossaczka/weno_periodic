import numpy as np

def transport_exact_solution(initial_cond, time, dx, transport_coef):
    roll_exact = transport_coef * time / dx
    int_roll_0 = int(np.floor(roll_exact))
    int_roll_1 = int(int_roll_0 + 1)
    weight_0 = int_roll_1 - roll_exact
    weight_1 = roll_exact - int_roll_0
    return np.roll(initial_cond, int_roll_0) * weight_0 + np.roll(initial_cond, int_roll_1) * weight_1

if __name__=="__main__":
    import matplotlib.pyplot as plt
    steps =50
    dx = 2*np.pi/(steps-1)
    x = np.linspace(0,2*np.pi-dx, steps)
    initial_cond = np.sin(x)
    solution=transport_exact_solution(initial_cond=initial_cond, time=5, dx=dx, transport_coef=1)
    plt.plot(np.stack([initial_cond, solution]).T)
