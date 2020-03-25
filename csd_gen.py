import seaborn as sb
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt


# constants
e = -1.60217662 * 10**-19

# params
c_1 = 51 * 10**-18
c_2 = 50 * 10**-18
c_m = 5 * 10**-18

c_g1 = 1 * 10**-18
c_g2 = 1 * 10**-18

# Calculated constants
e_c1 = e**2 * c_1/ (c_1 * c_2 - c_m**2)
e_c2 = e**2 * c_2/ (c_1 * c_2 - c_m**2)
e_cm = e**2 * c_m/ (c_1 * c_2 - c_m**2)

def calculate_energy(n_1, n_2, v_g1, v_g2):
    '''Returns energy of dot with occupation n_1, n_2 with applied voltages v_g1, v_g2.
    Dependent on c_1, c_2, c_m, c_g1, c_g2, c_g1_d2 and c_g2_d1 defined as parameters outside the function
    '''
    return 1/2 * n_1**2 * e_c1 + 1/2 * n_2**2 * e_c2 + n_1 * n_2 * e_cm - 1/abs(e) * \
         ( c_g1 * v_g1 * (n_1 * e_c1 + n_2 * e_cm) + c_g2 * v_g2 * (n_1 * e_cm + n_2 * e_c2)) + 1/e**2 * \
         ( 1/2 * c_g1**2 * v_g1**2 * e_c1 + 1/2 * c_g2**2 * v_g2**2 * e_c2 + c_g1 * v_g1 * c_g2 * v_g2 * e_cm)

def lowest_energy(v_g1, v_g2):
    '''Returns occupation (n_1, n_2) with lowest energy for applied gate voltages v_g1, v_g2, with the approximation that c_m << c_1, c_2
    Dependent on c_1, c_2, c_m, c_g1, c_g2, c_g1_d2 and c_g2_d1 defined as parameters outside the function
    '''
    # get lowest energy assuming a continuous variable function

    n_1 = 1/(1 - e_cm ** 2/(e_c1 * e_c2)) * 1/abs(e) * (c_g1 * v_g1 * (1 - e_cm ** 2/(e_c1 * e_c2)) + c_g2 * v_g2 * (e_cm/e_c2 - e_cm/e_c1))
    n_2 = -n_1 * e_cm/e_c2 + 1/abs(e) * (c_g1 * v_g1 * e_cm/e_c2 + c_g2 * v_g2)

    # goes over 4 closest integer lattice points to find integer solution with lowest energy

    n_trials = [(math.floor(n_1), math.floor(n_2)), (math.floor(n_1) + 1, math.floor(n_2)), (math.floor(n_1), math.floor(n_2) + 1), (math.floor(n_1) + 1, math.floor(n_2) + 1)]
    n_energies = [calculate_energy(*occupation, v_g1, v_g2) for occupation in n_trials]
    state = n_trials[n_energies.index(min(n_energies))]
    if state[0] >= 0 and state[1] >= 0:
        return state
    if state[0] < 0 and state[1] < 0:
        return (0,0)
    elif state[0] < 0:
        return (0, state[1])
    else:
        return (state[0], 0)


def plot_csd(v_g1_max, v_g2_max, v_g1_min = 0, v_g2_min = 0, num = 100):
    data = [[round(v_g1_min + i/num * (v_g1_max - v_g1_min), 4), round(v_g2_min + j/num * (v_g2_max - v_g2_min), 4), lowest_energy(v_g1_min + i/num * (v_g1_max - v_g1_min), v_g2_min + j/num * (v_g2_max - v_g2_min))[0] - lowest_energy(v_g1_min + i/num * (v_g1_max - v_g1_min), v_g2_min + j/num * (v_g2_max - v_g2_min))[1]] for i in range(num) for j in range(num)]
    df = pd.DataFrame(data, columns=['V_g1', 'V_g2', 'Occupation'])
    df = df.pivot_table(index= 'V_g1', columns = 'V_g2', values = 'Occupation')
    p1 = sb.heatmap(df, cbar = False, xticklabels=int(num/5), yticklabels=int(num/5))
    p1.axes.invert_yaxis()
    plt.show()

plot_csd(0.3, 0.3, num=100)