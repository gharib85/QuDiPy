'''
File used to generate and plot charge stability diagrams assuming known capacitances of system.
'''

import seaborn as sb
import pandas as pd
import math
import matplotlib.pyplot as plt

e = 1.602176634 * 10**-19  # TODO figure out relative imports for common constants


class Csd:

    def __init__(self, c_1, c_2, c_m, c_g1, c_g2):

        # Capacitances between dots and resevoirs
        self.c_1 = c_1
        self.c_2 = c_2

        # Capacitances between dots
        self.c_m = c_m

        # Gate-dot capacitances
        self.c_g1 = c_g1
        self.c_g2 = c_g2

        # Calculated constants
        # Dot charging energy
        self.e_c1 = e**2 * c_1 / (c_1 * c_2 - c_m**2)
        self.e_c2 = e**2 * c_2 / (c_1 * c_2 - c_m**2)

        # Electrostatic coupling energy
        self.e_cm = e**2 * c_m / (c_1 * c_2 - c_m**2)

    def calculate_energy(self, n_1, n_2, v_g1, v_g2):
        '''Returns energy of dot with occupation n_1, n_2 with applied voltages v_g1, v_g2.
        Dependent on c_1, c_2, c_m, c_g1, c_g2, c_g1_d2 and c_g2_d1 defined when object is initialized
        '''
        return 1/2 * n_1**2 * self.e_c1 + 1/2 * n_2**2 * self.e_c2 + n_1 * n_2 * self.e_cm - 1/abs(e) * (
            self.c_g1 * v_g1 * (n_1 * self.e_c1 + n_2 * self.e_cm) + self.c_g2 * v_g2 * (
                n_1 * self.e_cm + n_2 * self.e_c2)) + 1/e**2 * (1/2 * self.c_g1**2 * v_g1**2 * self.e_c1 + 1/2 * self.c_g2**2 *
                                                                v_g2**2 * self.e_c2 + self.c_g1 * v_g1 * self.c_g2 * v_g2 * self.e_cm)

    def lowest_energy(self, v_g1, v_g2):
        '''Returns occupation (n_1, n_2) with lowest energy for applied gate voltages v_g1, v_g2, with the
        approximation that c_m << c_1, c_2. Dependent on c_1, c_2, c_m, c_g1, c_g2, c_g1_d2 and c_g2_d1
        defined when object is initialized.
        '''
        # get lowest energy assuming a continuous variable function

        n_1 = 1/(1 - self.e_cm ** 2/(self.e_c1 * self.e_c2)) * 1/abs(e) * (self.c_g1 * v_g1 * (
             1 - self.e_cm ** 2 / (self.e_c1 * self.e_c2)) + self.c_g2 * v_g2 * (self.e_cm/self.e_c2 - self.e_cm/self.e_c1))
        n_2 = -n_1 * self.e_cm/self.e_c2 + 1 / \
            abs(e) * (self.c_g1 * v_g1 * self.e_cm/self.e_c2 + self.c_g2 * v_g2)

        # goes over 4 closest integer lattice points to find integer solution with lowest energy

        n_trials = [(math.floor(n_1), math.floor(n_2)), (math.floor(n_1) + 1, math.floor(n_2)),
                    (math.floor(n_1), math.floor(n_2) + 1), (math.floor(n_1) + 1, math.floor(n_2) + 1)]
        n_energies = [self.calculate_energy(
            *occupation, v_g1, v_g2) for occupation in n_trials]
        state = n_trials[n_energies.index(min(n_energies))]
        if state[0] >= 0 and state[1] >= 0:
            return state
        if state[0] < 0 and state[1] < 0:
            return (0, 0)
        elif state[0] < 0:
            return (0, state[1])
        else:
            return (state[0], 0)

    def generate_csd(self, v_g1_max, v_g2_max, c_cs_1, c_cs_2, v_g1_min=0, v_g2_min=0, num=100):
        ''' Generates the charge stability diagram between v_g1(2)_min and v_g1(2)_max with num by num data points in 2D
        '''
        data = [[round(v_g1_min + i/num * (v_g1_max - v_g1_min), 4), round(v_g2_min + j/num * (v_g2_max - v_g2_min), 4),
                (self.lowest_energy(v_g1_min + i/num * (v_g1_max - v_g1_min), v_g2_min + j/num * (
                 v_g2_max - v_g2_min))[0] * c_cs_1 + self.lowest_energy(v_g1_min + i/num * (v_g1_max - v_g1_min),
                 v_g2_min + j/num * (v_g2_max - v_g2_min))[1] * c_cs_2) * 10**9] for i in range(num) for j in range(num)
                ]
        df = pd.DataFrame(data, columns=['V_g1', 'V_g2', 'Current'])
        self.csd = df.pivot_table(
            index='V_g1', columns='V_g2', values='Current')

    def plot_csd(self, v_g1_max, v_g2_max, v_g1_min=0, v_g2_min=0, num=100):
        '''Plots current charge stability diagram stored in object. If there is no charge stability diagram, generates
        one based on given parameters
        '''
        if self.csd is None:
            self.generate_csd(v_g1_max, v_g2_max, v_g1_min, v_g2_min, num)

        p1 = sb.heatmap(self.csd, xticklabels=int(
            num/5), yticklabels=int(num/5), cbar_kws={'label': 'Current (arb.)'})
        p1.axes.invert_yaxis()
        plt.show()

        df_der_row = self.csd.diff(axis=0)
        df_der_col = self.csd.diff(axis=1)
        df_der = pd.concat([df_der_row, df_der_col]).max(level=0)
        # xticklabels=int(num/5), yticklabels=int(num/5)
        p2 = sb.heatmap(df_der, cbar=True, xticklabels=int(
            num/5), yticklabels=int(num/5))
        p2.axes.invert_yaxis()
        self.csd_der = df_der
        plt.show()
