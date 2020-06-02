'''
File used to analyze charge stability diagrams in order to produce capacitance matrices, virtual gating, lever arms, etc.
'''

import numpy as np
import matplotlib.pyplot as plt

class CSDAnalysis:


    def __init__(self, csd):
        '''Initialize
        '''
        self.csd = csd
        self.csd_bitmap = None

    def generate_bitmap(self, threshold):
        '''
        Transforms the charge stability diagram into a bitmap. Threshold determines whether bit is considered 'on' or 'off'
        '''
        self.csd_bitmap = self.csd.mask(self.csd != 0, other=1)


    def hough_transform(self, angle_res=1):
        '''
        '''
        img = self.csd_bitmap
        # Rho and Theta ranges
        thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_res))
        width, height = img.shape
        diag_len = np.ceil(np.sqrt(width ** 2 + height ** 2))   # max_dist
        rhos = np.linspace(-diag_len, diag_len, int(diag_len * 2.0))

        # Cache some resuable values
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        num_thetas = len(thetas)

        # Hough accumulator array of theta vs rho
        accumulator = np.zeros((int(2 * diag_len), num_thetas), dtype=np.uint64)
        y_idxs, x_idxs = np.nonzero(img.to_numpy())  # (row, col) indexes to edges

        # Vote in the hough accumulator
        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]

            for t_idx in range(num_thetas):
                # Calculate rho. diag_len is added for a positive index
                rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len)
                accumulator[rho, t_idx] += 1

        return accumulator, thetas, rhos
