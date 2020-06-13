'''
File used to analyze charge stability diagrams in order to produce capacitance matrices, virtual gating, lever arms, etc.
For a good review of the Hough transform, check https://alyssaq.github.io/2014/understanding-hough-transform/.
Hough transfrom code based off of https://github.com/alyssaq/hough_transform.
'''

import numpy as np
import matplotlib.pyplot as plt

class CSDAnalysis:
    '''
    Initialize the charge stability diagram analysis class which analyzes charge stability diagrams to extract parameters. 

    '''
    def __init__(self, csd):
        '''
         
        Parameters
        ----------
        csd : CSD object from csd_gen

        Returns
        -------
        None

        '''
        self.csd = csd

    def generate_bitmap(self, threshold):
        '''
        Transforms the charge stability diagram into a bitmap. Threshold determines whether bit is considered 'on' or 'off'

        Parameters
        ----------
        threshold: threshold which determines whether bit is considered 'on' or 'off' 

        Returns
        -------
        None

        '''
        self.csd_bitmap = self.csd.mask(abs(self.csd) > threshold, other=1).mask(abs(self.csd) <= threshold, other=0)


    def hough_transform(self, num_thetas=180, rho_num=100):
        '''
        Transforms the charge stability diagram into a bitmap. Threshold determines whether bit is considered 'on' or 'off'

        Parameters
        ----------
        None

        Keyword Arguments
        -----------------
        num_thetas: number of angle points to sweep over (default 180) 
        rho_num: number of distance numbers to start with. Number of distnce points in the end will be roughly 2^(3/2) times the original amount (default 100)

        Returns
        -------
        Accumulator: 2D array with counts of each theta and 
        thetas: values of theta for which accumulator swept over
        rhos: values of distance for which accumulator swept over

        '''
        # Get charge stability diagram bitmap
        img = self.csd_bitmap
        
        # Rho and Theta ranges
        thetas = np.deg2rad(np.linspace(-90.0, 90.0, num=num_thetas))
        width = img.columns[-1]
        height = img.index[-1]
        diag_len = np.sqrt(width ** 2 + height ** 2)   # max_dist

        index_height, index_width = img.shape
        index_diag_len = int(round(np.sqrt(index_width ** 2 + index_height ** 2)))
        rhos = np.linspace(-diag_len, diag_len, num=2*index_diag_len)

        # Cache some resuable values
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)  

        # Hough accumulator array of theta vs rho
        accumulator = np.zeros((2 * index_diag_len, num_thetas), dtype=np.uint64)
        y_idxs, x_idxs = np.nonzero(img.to_numpy())  # (row, col) indexes to edges

        # Vote in the hough accumulator
        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]

            for t_idx in range(num_thetas):
                # Calculate rho. diag_len is added for a positive index
                rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + index_diag_len)
                accumulator[rho, t_idx] += 1

        return accumulator, thetas, rhos
