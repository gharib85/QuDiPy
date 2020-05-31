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


    def hough_transform(self, angle_res):
        '''
        '''

        height = self.csd
        width = self.csd

        # Specify angle vector
        thetas = np.linspace(-np.pi/2, np.pi/2, 360/angle_res)

        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        len_t = len(thetas)

        accumulator = np.zeros()