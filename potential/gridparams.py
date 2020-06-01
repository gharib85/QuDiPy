#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:08:07 2020

@author: simba
"""
import numpy as np

class GridParameters:
    '''
    
    Initialize the grid parameter class. Handles all things related to the 
    grid settings for the simulations.
    
    TODO: Add class variables to account for 1D and 2D data types (might be
    useful to have something like that for shuttling simulations later)
    
    '''
    
    def __init__(self, xx, yy, potential):
        '''
        
        Parameters
        ----------
        xx : array
            grid coordinates along x.
        yy : array
            grid coordinates along y.
        potential : meshgrid array
            potential values along x-y coordinates where 2DEG is formed.

        Returns
        -------
        None.

        '''
        self.xx = xx
        self.yy = yy
        
        self.XX, self.YY= meshgrid(xx, yy, sparse=False, indexing='xy');
        self.nx, self.ny = np.shape(self.XX);
        
        self.VV = potential;
        
        
    def convert_MG_to_NO(self, data_MG):
        '''
        
        Method for converting data from meshgrid to natural order format       

        Parameters
        ----------
        dataMG : meshgrid array
            data to convert to n.

        Returns
        -------
        None.

        '''
        
        
    def convert_NO_to_MG(self, data_NO):
        data_MG = np.reshape(data_NO,);
        # MG = np.reshape(vNO,[nx,ny]).';
        
        return data_MG
        
        
    def slice_potential(self, slice_coordinate, slice_axis)
        
        
        
        
        
        
        
        
        
        