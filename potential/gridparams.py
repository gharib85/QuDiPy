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
        self.xx = np.array(xx)
        self.yy = np.array(yy)
        
        self.XX, self.YY= np.meshgrid(xx, yy, sparse=False, indexing='xy');
        self.ny, self.nx = np.shape(self.XX);
        
        self.VV = np.array(potential);
        
        
    def convert_MG_to_NO(self, data_MG):
        '''
        
        Method for converting data from meshgrid to natural order format       

        Parameters
        ----------
        dataMG : 2D array in meshgrid format
            data to convert.

        Returns
        -------
        data_NO : 1D array in natrual order format
            converted data.

        '''
        
        data_MG = np.transpose(data_MG)
        data_NO = np.reshape(data_MG, (self.nx*self.ny, 1), order='F');

        return data_NO
        
        
    def convert_NO_to_MG(self, data_NO):
        '''
        
        Method for converting data from natural order to meshgrid format

        Parameters
        ----------
        data_NO : 1D array in natural order format
            data to convert.

        Returns
        -------
        data_MG : 2D array in meshgrid format
            converted data.

        '''
        
        data_MG = np.reshape(data_NO, (self.nx,self.ny), order='F');
        data_MG = np.transpose(data_MG)
        
        return data_MG
        
        
    def slice_potential(self, slice_coordinate, slice_axis):
        
        return
        
        
        
        
        
        
        
        
        
        