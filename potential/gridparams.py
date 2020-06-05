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
        
    '''
    
    def __init__(self, xx, yy=None, potential=None):
        '''
        
        Parameters
        ----------
        xx : array
            Grid coordinates along x with uniform spacing.
        yy : array
            Grid coordinates along y with uniform spacing.
        potential : array
            Potential values along x-y coordinates where 2DEG is formed. Must
            be in meshgrid format (y,x).

        Returns
        -------
        None.

        '''
        self.VV = np.array(potential);
        
        self.xx = np.array(xx)
        self.dx = xx[1] - xx[0]
        self.nx = len(xx)  
                
        # Check if y coordinates were inputted as an argument
        if yy is None:
            self.grid_type = '1D'
            
            # Check that coordinate data matches potential
            if self.nx != self.VV.shape[0]:
                raise ValueError("x coordinate grid points do not match number"\
                                " of potential x-coordinates.")
        # y coordinates were inputted
        else:         
            self.grid_type = '2D'
            self.yy = np.array(yy)
            self.dy = yy[1] - yy[0]
            self.XX, self.YY= np.meshgrid(xx, yy, sparse=False, indexing='xy')
            self.ny, self.nx = np.shape(self.XX)
            
            # Check that coordinate data matches potential
            if self.nx != self.VV.shape[1]:
                raise ValueError("x coordinate grid points do not match number"\
                                " of potential x-coordinates.")
            if self.nx != self.VV.shape[1]:
                raise ValueError("y coordinate grid points do not match number"\
                                " of potential y-coordinates.")
        
        
    def convert_MG_to_NO(self, data_MG):
        '''
        
        Converts data from meshgrid to natural order format       

        Parameters
        ----------
        dataMG : 2D array
            Data to convert in meshgrid format.

        Returns
        -------
        data_NO : 1D array
            Converted data in natural order format.

        '''
        
        data_MG = np.transpose(data_MG)
        data_NO = np.reshape(data_MG, (self.nx*self.ny, 1), order='F');

        return data_NO
        
        
    def convert_NO_to_MG(self, data_NO):
        '''
        
        Converts data from natural order to meshgrid format

        Parameters
        ----------
        data_NO : 1D array
            Data to convert in natural order format

        Returns
        -------
        data_MG : 2D array
            Converted data in meshgrid order format.

        '''
        
        data_MG = np.reshape(data_NO, (self.nx,self.ny), order='F');
        data_MG = np.transpose(data_MG)
        
        return data_MG
    
    def update_potential(self, new_pot):
        '''
        
        Assign a new potential to the class

        Parameters
        ----------
        new_pot : 2D array in meshgrid format
            New potential to assign.

        Returns
        -------
        None.

        '''
        
        self.VV = new_pot
        
        
    def slice_potential(self, slice_coordinate, slice_axis):
        
        pass
        
        
        
        
        
        
        
        
        
        