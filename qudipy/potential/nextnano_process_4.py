import numpy as np
from scipy.interpolate import RegularGridInterpolator
import os
import re

########## Global Variables ##########

# number of gates
numberOfGates = 5

# gate voltages
V1 = [0.1]
V2 = [0.2]
V3 = [0.2]
V4 = [0.2, 0.22, 0.24, 0.25, 0.26]
V5 = [0.1]

########## Helper Functions ##########

def is_float(string):
    """ True if given string is float else False"""
    try:
        return float(string)
    except ValueError:
        return False

def is_int(string):
    """ True if given string is int else False"""
    try:
        return int(string)
    except ValueError:
        return False

def loadFile(filename):
    """
    returns a 3d array for potential.dat
            a tuple of 3 element, x, y, z for coord files
    """
    data = []
    x = []
    y = []
    z = []
    counter = 0
    with open(filename, 'r') as f:
        d = f.readlines()
        if filename[-4:] == '.dat':
            for i in d:
                k = i.rstrip().split(" ")
                data.append(float(k[0]))     
            data = np.array(data, dtype='O')
            return data
        else:
            for i in d:
                k = i.rstrip().split(" ")
                if is_float(i)==False:
                    # append number list if the element is an int but not float
                    try:
                        int(i)
                        if counter == 0:
                            x.append(float(k[0]))
                        elif counter == 1:
                            y.append(float(k[0]))
                        else:
                            z.append(float(k[0]))
                    # ValueError happens when it hits an empty line
                    except ValueError:
                        # print(i)
                        counter+=1
                # counter keeps track of which coord the data belong to
                elif counter == 0:
                    x.append(float(k[0]))
                elif counter == 1:
                    y.append(float(k[0]))
                else:
                    z.append(float(k[0]))
            x = np.array(x, dtype='O')
            y = np.array(y, dtype='O')
            z = np.array(z, dtype='O')
            return x, y, z

def slicePotential2D(potential, x, y, z, slice):
    """
    input:  1d potential list, 
            x, y ,z coordinates
            slice is the x-y plane we want
    output: a 2d array of the potentials in the x-y plane
    """
    index = np.where(z==slice)[0]
    N = len(x)
    M = len(y)
    Q = len(z)
    pot3DArray = np.reshape(potential,(N,M,Q))
    pot2DArray = pot3DArray[:, :, index]
    return pot2DArray

def slicePotential3D(potential, x, y, z):
    """
    input:  1d potential list, 
            x, y ,z coordinates
            slice is the x-y plane we want
    output: a 2d array of the potentials in the x-y plane
    """
    N = len(x)
    M = len(y)
    Q = len(z)
    pot3DArray = np.reshape(potential,(N,M,Q))
    return pot3DArray

def parseVoltage(filename):
    """
    input: a string, the filename 
           an int, number of gates
    output: a list of voltages of each gate
    """
    org = re.split("[_/]",filename)
    s = []
    delete = []
    for i in org:
        try:
            if float(i) < 100:
                s.append(float(i))
        except ValueError:
            delete.append(i)
    return s

def importFolder(foldername):
    """
    input: a string, name of the folder where nextnano++ files are stored 
    output: a list, where each element is a list of voltages, potentials, and coordinates
    """
    L = []                  # each element in L would be a list of voltages, potentials, and coordinates
    counter = 0             # track which subdirectory 
    for subdir, dirs, files in os.walk(folder):
        if subdir != folder and subdir[-7:] != '/output':
            counter += 1
            voltage = parseVoltage(subdir)
            L.append([voltage])
        for file in files:
            filename = os.path.join(subdir, file)
            # always first .dat then .coord
            if filename[-4:] == '.dat' or filename[-6:] == '.coord':
                L[counter-1].append(loadFile(filename))
    return L

def reshapePotential(potentialL, voltages, coord, slice):
    """
    input:  a list, where each element is a list of voltages, potentials, and coordinates
            a list of gate voltages
            a float indicating the x-y plane
    output: an n-dimensial potential file, where n = number of gates + 2
    """

    # loop through each combination of gate voltages
    for i in potentialL:
        # slice an x-y plane of the potentials
        print(i[1].shape)
        potential2D = slicePotential2D(i[1], i[2][0], i[2][1], i[2][2], slice)
        i[1] = potential2D
        # reverse the list of voltages for sorting purpose
        i[0].reverse()
    potentialL.sort()

    # stack up the potential arrays in the correct order
    potential_elmt = ()
    for i in range(len(potentialL)):
        potential_elmt = potential_elmt + (potentialL[i][1],) 
    potential_overall = np.stack(potential_elmt, axis = 0)
    # potential_overall = np.stack((potentialL[0][1], potentialL[1][1], potentialL[2][1],potentialL[3][1],potentialL[4][1]),axis=0)

    # get the shape of the potential based on the number of gates and the voltages of each gate
    shape = ()
    for v in voltages:
        if len(v) > 1:
            shape = shape + (len(v),)
    shape = shape+ (len(coord[0]), len(coord[1]))
    
    potential_reshaped = np.reshape(potential_overall,shape)
    return potential_reshaped


def interp(potential, voltages, coord):
    """
    inputs: 
        potential is a n-dimensional array of 
        voltages is a list of gate voltages
        coord is a
    output:
        interpolating function with inputs of gate voltages and coordinates
    """
    x = [float(xi) for xi in coord[0]]
    y = [float(yi) for yi in coord[1]]
    variables = ()
    for v in voltages:
        if len(v) > 1:
            variables = variables + (v,)
    variables = variables+ (x,y)
    interpolating_func = RegularGridInterpolator(variables, potential)
    return interpolating_func


########## Tests ##########

folder = 'nextnanoSims_Small'

potentialL = importFolder(folder)
voltages = [V1, V2, V3, V4, V5]
coord = potentialL[0][2]
potentialND = reshapePotential(potentialL, voltages, coord, -1)
print(potentialND.shape)
out = interp(potentialND, voltages, coord)
print(out([0.2,1.0,1.0]))