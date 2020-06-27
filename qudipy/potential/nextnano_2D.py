import csv
import os

def importFolder(foldername):
    """
    input: a string, name of the folder where nextnano++ files are stored 
    output: a list, where each element is a list of voltages, potentials, and coordinates
    """
    for subdir, dirs, files in os.walk(foldername):
        print("2")
        for file in files:
            filename = os.path.join(subdir, file)
            with open(filename, newline='') as myFile:
                reader = csv.reader(myFile)
                for row in reader:
                    print(row)
    return 1

foldername = "Sliced_potentials"
importFolder(foldername)