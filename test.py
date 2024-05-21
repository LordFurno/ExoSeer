#Making sure data is different
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
combinations=[('N2',), ('O2',), ('CO2',), ('H2O',), ('N2', 'O2'), ('N2', 'CO2'), ('N2', 'H2O'), ('O2', 'CO2'), ('O2', 'H2O'), ('CO2', 'H2O'), ('N2', 'O2', 'CO2'), ('N2', 'O2', 'H2O'), ('N2', 'CO2', 'H2O'), ('O2', 'CO2', 'H2O'), ('N2', 'O2', 'CO2', 'H2O')]
test=[]
for i in range(10):
    index=random.randint(1,4080)
    print(index)
    #C:\Users\Tristan\Downloads\ExoSeer\Data\Training\CO2\CO2-1.csv
    #C:\Users\Tristan\Downloads\ExoSeer\Data\Training\O2
    filePath=r'C:\Users\Tristan\Downloads\ExoSeer\Data\Training\O2\O2' + f'-{index}.csv'
    data=pd.read_csv(filePath)
    wavelength=np.array(data.iloc[:, 0])
    total=np.array(data.iloc[:, 1])
    plt.plot(wavelength,total)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.rcParams.update({'font.size': 15})
    plt.xlabel("Wavelength [um]",fontsize=15)
    plt.ylabel("Transmittance",fontsize=15)
    plt.show(block=True)
