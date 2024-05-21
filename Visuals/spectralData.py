import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def dipFinder(filePath):#filePath is the exoplanet csv file. MOlecule is just the name of the molecule to find dips.
    df=pd.read_csv(filePath)
    wavelengths=df["CENTRALWAVELNG"]
    transitDepths=df["PL_TRANDEP"]
    dipIndexes=np.where((transitDepths < np.roll(transitDepths, 1)) & (transitDepths < np.roll(transitDepths, -1)))[0]
    dipLocation=[wavelengths[i] for i in dipIndexes]#Gets what wavelength they are at
    dipValue=[transitDepths[i] for i in dipIndexes]#Gets the transit detph at that point

    return (dipLocation,dipValue)
filePath=r"C:\Users\Tristan\Downloads\table_HAT-P-18-b-Fu-et-al.-2022.csv"
data=pd.read_csv(filePath)
wavelengths=data["CENTRALWAVELNG"]
transitDepths=data["PL_TRANDEP"]
dipLocation,depth=dipFinder(filePath)
plt.figure(0,figsize=(10,2.81))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title("HAT-P-18b Fu et al. 2022 Spectrum Preview",fontsize=15)
plt.ylabel("Transit Depth (%)",fontsize=15)
plt.xlabel("Central Wavelength (microns)",fontsize=15)
plt.rcParams.update({'font.size': 15})
# plt.scatter(dipLocation,depth,color="red")
plt.plot(wavelengths,transitDepths)
plt.savefig("HAT.png")
plt.show(block=True)