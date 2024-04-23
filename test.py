import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def dipFinder(filePath):#filePath is the exoplanet csv file. MOlecule is just the name of the molecule to find dips.
    df=pd.read_csv(filePath)
    wavelengths=df["CENTRALWAVELNG"]
    transitDepths=df["PL_TRANDEP"]
    derivative=np.gradient(transitDepths,wavelengths)
    dipIndexes=np.where(derivative<0)[0]
    dipLocation=[wavelengths[i] for i in dipIndexes]#Gets what wavelength they are at
    dipValue=[transitDepths[i] for i in dipIndexes]#Gets the transit detph at that point
    return (dipLocation,dipValue)
filePath=r"C:\Users\Tristan\Downloads\table_K2-18-b-Madhusudhan-et-al.-2023.csv"
data=pd.read_csv(filePath)

wavelengths=data["CENTRALWAVELNG"]
transitDepths=data["PL_TRANDEP"]



data=pd.read_csv(filePath)
plt.figure(figsize=(15, 15))
# plt.title("K2-18 b Madhusudhan et al. 2023 Spectrum")
plt.plot(wavelengths,transitDepths)
plt.ylabel("Transit Depth (%)")
plt.xlabel("Central Wavelength (microns)")
# plt.scatter(dipLocation,dipValue,color="red")
plt.show(block=True)
# plt.plot(locations,"r")
