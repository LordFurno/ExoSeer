import hapi
import os
import numpy as np
import pandas as pd
import PySimpleGUI as sg
def convertCMtoUM(cm):#Convrets cm^-1 to microns
   return 10000/cm 
def calculatePercentDiff(val1,val2):
    numerator=abs(val1-val2)
    denominator= (val1+val2)/2
    return numerator/denominator
def getZScore(average,dataPoint,standardDeviation):#Gets the z-score for a data point
    return (dataPoint-average)/standardDeviation
def normalize(arr):#Nomralizes a list
    return [i/max(arr) for i in arr]
def getWeights(arr,molecIntensity):#Gets the weight for each line based on their intensity compared to the rest
    maxIntensity=max(molecIntensity)
    weights=[(molecIntensity[i]/maxIntensity) for i in arr]
    return weights
def calculateWeightedAverage(weight,zScores):
    weightedSum=sum(zScores[i]*weight[i]*10 for i in range(len(zScores)))#Multiply by 100, because deicmal times deicmal just gets smaller
    totalWeight=sum(weight)
    return weightedSum/totalWeight

def getMoleculeData():
    names={
        1:"H2O",
        2:"CO2",
        3:"O3",
        4:"N2O",
        5:"CO",	
        6:"CH4",
        7:"O2",	
        8:"NO",	
        9:"SO2",	
        10:	"NO2",	
        11:	"NH3",	
        12:	"HNO3",	
        13:	"OH",
        14:	"HF",
        15:	"HCl",
        16:	"HBr",
        17:	"HI",
        18:	"ClO",
        19:	"OCS",
        20:	"H2CO",
        21:	"HOCl",
        22:	"N2",
        23:	"HCN",
        24:	"CH3Cl",
        25:	"H2O2",
        26:	"C2H2",
        27:	"C2H6",
        28:	"PH3",
        29:	"COF2",
        30:	"SF6",
        31:	"H2S",
        32:	"HCOOH",
        33:	"HO2",
        34:	"O",
        35:	"ClONO2",
        36:	"NO+",
        37:	"HOBr",
        38:	"C2H4",
        39:	"CH3OH",
        40:	"CH3Br",
        41:	"CH3CN",
        42:	"CF4",
        43:	"C4H3",
        44:	"HC3N",
        45:	"H2",
        46:	"CS",
        47:	"SO3",
        48:	"C2N2",
        49:	"COC12",
        50:	"SO",
        51:	"CH3F",
        52:	"GeH4",
        53:	"CS2",
        54:	"CH3I",
        55:	"NF3",
    }  
    for molecID in range(1,56):
        try:#If wavelengths doesn't fit
            hapi.fetch(names[molecID],molecID,1,1500,5000)#Just use the first isotopolouge, the most most abundant
            curFilePath=r"C:\Users\Tristan\Downloads\ExoSeer" +f"\{names[molecID]}.data"
            newFilePath=r"C:\Users\Tristan\Downloads\ExoSeer\Data\LineData"+f"\{names[molecID]}.data"

            headerFilePath=r"C:\Users\Tristan\Downloads\ExoSeer" +f"\{names[molecID]}.header"
            os.rename(curFilePath,newFilePath)
            os.remove(headerFilePath)
        except:
            pass
def dipFinder(filePath):#filePath is the exoplanet csv file. MOlecule is just the name of the molecule to find dips.


    df=pd.read_csv(filePath)
    wavelengths=df["CENTRALWAVELNG"]
    transitDepths=df["PL_TRANDEP"]
    derivative=np.gradient(transitDepths,wavelengths)
    dipIndexes=np.where(derivative<0)[0]
    dipLocation=[wavelengths[i] for i in dipIndexes]#Gets what wavelength they are at
    dipValue=[transitDepths[i] for i in dipIndexes]#Gets the transit detph at that point

    return (dipLocation,dipValue)

def detectMolecule(molecule,dipLocation):
    moleculeFilePath=r"C:\Users\Tristan\Downloads\ExoSeer\Data\LineData"+f"\{molecule}.data"
    moleculeData=pd.read_csv(moleculeFilePath,header=None,engine="python")#Sep is how the values are seperated in the data

    #print(moleculeData)
    # print(moleculeData.iloc[:,0])
    dipLocation=set(dipLocation)
    molecWavelength=[]
    molecIntensity=[]
    for row in range(len(moleculeData)):
        value=str(moleculeData.iloc[row][0])
        
        value=value.split(' ')
        value=list(filter(lambda x:x!="",value))

        molecWavelength.append(convertCMtoUM(float(value[1])))#Converts to micrometers
        molecIntensity.append(float(value[2]))

    intensityMean=np.mean(molecIntensity)
    intensityZscore=[]
    standardDeviation=np.std(molecIntensity)
    indexes=[]
    for i,a in enumerate(molecWavelength):
        print((i,len(molecWavelength)))
        difference=abs(min(dipLocation,key=lambda x:abs(x-a)) - a)

        
        if difference<=0.001:#Need to find what is qualified as "lines up", it doesn't have to be exactly the same, just very close
            intensityZscore.append(getZScore(intensityMean,molecIntensity[i],standardDeviation))
            indexes.append(i)
            # scores.append(score)
    if len(intensityZscore)>0:
        
        averageZScore=np.mean(intensityZscore)
        # print(averageZScore)


        if averageZScore>0 :#Is or above average
            print(averageZScore)
            normalizedZ=normalize(intensityZscore)#Normalized all z scores
            weights=getWeights(indexes,molecIntensity)#Calculates the weights for the line based on intensity
            weightAverage=calculateWeightedAverage(weights,normalizedZ)#Gets the weighted average with the new weights


            percentMatch=len(intensityZscore)/len(molecIntensity)*100#Gets what percent of the lines match
            #Maybe don't add them, should look into changing
            return max(100,weightAverage+percentMatch)
        else:
            print(averageZScore)
            

        #Likeley that molecule exsists


    else:
        return 0#No overlay lines, so molecule doesn't exsist





# location,values=dipFinder(r"C:\Users\Tristan\Downloads\ExoSeer\ExoplanetDataTest.csv")
# print("dips found")
# print(detectMolecule("CO2",location))
# print(numberOfLines)

names={#
        1:"H2O",
        2:"CO2",
        3:"O3",
        4:"N2O",
        5:"CO",	
        6:"CH4",
        7:"O2",	
        8:"NO",	
        9:"SO2",	
        10:	"NO2",	
        11:	"NH3",	
        12:	"HNO3",	
        13:	"OH",
        14:	"HF",
        15:	"HCl",
        16:	"HBr",
        17:	"HI",
        18:	"ClO",
        19:	"OCS",
        20:	"H2CO",
        21:	"HOCl",
        22:	"N2",
        23:	"HCN",
        24:	"CH3Cl",
        25:	"H2O2",
        26:	"C2H2",
        27:	"C2H6",
        28:	"PH3",
        29:	"COF2",
        30:	"SF6",
        31:	"H2S",
        32:	"HCOOH",
        33:	"HO2",
        34:	"O",
        35:	"ClONO2",
        36:	"NO+",
        37:	"HOBr",
        38:	"C2H4",
        39:	"CH3OH",
        40:	"CH3Br",
        41:	"CH3CN",
        42:	"CF4",
        43:	"C4H3",
        44:	"HC3N",
        45:	"H2",
        46:	"CS",
        47:	"SO3",
        48:	"C2N2",
        49:	"COC12",
        50:	"SO",
        51:	"CH3F",
        52:	"GeH4",
        53:	"CS2",
        54:	"CH3I",
        55:	"NF3",
    }  

sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.
names=list(names.values())
layout = [[sg.Text("Choose a folder: "), sg.Input(key="-PATH-" ,change_submits=True), sg.FileBrowse(key="-FILE-")],[sg.Button("Submit")],
         [sg.Text('Molecule to detect: '), sg.Combo(names, font=('Arial Bold', 12),  expand_x=True, enable_events=True,  readonly=True, key='-MOLECULE-')],
         [sg.Button("Calculate")],
         [sg.Text("",key="-STATUS-")]
         
         
         ]
filePath=None
window = sg.Window('Window Title', layout,size=(700,300))
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    if event=="Submit":
        filePath=values["-PATH-"]
        if ".csv" not in filePath:
            window["-STATUS-"].update("Please select a .csv file")
        else:
            dipLocation,dipValue=dipFinder(filePath)

    if event=="Calculate":
        if filePath!=None and values["-MOLECULE-"]!="":
            molecule=values["-MOLECULE-"]
            probability=detectMolecule(molecule,dipLocation)
            print(f"The probability that {molecule} exsists in the atmosphere of this planet is: {probability}")

            
        



 