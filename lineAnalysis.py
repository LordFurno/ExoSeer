import hapi
import os
import numpy as np
import pandas as pd
import PySimpleGUI as sg
from scipy.stats import bernoulli
import scipy.stats as stats

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
    weightedSum=sum(zScores[i]*weight[i]*100 for i in range(len(zScores)))#Multiply by 100, because deicmal times deicmal just gets smaller
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

            curHeader=r"C:\Users\Tristan\Downloads\ExoSeer" +f"\{names[molecID]}.header"
            newHeader=r"C:\Users\Tristan\Downloads\ExoSeer\Data\LineData" +f"\{names[molecID]}.header"
            os.rename(curFilePath,newFilePath)
            os.rename(curHeader,newHeader)
        except:
            pass
def dipFinder(filePath):#filePath is the exoplanet csv file. MOlecule is just the name of the molecule to find dips.
    df=pd.read_csv(filePath)
    wavelengths=df["CENTRALWAVELNG"]
    transitDepths=df["PL_TRANDEP"]
    derivative=np.gradient(transitDepths,wavelengths)
    # dipIndexes=np.where(derivative<0)[0]
    dipIndexes=np.where((transitDepths < np.roll(transitDepths, 1)) & (transitDepths < np.roll(transitDepths, -1)))[0]
    dipLocation=[wavelengths[i] for i in dipIndexes]#Gets what wavelength they are at
    dipValue=[transitDepths[i] for i in dipIndexes]#Gets the transit detph at that point

    return (dipLocation,dipValue)




def detectMolecule(molecule,dipLocation,dipValue):

    def likelihood(obs, exp, moleculePresent=True):
        differences = np.abs(np.array(exp) - np.array(obs))
        likelihoods = []
        for value in differences:
            # Define a weighting function based on the magnitude of differences
            weight = 1/(1 + value)  # Example weighting function
            if moleculePresent:
                likelihoods.append(weight*1.001)  # Adjust likelihood based on weight
            else:
                likelihoods.append(weight*0.94)  # Adjust likelihood based on weight
        return np.prod(likelihoods)
    
    def prior(p):#Prior probability
        return bernoulli.pmf(p, 0.01)#Bernoulli distrbution, 5% chance that molecule is present, 95% chance that it is absent.

    def calculatePosterior(obs,exp,moleculePresent=True):#calculating posterior
        #Calculate likelihood
        likelihood_value=likelihood(obs, exp, moleculePresent=moleculePresent)
        #Calculate prior probability
        prior_prob=prior(1 if moleculePresent else 0)
        # Compute posterior probability using Bayes' theorem
        posterior_prob = (likelihood_value * prior_prob) / (likelihood(obs, exp, moleculePresent=True) * prior(1) + likelihood(obs, exp, moleculePresent=False) * prior(0))
        return posterior_prob

    #HITRAN molecule id's
    ids={'H2O': 1, 'CO2': 2, 'O3': 3, 'N2O': 4, 'CO': 5, 'CH4': 6, 'O2': 7, 'NO': 8, 'SO2': 9, 'NO2': 10, 'NH3': 11, 'HNO3': 12, 'OH': 13, 'HF': 14, 'HCl': 15, 'HBr': 16, 'HI': 17, 'ClO': 18, 'OCS': 19, 'H2CO': 20, 'HOCl': 21, 'N2': 22, 'HCN': 23, 'CH3Cl': 24, 'H2O2': 25, 'C2H2': 26, 'C2H6': 27, 'PH3': 28, 'COF2': 29, 'SF6': 30, 'H2S': 31, 'HCOOH': 32, 'HO2': 33, 'O': 34, 'ClONO2': 35, 'NO+': 36, 'HOBr': 37, 'C2H4': 38, 'CH3OH': 39, 'CH3Br': 40, 'CH3CN': 41, 'CF4': 42, 'C4H3': 43, 'HC3N': 44, 'H2': 45, 'CS': 46, 'SO3': 47, 'C2N2': 48, 'COC12': 49, 'SO': 50, 'CH3F': 51, 'GeH4': 52, 'CS2': 53, 'CH3I': 54, 'NF3': 55}
    test=hapi.fetch(molecule,ids[molecule],1,1500,5000)
    nu,coef = hapi.absorptionCoefficient_Lorentz(SourceTables=molecule,HITRAN_units=False)
    nu,trans = hapi.transmittanceSpectrum(nu,coef)
    wavelengths=[]

    for value in nu:
        wavelengths.append(convertCMtoUM(value))

    os.remove(r"C:\Users\Tristan\Downloads\ExoSeer"+f"\{molecule}.data")
    os.remove(r"C:\Users\Tristan\Downloads\ExoSeer"+f"\{molecule}.header")
    observedWavelength=[]
    observedTransit=[]

    expectedWavelength=[]
    expectedTransit=[]


    for i,value in enumerate(dipValue):#Converting into transmittance
        dipValue[i]=1-(value/100)

    for index,value in enumerate(dipLocation):#Finds matching dips and lines, wavelengths must be close and transmittance must be close
        closestMatchingValue=min(wavelengths,key=lambda x:abs(x-value))
        # print(closestMatchingValue-value)
        wavelengthIndex=wavelengths.index(closestMatchingValue)

        if abs(closestMatchingValue-value)<=0.001:
            observedWavelength.append(value)
            observedTransit.append(dipValue[index])

            expectedWavelength.append(closestMatchingValue)
            expectedTransit.append(trans[wavelengthIndex])

    moleculeFilePath=r"C:\Users\Tristan\Downloads\ExoSeer\Data\LineData"+f"\{molecule}.data"
    moleculeData=pd.read_csv(moleculeFilePath,header=None,engine="python")#Sep is how the values are seperated in the data

    print("Observed and expected calculated")
    dipLocation=set(dipLocation)
    molecWavelength=[]
    molecIntensity=[]
    intensityZscore=[]#This will be used to calculate threshold to perform bayesian inference or not
    for row in range(len(moleculeData)):
        value=str(moleculeData.iloc[row][0])
        
        value=value.split(' ')
        value=list(filter(lambda x:x!="",value))

        molecWavelength.append(convertCMtoUM(float(value[1])))#Converts to micrometers
        molecIntensity.append(float(value[2]))
    intensityMean=np.mean(molecIntensity)
    standardDeviation=np.std(molecIntensity)
    for i,a in enumerate(molecWavelength):
        difference=abs(min(dipLocation,key=lambda x:abs(x-a)) - a)
        if difference<=0.001:#Need to find what is qualified as "lines up", it doesn't have to be exactly the same, just very close
            intensityZscore.append(getZScore(intensityMean,molecIntensity[i],standardDeviation))

    if len(intensityZscore)>0:
        averageZScore=np.mean(intensityZscore)

        if averageZScore>=0 or abs(averageZScore)<0.001:#Is or above average
            differences=np.array(expectedTransit)-np.array(observedTransit)#Differences
            
            muPrior=3 #Prior mean for the deviations (assuming no bias in deviation)
            sigmaPrior=1  #Prior standard deviation for the mean (can be adjusted)
            alphaPrior=1  #Prior shape parameter for the gamma distribution (precision)
            betaPrior=1   #Prior rate parameter for the gamma distribution (precision)

            differenceMean=np.mean(differences)
            differenceVariance=np.var(differences)

            n=len(differences)

            sigmaPrior2=sigmaPrior**2#Prior variance
            sigmaPost2 = 1 / (n / differenceVariance + 1 / sigmaPrior2)#Posterior variance
            mu_post = sigmaPost2 * (differenceMean / differenceVariance + muPrior / sigmaPrior2)#Posterior mean

            alphaPost = alphaPrior+n / 2#Posterior shape
            betaPost = betaPrior + 0.5 * np.sum((differences - differenceMean) ** 2)#Posterior rate

            posteriorMu = stats.norm(loc=mu_post, scale=np.sqrt(sigmaPost2))#Creates normal (gaussian) distrbution representing posterior distrbution of mean differnces
            PosteriorTau = stats.gamma(a=alphaPost, scale=1 / betaPost)#Creates a gamma distribution representing the posterior distribution of the precision

            threshold = 0  # Adjust the threshold according to context
            prob_present = posteriorMu.sf(threshold)  # Survival function (1 - CDF) of the posterior mu

            print(f"Probability that molecule is present: {prob_present*100}")

            return prob_present*100
            #Perform inference
        else:
            print(averageZScore)
    else:
        print("No overlay lines with dips. Molecule isn't present")




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
sg.LOOK_AND_FEEL_TABLE['Exoseer'] = {'BACKGROUND': '#1C1C1C', 
                                        'TEXT': '#A8D8EA', 
                                        'INPUT': '#5582A2',
                                        'TEXT_INPUT': '#FFFFFF', 
                                        'SCROLL': '#99CC99', 
                                        'BUTTON': ('#1C1C1C', '#BBD5DE'), 
                                        'PROGRESS': ('#D1826B', '#CC8019'), 
                                        'BORDER': 1, 'SLIDER_DEPTH': 0,  
'PROGRESS_DEPTH': 0, } 
sg.theme('Exoseer')

names=list(names.values())
layout = [[sg.Text("Choose a folder: ",font=('Inter Bold', 12)), sg.Input(key="-PATH-" ,change_submits=True), sg.FileBrowse(key="-FILE-")],[sg.Button("Submit")],
         [sg.Text('Molecule to detect: ',font=('Inter Bold', 12)), sg.Combo(names, font=('Inter Bold', 12),  expand_x=True, enable_events=True,  readonly=True, key='-MOLECULE-')],
         [sg.Button("Calculate")],
         [sg.Text("",key="-STATUS-")]
         
         
         ]
filePath=None
window = sg.Window('Window Title', layout,size=(700,300))
while True:
    event, values=window.read()
    if event==sg.WIN_CLOSED or event=='Cancel': #if user closes window or clicks cancel
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
            probability=detectMolecule(molecule,dipLocation,dipValue)
            print(f"The probability that {molecule} exsists in the atmosphere of this planet is: {probability}%")

            
        



 