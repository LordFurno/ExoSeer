import random
import numpy as np
import os
import math
import subprocess
import csv
import time
start=time.time()
#Helper functions
def calculateGravity(radius,density):#Radius is in km and density is in g/cm^3
    gravConstant=6.6743*(10**-11)
    volume=(radius**3)*(4/3)*math.pi * 1000000000#m^3
    density*=1000#To convert g/cm^3 into kg/m^3
    mass=density*volume#In kg
    gravity= (gravConstant*mass)/(radius**2) / 1000000#To get it into right units
    return gravity
def createParameterFile(parameters,filePath):
    with open(r'C:\Users\Tristan\Downloads\ExoSeer\Data\template.txt', 'r') as original_file:
        # Read the contents of the original file
        template = original_file.read()

    #parameters is a dictionary with the parameters to be changed to what value
    #Go through text file and modify it
    with open(filePath,"w") as f:
        f.write(template)
    with open(filePath, "r+") as f:
        content=f.read()
        for p in parameters:
            content=content.replace(p,p+str(parameters[p]))
        f.seek(0)
        f.write(content)
        f.truncate()
        f.close
def getNumericalData(filePath):#Get numerical data from text file
    with open(filePath,'r') as f:
        lines=f.readlines()
    numericalData=[]
    for line in lines:
        if not line.startswith("#") or line.startswith("# Wave/freq"):
            numericalValue=line.split()
            numericalData.append(numericalValue)
    return numericalData
def writeToCSV(data,outputFilePath):#Converts numerical data into a csv file
    with open(outputFilePath,"w",newline='') as f:
        writer=csv.writer(f)
        for row in data:
            writer.writerow(row)




molecules={"N2":"Nitrogen","O2":"Oxygen","CO2":"CarbonDioxide","He":"Helium","CH4":"Methane","H2":"Hydrogen","H2O":"Water"}
stars=["M","G"]#Red dwarf and yellow dwarf 

densities=[i/100 for i in range(50,600,10)]
radius=[i for i in range(4,21)]#The ranges for the raidus, it will randomly pick between 4k and 21k

redDwarfTemp=list(range(2000,3501))
redDwarfRad=[i/100 for i in range(10,61)]

yelDwarfTemp=list(range(5300,6001))
yelDwarfRad=[i/100 for i in range(90,111)]

starDistances=[i for i in range(10,50,10)]#The range for the star distances

HITRANValues={"N2":"HIT[22]","O2":"HIT[7]","CO2":"HIT[2]","He":"HIT[0]0","CH4":"HIT[6]","H2":"HIT[45]","H2O":"HIT[1]"}#For the parameter atmosphere-type

trainingFilePath=r'C:\Users\Tristan\Downloads\ExoSeer\Data\Training'

#Just create folders for each molecule, svm will be one vs all, this will still let it allow to classify multiple molecules.

#Creating the relevant folder for each molecule
for molecule in molecules.values():
    newPath=r'C:\Users\Tristan\Downloads\ExoSeer\Data\Training'
    newPath+=f'\{molecule}'
    os.makedirs(newPath)
#Consider changnign molecule abundance to percentage and change it to 100%, this makes it so the data isn't a straight line.

for molecule in molecules:
    counter=1
    folderPath=trainingFilePath+f'{molecules[molecule]}'
    for i in range(10):#1700 files for each molecule
        for star in stars:#2 iterations
            for j in starDistances:#5 iterations
                starDistanceRange=[a/100 for a in range(j,j+11)]
                for k in radius:#17 iterations
                    
                    radiusRange=[b for b in range(k*1000,(k+1)*1000)]
                    rad=random.choice(radiusRange)
                    density=random.choice(densities)
                    starDist=random.choice(starDistanceRange)
                    if star=="M":
                        starTemp=random.choice(redDwarfTemp)
                        starRad=random.choice(redDwarfRad)
                    else:
                        starTemp=random.choice(yelDwarfTemp)
                        starRad=random.choice(yelDwarfRad)
                    gravity=calculateGravity(rad,density)

                    starDistance=random.choice(starDistanceRange)
                    parameters={'<OBJECT-DIAMETER>':rad*2,'<OBJECT-GRAVITY>':gravity,'<OBJECT-STAR-DISTANCE>':starDist,'<OBJECT-STAR-TYPE>':star,'<OBJECT-STAR-TEMPERATURE>':starTemp,'<OBJECT-STAR-RADIUS>':starRad,'<ATMOSPHERE-GAS>':molecule,'<ATMOSPHERE-TYPE>':HITRANValues[molecule]} 
                    parameterFolder=r'C:\Users\Tristan\Downloads\ExoSeer\Data\Parameters'
                    dataFolder=r'C:\Users\Tristan\Downloads\ExoSeer\Data\Training'+f'\{molecules[molecule]}'
                    parameterFile=r'C:\Users\Tristan\Downloads\ExoSeer\Data\Parameters' + f'\{molecules[molecule]}{counter}'+'.txt'
                    createParameterFile(parameters,parameterFile)
                    #Upload to PSG API recieve data, add to training data folder
                    curlCommand=f'curl -d key=API_KEY -d type=trn -d whdr=y --data-urlencode file@"{parameterFile}" https://psg.gsfc.nasa.gov/api.php'
                    output=subprocess.check_output(curlCommand,shell=True,text=True)
                    
                    with open(r'C:\Users\Tristan\Downloads\ExoSeer\Data\temp.txt','w') as dataFile:
                        dataFile.write(output)
                        extracted=getNumericalData(r'C:\Users\Tristan\Downloads\ExoSeer\Data\temp.txt')
                        writeToCSV(extracted,dataFolder+f'\{molecules[molecule]}{counter}'+'.csv')
                        dataFile.close()
                    
                    #First column of data is wavelength in um. (x axis)
                    #6th column of data is contrast (y axis)
                    counter+=1

print("Training data generated")
counter=1
for i in range(2380):
    molecule=random.choice(list(molecules.keys()))
    star=random.choice(stars)
    starDist=random.choice(starDistances)
    density=random.choice(densities)
    rad=random.randrange(4000,20000)
    if star=="M":
        starTemp=random.choice(redDwarfTemp)
        starRad=random.choice(redDwarfRad)
    else:
        starTemp=random.choice(yelDwarfTemp)
        starRad=random.choice(yelDwarfRad)
    gravity=calculateGravity(rad,density)
    parameters={'<OBJECT-DIAMETER>':rad*2,'<OBJECT-GRAVITY>':gravity,'<OBJECT-STAR-DISTANCE>':starDist,'<OBJECT-STAR-TYPE>':star,'<OBJECT-STAR-TEMPERATURE>':starTemp,'<OBJECT-STAR-RADIUS>':starRad,'<ATMOSPHERE-GAS>':molecule,'<ATMOSPHERE-TYPE>':HITRANValues[molecule]} 
    dataFolder=r'C:\Users\Tristan\Downloads\ExoSeer\Data\Testing'
    parameterFile=r'C:\Users\Tristan\Downloads\ExoSeer\Data\Testing\param.txt'
    createParameterFile(parameters,parameterFile)
    #Upload to PSG API recieve data, add to training data folder
    curlCommand=f'curl -d key=API_KEY -d type=trn -d whdr=y --data-urlencode file@"{parameterFile}" https://psg.gsfc.nasa.gov/api.php'
    output=subprocess.check_output(curlCommand,shell=True,text=True)
    
    with open(r'C:\Users\Tristan\Downloads\ExoSeer\Data\temp.txt','w') as dataFile:
        dataFile.write(output)
        extracted=getNumericalData(r'C:\Users\Tristan\Downloads\ExoSeer\Data\temp.txt')
        writeToCSV(extracted,dataFolder+f'\{molecules[molecule]}{counter}'+'.csv')
        dataFile.close()
    
    #First column of data is wavelength in um. (x axis)
    #6th column of data is contrast (y axis)
    counter+=1
end=time.time()
print(end-start)
