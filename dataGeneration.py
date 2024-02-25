import random
import numpy as np
import os
import math
import subprocess
import csv
import time
import itertools
start=time.time()
#Helper functions

def getSubsets(arr):
    return list(itertools.chain.from_iterable(itertools.combinations(arr,r) for r in range(len(arr)+1)))[1:]#To remove empty subset

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
def writeToCSV(data,outputFilePath,extraMolecules):#Converts numerical data into a csv file
    if extraMolecules!=[]:
        for i in range(len(extraMolecules)):
            temp=data[i]
            temp.append(extraMolecules)
            data[i]=temp
    with open(outputFilePath,"w",newline='') as f:
        writer=csv.writer(f)
        for row in data:
            writer.writerow(row)
def createParameterFileTesting(parameters,filePath):
    with open(r'C:\Users\Tristan\Downloads\ExoSeer\Data\template.txt', 'r') as original_file:
        # Read the contents of the original file
        template = original_file.read()
    diff=['<ATMOSPHERE-NGAS>','<ATMOSPHERE-GAS>','<ATMOSPHERE-ABUN>','<ATMOSPHERE-UNIT>','<ATMOSPHERE-TYPE>']
    #parameters is a dictionary with the parameters to be changed to what value
    #Go through text file and modify it
    with open(filePath,"r+") as f:
        f.write(template)
        lines=f.readlines()
    
    lines[42]='<ATMOSPHERE-NGAS>'+str(parameters['<ATMOSPHERE-NGAS>'])+"\n"
    lines[43]='<ATMOSPHERE-GAS>'+str(parameters['<ATMOSPHERE-GAS>'])+"\n"
    lines[45]='<ATMOSPHERE-ABUN>'+str(parameters['<ATMOSPHERE-ABUN>'])+"\n"
    lines[46]='<ATMOSPHERE-UNIT>'+str(parameters['<ATMOSPHERE-UNIT>'])+"\n"
    lines[44]='<ATMOSPHERE-TYPE>'+str(parameters['<ATMOSPHERE-TYPE>'])+"\n"
    with open(filePath,"w") as file:
        file.writelines(lines)
    with open(filePath, "r+") as f:
        
        
        content=f.read()
        for p in parameters:
            if p not in diff:
            #Issue is that <atmosphere-nGas> I have to change, which mean I have to change the nubmers too

                content=content.replace(p,p+str(parameters[p]))
        f.seek(0)
        f.write(content)
        f.truncate()
        f.close



molecules={"N2":"Nitrogen","O2":"Oxygen","CO2":"CarbonDioxide","H2O":"Water"}
stars=["M","G"]#Red dwarf and yellow dwarf 

densities=[i/100 for i in range(50,600,10)]
radius=[i for i in range(4,21)]#The ranges for the raidus, it will randomly pick between 4k and 21k

redDwarfTemp=list(range(2000,3501))
redDwarfRad=[i/100 for i in range(10,61)]

yelDwarfTemp=list(range(5300,6001))
yelDwarfRad=[i/100 for i in range(90,111)]

starDistances=[i for i in range(10,50,10)]#The range for the star distances

HITRANValues={"N2":"HIT[22]","O2":"HIT[7]","CO2":"HIT[2]","H2O":"HIT[1]"}#For the parameter atmosphere-type

trainingFilePath=r'C:\Users\Tristan\Downloads\ExoSeer\Data\Training'

#Just create folders for each molecule, svm will be one vs all, this will still let it allow to classify multiple molecules.
moleculeSubsets=getSubsets(molecules.keys())

#Creating the relevant folder for each molecule
#For the data, ignore the # and colouring marks, the csv reader is wrong. 


#Consider changnign molecule abundance to percentage and change it to 100%, this makes it so the data isn't a straight line.
for combination in moleculeSubsets:#16 iterations
    folderPath=r'C:\Users\Tristan\Downloads\ExoSeer\Data\Training'
    #How should I name the folder?
    folderPath+=f'\{"-".join(combination)}'
    os.makedirs(folderPath)
    counter=1
    for z in range(10):
        for star in stars:#2 iterations
            for j in starDistances:#5 iterations
                starDistanceRange=[a/100 for a in range(j,j+11)]
                for k in radius:#17 iterations
                    radiusRange=[b for b in range(k*1000,(k+1)*1000)]
                    chosenRadius=random.choice(radiusRange)
                    density=random.choice(densities)
                    starDistance=random.choice(starDistanceRange)
                    if star=="M":
                        starTemp=random.choice(redDwarfTemp)
                        starRad=random.choice(redDwarfRad)
                    else:
                        starTemp=random.choice(yelDwarfTemp)
                        starRad=random.choice(yelDwarfRad)
                    gravity=calculateGravity(chosenRadius,density)

                    #Create parameters, ready to be applied in a text file
                    parameters={'<OBJECT-DIAMETER>':chosenRadius*2,'<OBJECT-GRAVITY>':gravity,'<OBJECT-STAR-DISTANCE>':starDistance,'<OBJECT-STAR-TYPE>':star,'<OBJECT-STAR-TEMPERATURE>':starTemp,'<OBJECT-STAR-RADIUS>':starRad,'<ATMOSPHERE-NGAS>':len(combination),'<ATMOSPHERE-ABUN>':",".join(["1" for i in range(len(combination))]),'<ATMOSPHERE-UNIT>':",".join(["scl" for i in range(len(combination))])}
                    parameters['<ATMOSPHERE-GAS>']=",".join(combination)
                    hitranNames=[HITRANValues[a] for a in combination]
                    parameters['<ATMOSPHERE-TYPE>']=",".join(hitranNames)


                    parameterFolder=r'C:\Users\Tristan\Downloads\ExoSeer\Data\Parameters'
                    parameterFile=r'C:\Users\Tristan\Downloads\ExoSeer\Data\Parameters' + f'\{"-".join(combination)}-{counter}'+'.txt'
                    createParameterFile(parameters,parameterFile)#Assume that this works

                    curlCommand=f'curl -d key=8bd9208abbd2dd15f3dd -d type=trn -d whdr=y --data-urlencode file@"{parameterFile}" https://psg.gsfc.nasa.gov/api.php'
                    output=subprocess.check_output(curlCommand,shell=True,text=True)
                    with open(r'C:\Users\Tristan\Downloads\ExoSeer\Data\temp.txt','w') as dataFile:
                        dataFile.write(output)
                        extracted=getNumericalData(r'C:\Users\Tristan\Downloads\ExoSeer\Data\temp.txt')
                        writeToCSV(extracted,folderPath+f'\{"-".join(combination)}-{counter}'+'.csv',[])
                        dataFile.close()
                    
                    #First column of data is wavelength in um. (x axis)
                    #6th column of data is contrast (y axis)
                    counter+=1