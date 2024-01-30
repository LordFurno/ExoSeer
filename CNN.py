'''
I have all the molecule data to train this model on. 
For testing, I need to make some more data that will be a mix of different molecules. I will test the results based on total.
The total transimittance is equal to the average of the rest of the molecules.
'''
import torch
from torch.utils.data import DataLoader,Dataset,random_split
import pandas as pd
import os
import numpy as np
from torchvision import transforms
def loadCSV(filePath):
    return pd.read_csv(filePath)

class moleculeDataset(Dataset):
    def __init__(self,folderPath,molecule,transform=None):
        self.folderPath=folderPath
        self.molecule=molecule
        self.csvFiles=[f for f in os.listdir(folderPath) if f.endswith('.csv')]
        self.transform= transform
    def __len__(self):
        return len(self.csvFiles)
    def __getitem__(self,index):
        csvFile=self.csvFiles[index]
        filePath=os.path.join(self.folderPath,csvFile)#Gets file path
        data=loadCSV(filePath)
        if self.transform:
            data=np.array(data)#Turns into np array
            data=self.transform(data)#Turns into tensor.
        return data,self.molecule


trainSize=int(0.8*len())
waterDataset=moleculeDataset(folderPath=r"C:\Users\Tristan\Downloads\ExoSeer\Data\Training\Water",molecule="Water",transform=transforms.ToTensor())
#I need to combine this with a dataset that doesn't contain this molecule. In this case water.
#Use the NotMolecule folders to get the data that isn't the molecule we are training for. Combine the actual data and this to train the seperate CNN's.
trainSize=int(0.8*len(waterDataset))
testSize=len(waterDataset)-trainSize
trainDataset,testDataset=random_split(waterDataset,[trainSize,testSize])

trainDataloader=DataLoader(trainDataset,batch_size=32,shuffle=True)
testDataloader=DataLoader(testDataset,batch_size=32,shuffle=True)

print(waterDataset)