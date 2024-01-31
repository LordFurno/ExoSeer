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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
def loadCSV(filePath):
    return pd.read_csv(filePath)
def removeNan(inputTensor):
    nanMask=torch.isnan(inputTensor)
    newTensor=inputTensor[~nanMask]
    return newTensor
def meanFilter(tensor,window_size):
    cumsum=torch.cumsum(tensor, dim=0)
    cumsum[window_size:]=cumsum[window_size:]-cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size



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
        wavelength=data.iloc[:,1]#Wavelength column
        molecData=data.iloc[:,3]#Transmittance column



        if self.transform:
            wavelength,molecData=np.array(wavelength),np.array(molecData)
            wavelength,molecData=wavelength.reshape(1,-1),molecData.reshape(1,-1)
            wavelength=self.transform(wavelength)
            molecData=self.transform(molecData)
            data=torch.stack([wavelength, molecData], dim=1)#Puts wavelength and molec data together in 1 tensor. 
        return data,self.molecule#Wavelngth, molecule,label
class combineDataset(Dataset):
    def __init__(self,dataset1,dataset2):
        self.dataset1=dataset1
        self.dataset2=dataset2
        self.totaLength=len(dataset1)+len(dataset2)

    def __len__(self):
        return self.totaLength
    def __getitem__(self,index):
        if index<len(self.dataset1):
            return self.dataset1.__getitem__(index)
        else:
            newIndex=index-len(self.dataset2)
            return self.dataset2.__getitem__(newIndex)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1Molecule=nn.Conv2d(2,64,kernel_size=3,padding=1)
        self.conv1Mean=nn.Conv2d(2,64,kernel_size=3,padding=1)
        self.conv1Downsample=nn.Conv2d(2,64,kernel_size=3,padding=1)
        
        self.pool=nn.MaxPool2d(kernel_size=(1,2),stride=(1,2))
        
        self.conv2Combined=nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)#Input is 78, because this is what the 32+30+16 is.

        self.poolCombined=nn.MaxPool2d(kernel_size=1,stride=2)

        self.fc1=nn.Linear(16*1911,32)#16*1911, since that is the shape of my data
        self.fc2=nn.Linear(32,2)#Output layer is 2 because it is either present or not
    def forward(self,x,meanX,downX):
        #This entire function is so screwed up. So many nan values. Not sure how to fix it
        #Pass the values through convolution and pooling layers
        #Really need to fix up this functino, this is what is breaking my code

        x=x.float()
        meanX=meanX.float()
        downX=downX.float()
        x=x.view(32, 2, 1, 785)
        meanX=meanX.view(30,2,1,785)
        downX=downX.view(16,2,1,785)
        # print((x.size(),meanX.size(),downX.size()))
        #First dimension is different because for mean filter you end up removing 2 values, and with down sampling every second valkue you remove half
        #Literally the entire model is broken
 
        #x has some nan values
        #downX has some nan values
        #This is just because of the .to
        # print(x)
        # print(F.relu(self.conv1Molecule(x)))
        # print(F.relu(self.conv1Molecule(x)).size())
        x=self.pool(F.relu(self.conv1Molecule(x)))
        meanX=self.pool(F.relu(self.conv1Mean(meanX)))
        downX=self.pool(F.relu(self.conv1Downsample(downX)))



        # print("Size of x:", x.size())
        # print("Size of meanX:", meanX.size())
        # print("Size of downX:", downX.size())


        combinedX=torch.cat((x,meanX,downX),dim=0)#Concatenates the values
        # print(combinedX.size())
        combinedX=self.poolCombined(F.relu(self.conv2Combined(combinedX)))#Passes t hrough convolution and poolying layer
        # print(combinedX.size())
        combinedX=combinedX.view(32,-1)#Flattens combindeX tensor
        # print(combinedX.size())
        # print(combinedX)
        combinedX=F.relu(self.fc1(combinedX))#Passes through first fully connected layer
        output=self.fc2(combinedX)
        output=F.softmax(output,dim=1)
        return output


waterDataset=moleculeDataset(folderPath=r"C:\Users\Tristan\Downloads\ExoSeer\Data\Training\Water",molecule="Water",transform=transforms.ToTensor())
notWaterDataset=moleculeDataset(folderPath=r"C:\Users\Tristan\Downloads\ExoSeer\Data\Training\NotWater",molecule="notWater",transform=transforms.ToTensor())
waterDataset=combineDataset(waterDataset,notWaterDataset)


trainSize=int(0.8*len(waterDataset))
testSize=(len(waterDataset)-trainSize)//2
validationSize=(len(waterDataset)-trainSize)//2
trainDataset,testDataset,validationDataset=random_split(waterDataset,[trainSize,testSize,validationSize])
# print(len(validationDataset))


trainDataloader=DataLoader(trainDataset,batch_size=32,shuffle=True)
testDataloader=DataLoader(testDataset,batch_size=32,shuffle=True)
validationLoader=DataLoader(validationDataset,batch_size=32,shuffle=True,)#Otherwise errors


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=Model().to(device)

optimizer=optim.Adam(model.parameters(),lr=0.001)
criterion=nn.CrossEntropyLoss()

n=10
for epoch in range(n):
    model.train()
    for batch in trainDataloader:

        rawData,labels=batch
        mean_value=rawData.mean()
        std_value=rawData.std()
        rawData=(rawData-mean_value)/std_value#Normalizing



        # print(rawData.size())
        rawData=rawData.to(torch.float64)

        meanData=meanFilter(rawData,3)

        downSampled=rawData[::2]#Removes every 2.
        # print(rawData)
        # print(meanData)
        # print(downSampled)


        rawData=rawData.to(device)
        meanData=meanData.to(device)
        downSampled=downSampled.to(device)

        newLabels=[]
        for a in labels:
            if a=="Water":
                newLabels.append(1)
            else:
                newLabels.append(0)
        labels=torch.tensor(newLabels)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs=model(rawData,meanData,downSampled)
        # print(outputs)
        # print(labels)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        rawDimensions=list(rawData.size())
        meanDimensions=list(meanData.size())
        downSampledDimensions=list(downSampled.size())
        # print((rawDimensions,meanDimensions,downSampledDimensions))


    model.eval()
    with torch.no_grad():#Validation loop
        for batch in validationLoader:
            
            rawData,labels=batch
            mean_value=rawData.mean()
            std_value=rawData.std()
            rawData=(rawData-mean_value)/std_value#Normalizing
            if list(rawData.size())[0]==32:#Otherwise errors, since dimensions don't match
                meanData=meanFilter(rawData,3)

                downSampled=rawData[::2]#Removes every 2.
                # print(rawData)
                # print(meanData)
                # print(downSampled)


                rawData=rawData.to(device)
                meanData=meanData.to(device)
                downSampled=downSampled.to(device)
                newLabels=[]
                for a in labels:
                    if a=="Water":
                        newLabels.append(1)
                    else:
                        newLabels.append(0)
                labels=torch.tensor(newLabels)
                labels=labels.to(device)
                outputs=model(rawData,meanData,downSampled)

                valLoss=criterion(outputs,labels)

        # rawDimensions=list(rawData.size())
        # meanDimensions=list(meanData.size())
        # downSampledDimensions=list(downSampled.size())
        # print(rawData)
    print(f"Epoch {epoch+1}/{n}, Training Loss: {loss.item()}, Validation Loss: {valLoss.item()}")

