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
from matplotlib import pyplot as plt
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
def normalize(data,minVal,maxVal):
    scaledData=(data-data.min())/(data.max()-data.min())
    scaledData=scaledData*(maxVal-minVal)+minVal
    return scaledData


class moleculeDataset(Dataset):
    def __init__(self,folderPath,molecule,transform=None):
        self.folderPath=folderPath
        self.molecule=molecule
        self.csvFiles=[f for f in os.listdir(folderPath) if f.endswith('.csv')]
        # print(len(self.csvFiles))
        self.transform= transform
    def __len__(self):
        return len(self.csvFiles)
    
    def __getitem__(self,index):
        # print(index)
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
        return data,self.molecule#Wavelength, molecule,label
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
        
        self.pool=nn.MaxPool2d(kernel_size=(1,2),stride=(1,2))
        
        self.conv2Molecule=nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)#Input is 78, because this is what the 32+30+16 is.

        self.pool2=nn.MaxPool2d(kernel_size=1,stride=2)

        self.fc1=nn.Linear(16*784,32)#16*1911, since that is the shape of my data
        self.fc2=nn.Linear(32,2)#Output layer is 2 because it is either present or not
       
    def forward(self,x):

        x=x.float()
        x=x.view(32, 2, 1, 785)
        # print(x)


        x=self.pool(F.relu(self.conv1Molecule(x)))
        x=self.pool2(F.relu(self.conv2Molecule(x)))

        
      
        x=x.view(32,-1)#Flattens combindeX tensor
        # print(combinedX.size())
        # print(x.size())
        x=F.relu(self.fc1(x))#Passes through first fully connected layer
        output=self.fc2(x)

        output=torch.sigmoid(output)
        # print(output)
        return output[:,1]



molecule="Nitrogen"
notName="Not"+molecule
waterDataset=moleculeDataset(folderPath=r"C:\Users\Tristan\Downloads\ExoSeer\Data\Training"+f"\{molecule}",molecule=molecule,transform=transforms.ToTensor())
notWaterDataset=moleculeDataset(folderPath=r"C:\Users\Tristan\Downloads\ExoSeer\Data\Training"+f"\{notName}",molecule=notName,transform=transforms.ToTensor())
waterDataset=combineDataset(waterDataset,notWaterDataset)

indexes=list(range(len(waterDataset)))



trainSize=int(0.8*len(waterDataset))
testSize=(len(waterDataset)-trainSize)//2
validationSize=(len(waterDataset)-trainSize)//2


trainDataset,testDataset,validationDataset=random_split(waterDataset,[trainSize,testSize,validationSize])
# print(len(validationDataset))


trainDataloader=DataLoader(trainDataset,batch_size=32,shuffle=True)
testDataloader=DataLoader(testDataset,batch_size=32,shuffle=True)
validationLoader=DataLoader(validationDataset,batch_size=32,shuffle=True)


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=Model().to(device)

optimizer=optim.Adam(model.parameters(),lr=0.0017)
# optimizer=optim.RMSprop(model.parameters(), lr=0.01)
criterion=nn.BCELoss()
torch.manual_seed(42)#So I can reproduce the results
n=15
losses=[]
validCorrect=0
validTotal=0
for epoch in range(n):#It's really weird, only firstt value in batch is really accurate
    for batch in trainDataloader:
        model.train()
        rawData,labels=batch

        rawData=normalize(rawData,0.0,1.0)#Normalizing
        rawData=rawData.to(torch.float64)



        rawData=rawData.to(device)


        newLabels=[]
        for a in labels:
            if a==molecule:
                newLabels.append(1)
            else:
                newLabels.append(0)
        labels=torch.tensor(newLabels)
        labels=labels.float()
        labels=labels.to(device)
        
        optimizer.zero_grad()
        outputs=model(rawData)

        
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()



    with torch.no_grad():#Validation loop
        for batch in validationLoader:
            model.eval()
            rawData,labels=batch
            mean_value=rawData.mean()
            std_value=rawData.std()
            rawData=(rawData-mean_value)/std_value#Normalizing
            if list(rawData.size())[0]==32:#Otherwise errors, since dimensions don't match
                rawData=rawData.to(device)
                # meanData=meanData.to(device)
                # downSampled=downSampled.to(device)
                newLabels=[]
                for a in labels:
                    if a==molecule:
                        newLabels.append(1)
                    else:
                        newLabels.append(0)
                labels=torch.tensor(newLabels)
                labels=labels.to(device)
                outputs=model(rawData)
                for i,a in enumerate(outputs):
                    if a>0.5:#Water is present
                        if labels[i]==1:
                            validCorrect+=1
                    else:
                        if labels[i]==0:
                            validCorrect +=1
                    validTotal+=1



                labels=labels.float()
                valLoss=criterion(outputs,labels)

        # rawDimensions=list(rawData.size())
        # meanDimensions=list(meanData.size())
        # downSampledDimensions=list(downSampled.size())
        # print(rawData)
    print(f"Epoch {epoch+1}/{n}, Training Loss: {loss.item()}, Validation Loss: {valLoss.item()}, Validation Accuracy: {100 * validCorrect / validTotal}%")
    losses.append(loss.item())

model.eval()
correct=0
total=0
with torch.no_grad():
    for batch,labels in testDataloader:
        rawData=batch
        if list(rawData.size())[0]==32:#Otherwise errors, since dimensions don't match
            rawData=normalize(rawData,0.0,1.0)#Normalizing
            rawData=rawData.to(torch.float64)

            # meanData=meanFilter(rawData,3)
            # meanData=normalize(meanData,0.0,1.0)
            # meanData=meanData.to(torch.float64)

            # downSampled=rawData[::2]#Removes every 2.
            # downSampled=normalize(downSampled,0.0,1.0)
            # downSampled=downSampled.to(torch.float64)

            rawData=rawData.to(device)
            # meanData=meanData.to(device)
            # downSampled=downSampled.to(device)
            newLabels=[]
            for a in labels:
                if a==molecule:
                    newLabels.append(1)
                else:
                    newLabels.append(0)
            labels=torch.tensor(newLabels)
            labels=labels.float()
            labels=labels.to(device)
            outputs=model(rawData)
            for i,a in enumerate(outputs):
                if a>0.5:#Water is present
                    if labels[i]==1:
                        correct+=1
                else:
                    if labels[i]==0:
                        correct +=1
                total+=1

print(f"Accuracy on the test set: {100 * correct / total}%")
plt.plot(losses)
plt.ylabel("Training loss")
plt.show(block=True)
'''
Few things left to do. First of all make sure this model sin't overfitting.
 Secondly figure out how I want to do the multiple CNN files. Whether or not i want them in different files.
 Then save the trained models, and work on the code to extract features and 
'''