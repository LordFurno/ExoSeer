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
def oneHotEncoding(label):
    res=[0,0,0,0]
    for value in label.split("-"):
        if value=="N2":
            res[0]=1
        elif value=="O2":
            res[1]=1
        elif value=="CO2":
            res[2]=1
        elif value=="H2O":
            res[3]=1
    return res

class customDataset(Dataset):
    def __init__(self,folderPaths,transform=None):
        self.folderPaths=folderPaths
        self.transform=transform
        self.labels=[]
        self.samples=[]
        for label,directory in folderPaths:
            files=os.listdir(directory)
            for file in files:
                filePath=os.path.join(directory,file)
                self.samples.append(filePath)
                self.labels.append(label)
        # print(len(self.csvFiles))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,index):
        sample=self.samples[index]
        label=self.labels[index]

        # print(index)
        data=loadCSV(sample)

        wavelength=data.iloc[:,0]#Wavelength column
        total=data.iloc[:,1]#Total transmittance column

        if self.transform:
            wavelength,total=np.array(wavelength),np.array(total)
            wavelength,total=wavelength.reshape(1,-1),total.reshape(1,-1)
            wavelength=self.transform(wavelength)
            total=self.transform(total)
            data=torch.stack([wavelength,total],dim=0)#Combines the wavelength and total together, maintinas pairing
        return data,label#Wavelength, label


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        #Add proper inputs in brackets. Need to verify shape
        self.conv1=nn.Conv2d(2,32, kernel_size=(1,3))
        self.pool1=nn.MaxPool2d((1,2),stride=(1,2))
        self.conv2=nn.Conv2d(32,16,kernel_size=(1,3))
        self.fc1=nn.Linear(3104,32)#Shape of flattened tensor
        self.fc2=nn.Linear(32,4)
        pass

    def forward(self,x):
        x=x.float()
        x=x.squeeze(2)#Removes second dimension, it isn't needed

        #Shape is torch.Size([32, 32, 1, 785]), might not be good that height is 1
        x=self.pool1(F.relu(self.conv1(x)))

        x=self.pool1(F.relu(self.conv2(x)))
        x=x.view(x.size(0),-1)#Need proper size, to determine how to flatten

        x=F.relu(self.fc1(x))
        #Might need to apply an activation function to deal with negative values
        output=self.fc2(x)
        output=torch.sigmoid(output)
        output=output.view(-1,4)
        return output



torch.manual_seed(43)#So I can reproduce the results
np.random.seed(43)

combinations=[('N2',), ('O2',), ('CO2',), ('H2O',), ('N2', 'O2'), ('N2', 'CO2'), ('N2', 'H2O'), ('O2', 'CO2'), ('O2', 'H2O'), ('CO2', 'H2O'), ('N2', 'O2', 'CO2'), ('N2', 'O2', 'H2O'), ('N2', 'CO2', 'H2O'), ('O2', 'CO2', 'H2O'), ('N2', 'O2', 'CO2', 'H2O')]
folderDirectories=[]
folderPath=r'C:\Users\Tristan\Downloads\ExoSeer\Data\Training'
for combination in combinations:
    newPath=folderPath+f'\{"-".join(combination)}'
    folderDirectories.append(("-".join(combination),newPath))

dataset=customDataset(folderPaths=folderDirectories,transform=transforms.ToTensor())

trainSize=int(0.8*len(dataset))
testSize=(len(dataset)-trainSize)//2
validationSize=(len(dataset)-trainSize)//2


trainDataset,testDataset,validationDataset=random_split(dataset,[trainSize,testSize,validationSize])
# print(len(validationDataset))


trainDataloader=DataLoader(trainDataset,batch_size=32,shuffle=True)
testDataloader=DataLoader(testDataset,batch_size=32,shuffle=True)
validationLoader=DataLoader(validationDataset,batch_size=32,shuffle=True)


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=Model().to(device)
optimizer=optim.Adam(model.parameters(),lr=0.001)#Might need weight decay
criterion=nn.CrossEntropyLoss()
n=15
losses=[]
for epoch in range(n):
    for batch in trainDataloader:
        model.train()
        data,labels=batch[0],list(batch[1])
        #Convert the labels into 1-hot encoding
        #Each label will be an array that is 4 values long
        #[0,0,0,0]
        #The first value represents if N2 is present
        #The second value represents if O2 is present
        #The third value represents if CO2 is present
        #The fourth value represents if H2O is present
        for i,value in enumerate(labels):
            labels[i]=oneHotEncoding(value)
        data=normalize(data,0.0,1.0)#Normalizing
        data=data.to(torch.float64)
        data=data.to(device)

        labels=torch.tensor(labels)
        labels=labels.to(device)
        labels=labels.float()

        #Running the model
        optimizer.zero_grad()
        outputs=model(data)
        loss=criterion(outputs,labels)

    with torch.no_grad():#Validation loop
        for batch in validationLoader:
            model.eval()
            data,labels=batch[0],list(batch[1])
            if list(data.size())[0]==32:#To make sure that it is proper size
                data=normalize(data,0.0,1.0)
                data=data.to(torch.float64)
                data=data.to(device)
                for i,values in enumerate(labels):
                    labels[i]=oneHotEncoding(value)
                labels=torch.tensor(labels)
                labels=labels.to(device)
                outputs=model(data)
                validCorrect=0
                validTotal=0
                for i,a in enumerate(outputs):
                    res=[0.0]*4
                    if a[0]>0.5:
                        #N2 is present
                        res[0]=1.0
                    if a[1]>0.5:
                        #O2 is present
                        res[1]=1.0
                    if a[2]>0.5:
                        #CO2 is present
                        a[2]=1.0
                    if a[3]>0.5:
                        #H2O is present
                        a[3]=1.0
                    if res==labels[i]:
                        validCorrect+=1
                    validTotal+=1
                labels=labels.float()
                valLoss=criterion(outputs,labels)
    print(f"Epoch {epoch+1}/{n}, Training Loss: {loss.item()}, Validation Loss: {valLoss.item()}, Validation Accuracy: {100 * validCorrect / validTotal}%")
    losses.append(loss.item())





# model=Model().to(device)

# optimizer=optim.Adam(model.parameters(),lr=0.001,weight_decay=0.001)
# # optimizer=optim.RMSprop(model.parameters(), lr=0.01)
# scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.0005)
# criterion=nn.BCELoss()


# n=15
# losses=[]
# validCorrect=0
# validTotal=0
# for epoch in range(n):
#     for batch in trainDataloader:
#         model.train()
#         rawData,labels=batch

#         rawData=normalize(rawData,0.0,1.0)#Normalizing
#         rawData=rawData.to(torch.float64)



#         rawData=rawData.to(device)


#         newLabels=[]
#         for a in labels:
#             if a==molecule:
#                 newLabels.append(1)
#             else:
#                 newLabels.append(0)
#         labels=torch.tensor(newLabels)
#         labels=labels.float()
#         labels=labels.to(device)
        
#         optimizer.zero_grad()
#         outputs=model(rawData)

        
#         loss=criterion(outputs,labels)
#         loss.backward()
#         optimizer.step()



#     with torch.no_grad():#Validation loop
#         for batch in validationLoader:
#             model.eval()
#             rawData,labels=batch
#             mean_value=rawData.mean()
#             std_value=rawData.std()
#             rawData=(rawData-mean_value)/std_value#Normalizing
#             if list(rawData.size())[0]==32:#Otherwise errors, since dimensions don't match
#                 rawData=rawData.to(device)
#                 # meanData=meanData.to(device)
#                 # downSampled=downSampled.to(device)
#                 newLabels=[]
#                 for a in labels:
#                     if a==molecule:
#                         newLabels.append(1)
#                     else:
#                         newLabels.append(0)
#                 labels=torch.tensor(newLabels)
#                 labels=labels.to(device)
#                 outputs=model(rawData)
                    
#                 for i,a in enumerate(outputs):
#                     if a>0.5:#Water is present
#                         if labels[i]==1:
#                             validCorrect+=1
#                     else:
#                         if labels[i]==0:
#                             validCorrect +=1
#                     validTotal+=1



#                 labels=labels.float()
#                 valLoss=criterion(outputs,labels)

#         # rawDimensions=list(rawData.size())
#         # meanDimensions=list(meanData.size())
#         # downSampledDimensions=list(downSampled.size())
#         # print(rawData)
#     print(f"Epoch {epoch+1}/{n}, Training Loss: {loss.item()}, Validation Loss: {valLoss.item()}, Validation Accuracy: {100 * validCorrect / validTotal}%")
#     losses.append(loss.item())

# model.eval()
# correct=0
# total=0
# with torch.no_grad():
#     for batch,labels in testDataloader:
#         rawData=batch
#         if list(rawData.size())[0]==32:#Otherwise errors, since dimensions don't match
#             rawData=normalize(rawData,0.0,1.0)#Normalizing
#             rawData=rawData.to(torch.float64)

#             # meanData=meanFilter(rawData,3)
#             # meanData=normalize(meanData,0.0,1.0)
#             # meanData=meanData.to(torch.float64)

#             # downSampled=rawData[::2]#Removes every 2.
#             # downSampled=normalize(downSampled,0.0,1.0)
#             # downSampled=downSampled.to(torch.float64)

#             rawData=rawData.to(device)
#             # meanData=meanData.to(device)
#             # downSampled=downSampled.to(device)
#             newLabels=[]
#             for a in labels:
#                 if a==molecule:
#                     newLabels.append(1)
#                 else:
#                     newLabels.append(0)
#             labels=torch.tensor(newLabels)
#             labels=labels.float()
#             labels=labels.to(device)
#             outputs=model(rawData)
#             for i,a in enumerate(outputs):
#                 if a>0.5:#Water is present
#                     if labels[i]==1:
#                         correct+=1
#                 else:
#                     if labels[i]==0:
#                         correct +=1
#                 total+=1

# cnnFilePath=r"C:\Users\Tristan\Downloads\ExoSeer"+f"\{molecule}CNN"
# torch.save(model.state_dict,cnnFilePath)
# print(f"Accuracy on the test set: {100 * correct / total}%") 
# plt.plot(losses)
# plt.ylabel("Training loss")
# plt.show(block=True)
# '''
# Few things left to do. First of all make sure this model sin't overfitting.
#  Secondly figure out how I want to do the multiple CNN files. Whether or not i want them in different files.
#  Then save the trained models, and work on the code to extract features and 
# '''