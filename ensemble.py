import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,random_split
import os
class totalDataset(Dataset):
    def __init__(self,folderPath,transform=None):
        self.folderPath=folderPath
        self.csvFiles=[f for f in os.listdir(folderPath) if f.endswith('.csv')]
        self.transform=transform
    def __len__(self):
        return len(self.csvFiles)
    def __getitem__(self, index):
        csvFile=self.csvFiles[index]
        filePath=os.path.join(self.folderPath,csvFile)
        