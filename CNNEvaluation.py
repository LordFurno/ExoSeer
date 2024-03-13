import tensorflow as tf
tf.compat.v1.executing_eagerly_outside_functions()
import keras
from keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, recall_score, f1_score
import os
import pandas as pd
import numpy as np
import random
def oneHotEncoding(label):
    res=[0.0,0.0,0.0,0.0]
    for value in label.split("-"):
        if value=="N2":
            res[0]=1.0
        elif value=="O2":
            res[1]=1.0
        elif value=="CO2":
            res[2]=1.0
        elif value=="H2O":
            res[3]=1.0
    return res
def loadCSV(filePath):
    return pd.read_csv(filePath)
class CustomDataGenerator(Sequence):
    def __init__(self, samples, labels, batch_size=32):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.num_samples = len(samples)


    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))


    def __getitem__(self, index):
        batch_samples = self.samples[index * self.batch_size : (index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size : (index + 1) * self.batch_size]
        batch_data = []
        for sample in batch_samples:
            data = loadCSV(sample)
            #data.iloc[:, 0] is wavelength and data.iloc[:, 1] is total
            #Reshape each channel to have a single dimension
            wavelength = np.expand_dims(np.array(data.iloc[:, 0]), axis=-1)
            total = np.expand_dims(np.array(data.iloc[:, 1]), axis=-1)
            # Combine the channels to create the input data for each sample
            input_data = np.concatenate((wavelength, total), axis=-1)
            batch_data.append(input_data)
        # print(np.array(batch_))
        return np.array(batch_data), np.array(batch_labels)

combinations=[('N2',), ('O2',), ('CO2',), ('H2O',), ('N2', 'O2'), ('N2', 'CO2'), ('N2', 'H2O'), ('O2', 'CO2'), ('O2', 'H2O'), ('CO2', 'H2O'), ('N2', 'O2', 'CO2'), ('N2', 'O2', 'H2O'), ('N2', 'CO2', 'H2O'), ('O2', 'CO2', 'H2O'), ('N2', 'O2', 'CO2', 'H2O')]
folderDirectories=[]
folderPath=r'C:\Users\Tristan\Downloads\ExoSeer\Data\Training'
for combination in combinations:
    newPath=folderPath+f'\{"-".join(combination)}'
    folderDirectories.append(("-".join(combination),newPath))


test_samples=[]
test_labels=[]


np.random.seed(42)
random.seed(42)
all_samples=[]


testBalance={}

#Issue is that each traning batch it goes: this label, next label, next label and so on. Everything needs to be shuffled, while being balanced
for label, directory in folderDirectories:
    files = os.listdir(directory)
    testBalance[label]=0
    trainCounter=0
    random.shuffle(files) 
    for file in files:#2720 files per folder
        
        filePath = os.path.join(directory, file)
        all_samples.append((label,filePath))

random.shuffle(all_samples)


# print(len(all_samples))
for label,filePath in all_samples:

    testBalance[str(label)]+=1
    test_samples.append(filePath)
    test_labels.append(oneHotEncoding(label))



test_samples=np.array(test_samples)
test_labels=np.array(test_labels)

test_generator = CustomDataGenerator(test_samples,test_labels, batch_size=32)
model = keras.models.load_model(f'C:\Users\Tristan\Downloads\ExoSeer\savedCNN.keras')
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
