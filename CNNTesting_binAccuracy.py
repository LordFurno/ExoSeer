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
folderPath=r'C:\Users\User\Downloads\ExoSeer\Data\Training'
for combination in combinations:
    newPath=folderPath+f'\{"-".join(combination)}'
    folderDirectories.append(("-".join(combination),newPath))



train_samples=[]
train_labels=[]

test_samples=[]
test_labels=[]

val_samples=[]
val_labels=[]



np.random.seed(42)
random.seed(42)
all_samples=[]




trainBalance={}
valBalance={}
testBalance={}

#Issue is that each traning batch it goes: this label, next label, next label and so on. Everything needs to be shuffled, while being balanced
for label, directory in folderDirectories:
    files = os.listdir(directory)
    trainBalance[label]=0
    valBalance[label]=0
    testBalance[label]=0
    trainCounter=0
    random.shuffle(files) 
    for file in files:#2720 files per folder
        
        filePath = os.path.join(directory, file)
        all_samples.append((label,filePath))

random.shuffle(all_samples)



# print(len(all_samples))
for label,filePath in all_samples:
    if trainBalance[str(label)]<2176:
        trainBalance[str(label)]+=1
        train_samples.append(filePath)
        train_labels.append(oneHotEncoding(label))
    elif valBalance[str(label)]<272:
        valBalance[str(label)]+=1
        val_samples.append(filePath)
        val_labels.append(oneHotEncoding(label))
    else:
        testBalance[str(label)]+=1
        test_samples.append(filePath)
        test_labels.append(oneHotEncoding(label))
print(trainBalance)
print(valBalance)
print(testBalance)


train_samples=np.array(train_samples)
train_labels=np.array(train_labels)

val_samples=np.array(val_samples)
val_labels=np.array(val_labels)

test_samples=np.array(test_samples)
test_labels=np.array(test_labels)


# Create CustomDataGenerator for training and validation
train_generator = CustomDataGenerator(train_samples, train_labels, batch_size=32)
val_generator = CustomDataGenerator(val_samples, val_labels, batch_size=32)
test_generator = CustomDataGenerator(test_samples,test_labels, batch_size=32)



#Convert the labels into 1-hot encoding
#Each label will be an array that is 4 values long
#[0,0,0,0]
#The first value represents if N2 is present
#The second value represents if O2 is present
#The third value represents if CO2 is present
#The fourth value represents if H2O is present
model = keras.models.Sequential()
model.add(Conv1D(128, kernel_size=5, strides=2, activation='relu', input_shape=(785, 2),kernel_regularizer=regularizers.l2(0.03)))

model.add(Conv1D(64, kernel_size=3, strides=2, activation='relu',kernel_regularizer=regularizers.l2(0.02))) 

model.add(Conv1D(32,kernel_size=2,strides=2,activation='relu',kernel_regularizer=regularizers.l2(0.02)))      

model.add(Flatten())
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(4, activation='sigmoid'))#Final Layer using Sigmoid
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
history = model.fit(
    train_generator,  
    epochs=15,
    validation_data=val_generator,
    verbose=1
)

# Evaluate the model
loss, accuracy=model.evaluate(train_generator)
print(f'Validation Loss: {loss}, Validation Accuracy: {accuracy}')


# Make predictions on the validation data
val_predictions = model.predict(val_generator)
# Convert predicted probabilities to binary predictions
binary_predictions = (val_predictions > 0.5).astype(int)
# Get true labels from the validation generator
true_labels = val_generator.labels
# Calculate F1 score
precision = precision_score(true_labels, binary_predictions, average='micro')
recall = recall_score(true_labels, binary_predictions, average='micro')
f1 = f1_score(true_labels, binary_predictions, average='micro')
print(f'F1 Score: {f1}')
print(f'Recall score: {recall}')
print(f'Precision score: {precision}')
total_params = model.count_params()
print("Total number of parameters:", total_params)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
model.save("savedCNN2.keras")
