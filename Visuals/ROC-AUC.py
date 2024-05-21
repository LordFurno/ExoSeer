import tensorflow as tf
import random
import numpy as np
import pandas as pd
import os
from keras.utils import Sequence
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
def z_score_standardization(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    standardized_data = (data - mean) / std_dev
    return standardized_data
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
            wavelength=np.array(data.iloc[:, 0])
            total=np.array(data.iloc[:, 1])
            wavelength = np.expand_dims(z_score_standardization(wavelength), axis=-1)
            total = np.expand_dims(z_score_standardization(total), axis=-1)
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



train_samples=[]
train_labels=[]

test_samples=[]
test_labels=[]

val_samples=[]
val_labels=[]



np.random.seed(43)
random.seed(43)
all_samples=[]



test_samples=[]


#Issue is that each traning batch it goes: this label, next label, next label and so on. Everything needs to be shuffled, while being balanced
for label, directory in folderDirectories:
    files = os.listdir(directory)
    trainCounter=0
    random.shuffle(files)
    for file in files:#4080 files per folder
        filePath = os.path.join(directory, file)
        all_samples.append((label,filePath))

random.shuffle(all_samples)#Shuffles data, ensures that sample matches label

testIndexes=random.sample(range(len(all_samples)),int(len(all_samples)*0.2))#Randomly select indexes to make the testing dataset
testIndexes.sort(reverse=True)#So when we remove from original list, it actually removes the values

test_samples=[]
for index in testIndexes:
    test_samples.append(all_samples.pop(index))#This also removes from all_samples



all_labels,all_samples=list(zip(*all_samples))

all_labels,all_samples=list(all_labels),list(all_samples)
for i in range(len(all_labels)):#Converts label into one-hot vectors
    all_labels[i]=oneHotEncoding(all_labels[i])
all_labels,all_samples=np.array(all_labels),np.array(all_samples)


random.shuffle(test_samples)
test_labels,test_samples=list(zip(*test_samples))
test_labels,test_samples=list(test_labels),list(test_samples)
for i in range(len(test_labels)):
    test_labels[i]=oneHotEncoding(test_labels[i])
test_labels,test_samples=np.array(test_labels),np.array(test_samples)




# print(all_labels)
# print(all_samples)
# print(test_samples)
# print(test_labels)
print(test_labels)
test_generator=CustomDataGenerator(test_samples, test_labels, batch_size=32)
model=tf.keras.models.load_model(r'C:\Users\Tristan\Downloads\ExoSeer\CNN.keras')
val_predictions=model.predict(test_generator)
true_labels=test_generator.labels

num_labels=test_labels.shape[1]
names=["N2","O2","CO2","H2O"]
# Calculate AUC-ROC for each label
auc_roc_scores = []
for label_idx in range(num_labels):
    fpr,tpr,_=roc_curve(test_labels[:, label_idx], val_predictions[:, label_idx])#False positive, true positive rate 
    auc_score=roc_auc_score(test_labels[:, label_idx], val_predictions[:, label_idx])
    auc_roc_scores.append(auc_score)
    
    # Plot the ROC curve for each label
    plt.plot(fpr, tpr, label=f'{names[label_idx ]} (AUC: {auc_score:.2f})')

# Add plot labels and legend
plt.rcParams.update({'font.size': 22})
plt.xlabel('False Positive Rate',fontsize=15)
plt.ylabel('True Positive Rate',fontsize=15)
plt.title('ROC Curves for Molecule Classification')
plt.legend(loc='lower right')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show(block=True)


# Calculate the average AUC-ROC score
average_auc_roc = np.mean(auc_roc_scores)

print(f'Average AUC-ROC: {average_auc_roc:.2f}')