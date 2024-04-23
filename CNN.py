
import tensorflow as tf
tf.compat.v1.executing_eagerly_outside_functions()
import keras
from keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score,multilabel_confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import os
import pandas as pd
import numpy as np
import random


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
test_generator = CustomDataGenerator(test_samples, test_labels, batch_size=32)
#Train the model tommorow, evaluate on test generator
f1_scores = []
n_splits = 5  # Number of folds (adjust as necessary)
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for train_index, val_index in cv.split(all_samples, np.argmax(all_labels, axis=1)):
    # Split the data into training and validation sets
    X_train, X_val = all_samples[train_index], all_samples[val_index]
    y_train, y_val = all_labels[train_index], all_labels[val_index]

    # Create custom data generators for training and validation sets
    train_generator = CustomDataGenerator(X_train, y_train, batch_size=32)
    val_generator = CustomDataGenerator(X_val, y_val, batch_size=32)


    # Create a new model instance
    model = keras.models.Sequential()

    model.add(Conv1D(128, kernel_size=5, strides=2, activation='relu', input_shape=(785, 2),kernel_regularizer=regularizers.l2(0.05)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(128, kernel_size=5, strides=2, activation='relu', input_shape=(785, 2),kernel_regularizer=regularizers.l2(0.05)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(64, kernel_size=3, strides=2, activation='relu',kernel_regularizer=regularizers.l2(0.05))) 
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(32,kernel_size=2,strides=2,activation='relu',kernel_regularizer=regularizers.l2(0.05)))      
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.05)))

    model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.05)))

    model.add(Dropout(0.75))
    model.add(Dense(4, activation='sigmoid'))#Final Layer using Sigmoid
    # Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['binary_accuracy']
    )

    # Train the model on the training set
    model.fit(train_generator, epochs=15, verbose=1, validation_data=val_generator)

    # Evaluate the model on the validation set
    val_predictions = model.predict(val_generator)

    # Convert predictions to binary labels based on a threshold (e.g., 0.5)
    threshold = 0.5
    binary_val_predictions = (val_predictions > threshold).astype(int)

    # Calculate F1 score for each fold
    f1 = f1_score(y_val, binary_val_predictions, average='macro')
    f1_scores.append(f1)

# Calculate mean and standard deviation of F1 scores
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)
print(f1_scores)

print(f'Mean F1 Score: {mean_f1:.4f}')
print(f'Standard Deviation of F1 Score: {std_f1:.4f}')

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

val_predictions = model.predict(test_generator)
# Convert predicted probabilities to binary predictions
binary_predictions = (val_predictions > 0.5).astype(int)
# Get true labels from the validation generator
true_labels = test_generator.labels
# Calculate F1 score
precision = precision_score(true_labels, binary_predictions, average='micro')
recall = recall_score(true_labels, binary_predictions, average='micro')
f1 = f1_score(true_labels, binary_predictions, average='micro')
print(f'F1 Score: {f1}')
print(f'Recall score: {recall}')
print(f'Precision score: {precision}')


model.save("CNN.keras")
#I FIXED IT
'''
Epoch 1/15
1224/1224 [==============================] - 434s 354ms/step - loss: 1.4299 - binary_accuracy: 0.7817 - val_loss: 0.3616 - val_binary_accuracy: 0.8189
Epoch 2/15
1224/1224 [==============================] - 102s 83ms/step - loss: 0.3885 - binary_accuracy: 0.8020 - val_loss: 0.3266 - val_binary_accuracy: 0.8171
Epoch 3/15
1224/1224 [==============================] - 100s 81ms/step - loss: 0.3564 - binary_accuracy: 0.8049 - val_loss: 0.3158 - val_binary_accuracy: 0.8176
Epoch 4/15
1224/1224 [==============================] - 201s 164ms/step - loss: 0.3445 - binary_accuracy: 0.8048 - val_loss: 0.3168 - val_binary_accuracy: 0.8152
Epoch 5/15
1224/1224 [==============================] - 293s 240ms/step - loss: 0.3313 - binary_accuracy: 0.8063 - val_loss: 0.2992 - val_binary_accuracy: 0.8181
Epoch 6/15
1224/1224 [==============================] - 101s 82ms/step - loss: 0.3296 - binary_accuracy: 0.8047 - val_loss: 0.2953 - val_binary_accuracy: 0.8181
Epoch 7/15
1224/1224 [==============================] - 101s 82ms/step - loss: 0.3252 - binary_accuracy: 0.8072 - val_loss: 0.3041 - val_binary_accuracy: 0.8179
Epoch 8/15
1224/1224 [==============================] - 101s 82ms/step - loss: 0.3167 - binary_accuracy: 0.8073 - val_loss: 0.2999 - val_binary_accuracy: 0.8175
Epoch 9/15
1224/1224 [==============================] - 100s 82ms/step - loss: 0.3117 - binary_accuracy: 0.8070 - val_loss: 0.2880 - val_binary_accuracy: 0.8170
Epoch 10/15
1224/1224 [==============================] - 100s 82ms/step - loss: 0.3085 - binary_accuracy: 0.8084 - val_loss: 0.2884 - val_binary_accuracy: 0.8181
Epoch 11/15
1224/1224 [==============================] - 100s 82ms/step - loss: 0.3091 - binary_accuracy: 0.8080 - val_loss: 0.2786 - val_binary_accuracy: 0.8175
Epoch 12/15
1224/1224 [==============================] - 99s 81ms/step - loss: 0.3053 - binary_accuracy: 0.8075 - val_loss: 0.2810 - val_binary_accuracy: 0.8182
Epoch 13/15
1224/1224 [==============================] - 98s 80ms/step - loss: 0.3053 - binary_accuracy: 0.8086 - val_loss: 0.3123 - val_binary_accuracy: 0.8067
Epoch 14/15
1224/1224 [==============================] - 100s 82ms/step - loss: 0.3045 - binary_accuracy: 0.8086 - val_loss: 0.3050 - val_binary_accuracy: 0.8107
Epoch 15/15
1224/1224 [==============================] - 100s 81ms/step - loss: 0.3039 - binary_accuracy: 0.8082 - val_loss: 0.3001 - val_binary_accuracy: 0.8152
306/306 [==============================] - 16s 50ms/step
Epoch 1/15
1224/1224 [==============================] - 98s 80ms/step - loss: 1.4502 - binary_accuracy: 0.7826 - val_loss: 0.3719 - val_binary_accuracy: 0.8147
Epoch 2/15
1224/1224 [==============================] - 97s 79ms/step - loss: 0.3864 - binary_accuracy: 0.8031 - val_loss: 0.3326 - val_binary_accuracy: 0.8164
Epoch 3/15
1224/1224 [==============================] - 97s 79ms/step - loss: 0.3606 - binary_accuracy: 0.8039 - val_loss: 0.3297 - val_binary_accuracy: 0.8172
Epoch 4/15
1224/1224 [==============================] - 97s 79ms/step - loss: 0.3439 - binary_accuracy: 0.8054 - val_loss: 0.3058 - val_binary_accuracy: 0.8179
Epoch 5/15
1224/1224 [==============================] - 96s 79ms/step - loss: 0.3370 - binary_accuracy: 0.8049 - val_loss: 0.3032 - val_binary_accuracy: 0.8162
Epoch 6/15
1224/1224 [==============================] - 96s 79ms/step - loss: 0.3265 - binary_accuracy: 0.8063 - val_loss: 0.3038 - val_binary_accuracy: 0.8168
Epoch 7/15
1224/1224 [==============================] - 95s 78ms/step - loss: 0.3280 - binary_accuracy: 0.8057 - val_loss: 0.2939 - val_binary_accuracy: 0.8168
Epoch 8/15
1224/1224 [==============================] - 96s 78ms/step - loss: 0.3157 - binary_accuracy: 0.8068 - val_loss: 0.4845 - val_binary_accuracy: 0.7452
Epoch 9/15
1224/1224 [==============================] - 96s 78ms/step - loss: 0.3187 - binary_accuracy: 0.8062 - val_loss: 0.2888 - val_binary_accuracy: 0.8169
Epoch 10/15
1224/1224 [==============================] - 96s 78ms/step - loss: 0.3154 - binary_accuracy: 0.8068 - val_loss: 0.7069 - val_binary_accuracy: 0.7686
Epoch 11/15
1224/1224 [==============================] - 97s 79ms/step - loss: 0.3110 - binary_accuracy: 0.8069 - val_loss: 0.2886 - val_binary_accuracy: 0.8168
Epoch 12/15
1224/1224 [==============================] - 96s 79ms/step - loss: 0.3114 - binary_accuracy: 0.8082 - val_loss: 0.2868 - val_binary_accuracy: 0.8148
Epoch 13/15
1224/1224 [==============================] - 96s 78ms/step - loss: 0.3065 - binary_accuracy: 0.8070 - val_loss: 0.2778 - val_binary_accuracy: 0.8170
Epoch 14/15
1224/1224 [==============================] - 96s 79ms/step - loss: 0.3051 - binary_accuracy: 0.8080 - val_loss: 0.2863 - val_binary_accuracy: 0.8157
Epoch 15/15
1224/1224 [==============================] - 96s 79ms/step - loss: 0.3053 - binary_accuracy: 0.8073 - val_loss: 0.2829 - val_binary_accuracy: 0.8159
306/306 [==============================] - 15s 48ms/step
Epoch 1/15
1224/1224 [==============================] - 104s 84ms/step - loss: 1.4061 - binary_accuracy: 0.7864 - val_loss: 0.3859 - val_binary_accuracy: 0.8150
Epoch 2/15
1224/1224 [==============================] - 94s 77ms/step - loss: 0.3841 - binary_accuracy: 0.8052 - val_loss: 0.3486 - val_binary_accuracy: 0.8135
Epoch 3/15
1224/1224 [==============================] - 94s 77ms/step - loss: 0.3537 - binary_accuracy: 0.8078 - val_loss: 0.3661 - val_binary_accuracy: 0.8118
Epoch 4/15
1224/1224 [==============================] - 94s 77ms/step - loss: 0.3381 - binary_accuracy: 0.8102 - val_loss: 0.3041 - val_binary_accuracy: 0.8147
Epoch 5/15
1224/1224 [==============================] - 94s 77ms/step - loss: 0.3282 - binary_accuracy: 0.8096 - val_loss: 0.2997 - val_binary_accuracy: 0.8136
Epoch 6/15
1224/1224 [==============================] - 93s 76ms/step - loss: 0.3202 - binary_accuracy: 0.8092 - val_loss: 0.2984 - val_binary_accuracy: 0.8137
Epoch 7/15
1224/1224 [==============================] - 93s 76ms/step - loss: 0.3176 - binary_accuracy: 0.8103 - val_loss: 0.2903 - val_binary_accuracy: 0.8137
Epoch 8/15
1224/1224 [==============================] - 93s 76ms/step - loss: 0.3115 - binary_accuracy: 0.8103 - val_loss: 0.3699 - val_binary_accuracy: 0.8076
Epoch 9/15
1224/1224 [==============================] - 93s 76ms/step - loss: 0.3064 - binary_accuracy: 0.8098 - val_loss: 0.2845 - val_binary_accuracy: 0.8143
Epoch 10/15
1224/1224 [==============================] - 93s 76ms/step - loss: 0.3097 - binary_accuracy: 0.8094 - val_loss: 0.3046 - val_binary_accuracy: 0.8068
Epoch 11/15
1224/1224 [==============================] - 93s 76ms/step - loss: 0.3050 - binary_accuracy: 0.8106 - val_loss: 0.2807 - val_binary_accuracy: 0.8135
Epoch 12/15
1224/1224 [==============================] - 93s 76ms/step - loss: 0.3079 - binary_accuracy: 0.8091 - val_loss: 0.2835 - val_binary_accuracy: 0.8138
Epoch 13/15
1224/1224 [==============================] - 93s 76ms/step - loss: 0.3032 - binary_accuracy: 0.8096 - val_loss: 0.2856 - val_binary_accuracy: 0.8144
Epoch 14/15
1224/1224 [==============================] - 93s 76ms/step - loss: 0.3022 - binary_accuracy: 0.8093 - val_loss: 0.2842 - val_binary_accuracy: 0.8140
Epoch 15/15
1224/1224 [==============================] - 93s 76ms/step - loss: 0.3005 - binary_accuracy: 0.8101 - val_loss: 0.2913 - val_binary_accuracy: 0.8131
306/306 [==============================] - 14s 46ms/step
Epoch 1/15
1224/1224 [==============================] - 101s 81ms/step - loss: 1.4411 - binary_accuracy: 0.7814 - val_loss: 0.4680 - val_binary_accuracy: 0.7460
Epoch 2/15
1224/1224 [==============================] - 100s 82ms/step - loss: 0.3863 - binary_accuracy: 0.8022 - val_loss: 0.3363 - val_binary_accuracy: 0.8154
Epoch 3/15
1224/1224 [==============================] - 101s 82ms/step - loss: 0.3564 - binary_accuracy: 0.8058 - val_loss: 0.3180 - val_binary_accuracy: 0.8150
Epoch 4/15
1224/1224 [==============================] - 101s 83ms/step - loss: 0.3428 - binary_accuracy: 0.8071 - val_loss: 0.3171 - val_binary_accuracy: 0.8181
Epoch 5/15
1224/1224 [==============================] - 100s 82ms/step - loss: 0.3311 - binary_accuracy: 0.8076 - val_loss: 0.3004 - val_binary_accuracy: 0.8186
Epoch 6/15
1224/1224 [==============================] - 101s 82ms/step - loss: 0.3209 - binary_accuracy: 0.8082 - val_loss: 0.2976 - val_binary_accuracy: 0.8172
Epoch 7/15
1224/1224 [==============================] - 100s 82ms/step - loss: 0.3223 - binary_accuracy: 0.8076 - val_loss: 0.2900 - val_binary_accuracy: 0.8178
Epoch 8/15
1224/1224 [==============================] - 99s 81ms/step - loss: 0.3141 - binary_accuracy: 0.8070 - val_loss: 0.2847 - val_binary_accuracy: 0.8191
Epoch 9/15
1224/1224 [==============================] - 99s 81ms/step - loss: 0.3079 - binary_accuracy: 0.8081 - val_loss: 0.2963 - val_binary_accuracy: 0.8173
Epoch 10/15
1224/1224 [==============================] - 100s 82ms/step - loss: 0.3082 - binary_accuracy: 0.8091 - val_loss: 0.2814 - val_binary_accuracy: 0.8178
Epoch 11/15
1224/1224 [==============================] - 98s 80ms/step - loss: 0.3031 - binary_accuracy: 0.8099 - val_loss: 0.3085 - val_binary_accuracy: 0.8044
Epoch 12/15
1224/1224 [==============================] - 97s 79ms/step - loss: 0.3055 - binary_accuracy: 0.8095 - val_loss: 0.2822 - val_binary_accuracy: 0.8150
Epoch 13/15
1224/1224 [==============================] - 96s 78ms/step - loss: 0.3023 - binary_accuracy: 0.8097 - val_loss: 0.2811 - val_binary_accuracy: 0.8178
Epoch 14/15
1224/1224 [==============================] - 97s 79ms/step - loss: 0.2987 - binary_accuracy: 0.8089 - val_loss: 0.2761 - val_binary_accuracy: 0.8150
Epoch 15/15
ary_accuracy: 0.8103 - val_loss: 0.2757 - val_binary_accuracy: 0.8150
306/306 [==============================] - 14s 47ms/step
Epoch 1/15
1224/1224 [==============================] - 97s 78ms/step - loss: 1.4785 - binary_accuracy: 0.7838 - val_loss: 0.3767 - val_binary_accuracy: 0.8174
Epoch 2/15
1224/1224 [==============================] - 96s 78ms/step - loss: 0.3838 - binary_accuracy: 0.8036 - val_loss: 0.3282 - val_binary_accuracy: 0.8182
Epoch 3/15
1224/1224 [==============================] - 96s 79ms/step - loss: 0.3515 - binary_accuracy: 0.8083 - val_loss: 0.3090 - val_binary_accuracy: 0.8162
Epoch 4/15
1224/1224 [==============================] - 95s 78ms/step - loss: 0.3393 - binary_accuracy: 0.8083 - val_loss: 0.3039 - val_binary_accuracy: 0.8166
Epoch 5/15
1224/1224 [==============================] - 95s 77ms/step - loss: 0.3287 - binary_accuracy: 0.8097 - val_loss: 0.2989 - val_binary_accuracy: 0.8166
Epoch 6/15
1224/1224 [==============================] - 173s 141ms/step - loss: 0.3261 - binary_accuracy: 0.8106 - val_loss: 0.2933 - val_binary_accuracy: 0.8182
Epoch 7/15
1224/1224 [==============================] - 112s 92ms/step - loss: 0.3169 - binary_accuracy: 0.8105 - val_loss: 0.2959 - val_binary_accuracy: 0.8174
Epoch 8/15
1224/1224 [==============================] - 93s 76ms/step - loss: 0.3118 - binary_accuracy: 0.8112 - val_loss: 0.5570 - val_binary_accuracy: 0.7606
Epoch 9/15
1224/1224 [==============================] - 93s 76ms/step - loss: 0.3143 - binary_accuracy: 0.8100 - val_loss: 0.2851 - val_binary_accuracy: 0.8175
Epoch 10/15
1224/1224 [==============================] - 93s 76ms/step - loss: 0.3092 - binary_accuracy: 0.8108 - val_loss: 0.3108 - val_binary_accuracy: 0.8107
Epoch 11/15
1224/1224 [==============================] - 93s 76ms/step - loss: 0.3036 - binary_accuracy: 0.8098 - val_loss: 0.2795 - val_binary_accuracy: 0.8166
Epoch 12/15
1224/1224 [==============================] - 120s 98ms/step - loss: 0.3039 - binary_accuracy: 0.8112 - val_loss: 0.2906 - val_binary_accuracy: 0.8163
Epoch 13/15
1224/1224 [==============================] - 95s 77ms/step - loss: 0.3062 - binary_accuracy: 0.8095 - val_loss: 0.2828 - val_binary_accuracy: 0.8160
Epoch 14/15
1224/1224 [==============================] - 94s 77ms/step - loss: 0.3016 - binary_accuracy: 0.8101 - val_loss: 0.3071 - val_binary_accuracy: 0.8169
Epoch 15/15
1224/1224 [==============================] - 93s 76ms/step - loss: 0.2987 - binary_accuracy: 0.8111 - val_loss: 0.2760 - val_binary_accuracy: 0.8171
306/306 [==============================] - 14s 45ms/step
[0.8380044085335752, 0.8346133740758622, 0.8061091619960747, 0.8115301170473981, 0.8244381657633714]
Mean F1 Score: 0.8229
Standard Deviation of F1 Score: 0.0125
383/383 [==============================] - 110s 289ms/step - loss: 0.2751 - binary_accuracy: 0.8180
Test Loss: 0.27511295676231384, Test Accuracy: 0.8179942965507507
383/383 [==============================] - 17s 45ms/step
F1 Score: 0.8250103096832472
Recall score: 0.8090432907102142
Precision score: 0.8416202572218439
'''