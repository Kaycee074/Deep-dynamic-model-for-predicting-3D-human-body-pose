#%% 
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 17:37:45 2021

@author: 15186
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Reshape, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Lambda, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, load_model

# Loading training data 

print('Loading Train Data')
x_train = np.load('videoframes_clips_train.npy', mmap_mode='c')
print('Done')

# Loading test data 

print('Loading Test Data')
x_test = np.load('videoframes_clips_valid.npy', mmap_mode='c')
print('Done')

# Loading train labels

print('Loading Train Labels')
y_train = np.load('joint_3d_clips_train.npy', mmap_mode='c')
print('Done')

# Loading test labels

print('Loading Test Labels')
y_test = np.load('joint_3d_clips_valid.npy', mmap_mode='c')
print('Done')

#%% Model Training 

# Model Description

# Preprocessing Layer: Normalizing the input by dividing by 255.0
# Layer 1: Conv Layer with 16 filters of size 7x7 with 'Relu' activation function
# Layer 2: Max Pooling Layer of size 2x2 with strides 2
# Layer 3: Conv Layer with 32 filters of size 5x5 with 'Relu' activation function
# Layer 4: Max Pooling Layer of size 2x2 with strides 2
# Layer 5: Conv Layer with 32 filters of size 5x5 with 'Relu' activation function
# Layer 6: Max Pooling Layer of size 3x3 with strides 3
# Layer 7: Conv Layer with 64 filters of size 3x3 with 'Relu' activation function
# Layer 8: Max Pooling Layer of size 3x3 with strides 5
# Layer 9: LSTM Layer with output nodes 1024
# Layer 10: Fully connected layer with output nodes 256 with 'Relu' activation function
# Output Layer: Fully connected layer with output nodes 51 with 'Linear' activation function 
# Reshaping Layer: Reshaping the output compatible to provided output labels


# Model Parameters

input_shape_to_cnn = (x_train.shape[2],x_train.shape[3],x_train.shape[4]) # (224,224,3)
output_shape = (y_train.shape[1],y_train.shape[2],y_train.shape[3]) # (8,17,3)

# Model Hyperparameters

minibatch_size = 20
epoch = 20
learning_rate = 0.001

# Computing MPJME

def MPJPE(y_true, y_pred):
    error = tf.math.multiply(1000.0,tf.reduce_mean(tf.norm(y_pred-y_true, ord='euclidean', axis=3)))

    return error

# Defining CNN 

cnn = Sequential([
    Conv2D(16, (7,7), activation = 'relu', input_shape = input_shape_to_cnn, kernel_regularizer=regularizers.L2(0.001) ),
    BatchNormalization(),
    #Dropout(0.2),
    MaxPooling2D(pool_size=(2,2), strides=2),
    
    Conv2D(32, (5,5), activation = 'relu', kernel_regularizer=regularizers.L2(0.001)),
    BatchNormalization(),
    #Dropout(0.2),
    MaxPooling2D(pool_size=(2,2), strides=2),
    
    Conv2D(32, (5,5), activation = 'relu', kernel_regularizer=regularizers.L2(0.001)),
    BatchNormalization(),
    #Dropout(0.2),
    MaxPooling2D(pool_size=(3,3), strides=3),
    
    Conv2D(64, (3,3), activation = 'relu', kernel_regularizer=regularizers.L2(0.001)), 
    BatchNormalization(),
    #Dropout(0.2),
    MaxPooling2D(pool_size=(3,3), strides=5),
    ])

# Defining LSTM

lstm = Sequential([
    LSTM(units=1024, return_sequences=True)
    ])

# Defining MLP

mlp = Sequential([
    Dense(256, activation='relu', kernel_regularizer = regularizers.L2(0.01)),
    Dropout(0.2),
    Dense(51, activation='linear', kernel_regularizer = regularizers.L2(0.01))
    ])

# Main Model 

model = Sequential([
    Lambda(lambda x: tf.divide(tf.cast(x,tf.float32),255.0)),
    TimeDistributed(cnn),
    
    TimeDistributed(Flatten()),
    lstm,

    TimeDistributed(mlp),
    Reshape(output_shape)
])

model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mse', MPJPE],) 
    
history = model.fit(x_train, y_train, epochs=epoch, batch_size = minibatch_size, verbose=1, validation_data=(x_test, y_test))    
# history = model.fit(x_train[1:100], y_train[1:100], epochs=epoch, batch_size = minibatch_size, verbose=1, validation_data=(x_test[1:20], y_test[1:20]))

#%% Saving the Model and per epoch data

# Saving the Model

model.save("Human_3D_Pose_Model.h5")

dependencies = {
    'MPJPE': MPJPE
}

# Saving Loss and MPJME by epochs

np.save('per_epoch_error.npy',history.history)

#%% Plotting Loss and MPJPE by epochs

# Loading Loss and MPJME by epochs

per_epoch_error = np.load('per_epoch_error.npy',allow_pickle='TRUE').item()

train_loss_per_epoch = per_epoch_error["loss"]
test_loss_per_epoch = per_epoch_error["val_loss"]
train_MPJPE_per_epoch = per_epoch_error["MPJPE"]
test_MPJPE_per_epoch = per_epoch_error["val_MPJPE"]

# Evaluating Loss by Epochs

plt.plot(train_loss_per_epoch, label ="Train Loss")
plt.plot(test_loss_per_epoch, label ="Test Loss")
plt.legend()
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.show()

# Evaluating MPJME by Epochs

plt.plot(train_MPJPE_per_epoch, label ="Train MPJPE")
plt.plot(test_MPJPE_per_epoch, label ="Test MPJPE")
plt.legend()
plt.xlabel('Number of epochs')
plt.ylabel('Mean Per Joint Position Error (in mm)')
plt.show()

#%% Model Load and Evaluation Script

# Loading the Model

model = load_model("Human_3D_Pose_Model.h5",custom_objects=dependencies)

# Evaluating the Model

train_loss,train_mse, train_MPJPE = model.evaluate(x_train, y_train)
test_loss, test_mse, test_MPJPE = model.evaluate(x_test, y_test)

# Printing Performance Metrics

print("Training Loss: ", train_loss)
print("Training MPJPE: ", train_MPJPE," mm")
print("Testing Loss: ", test_loss)
print("Testing MPJPE: ", test_MPJPE," mm")


