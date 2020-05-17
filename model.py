import io
import csv
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Flatten, Activation, Dense, Lambda, Dropout
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

''' helper functions for data processing '''
def flip_n_invert(image_list,angles):
    '''return flipped images from list and inverting angles for mirrored operation'''
    # initializing empty lists
    inv_images = []
    inv_angles = []
    for image in image_list:
        inv_images.append(cv2.flip(image,1)) # flipping horizontally
    for angle in angles:
        inv_angles.append(-1 * angle)
    return inv_images,inv_angles

def preprocess(image):
    '''preprocess image: convert bgr to rgb image'''
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

'''Importing csv data'''
lines = []
with open('./refined_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = [] # images list
angles = [] # steering angles

for line in lines:
    image_center = preprocess(cv2.imread(line[0])) # center camera image
    image_left = preprocess(cv2.imread(line[1])) # left camera image
    image_right = preprocess(cv2.imread(line[2])) # right camera image
    images.append(image_center)
    images.append(image_left)
    images.append(image_right)
    angle_center = float(line[3]) # steering angle recorded
    angle_left = float(line[3]) + 0.2 # positive offset for left image
    angle_right = float(line[3]) - 0.2 # negative offset for right image
    angles.append(angle_center)
    angles.append(angle_left)
    angles.append(angle_right)

''' augmenting image set for generalizing the dataset (to get rid of left turn bias) '''
center_images,steering_angles = flip_n_invert(images,angles) # flipping images and corresponding angles
images.extend(center_images)
angles.extend(steering_angles)

X_train,X_valid,y_train,y_valid = train_test_split(images,angles,test_size=0.3)  # training and validation set split

def generator(X,y, batch_size=32):
    ''' generator function for batch processing images and angles for neural network training '''
    num_samples = len(X)
    while 1: # infinite loop to go till the end of the list
        shuffle()
        for offset in range(0,num_samples,batch_size): # batch processing
            batch_x,batch_y = X[offset:offset+batch_size],y[offset:offset+batch_size]
        
        X_batch = np.array(batch_x)
        y_batch = np.array(batch_y)
        yield shuffle(X_batch,y_batch)

# batch size for the dataset
batch_size = 128

train_generator = generator(X_train,y_train,batch_size=batch_size) # training set generator
valid_generator = generator(X_valid,y_valid,batch_size=batch_size) # validation set generator

# model architecture, resembles NVIDIA architecture
model = Sequential()
model.add(Cropping2D(((70,25),(0,0)),input_shape=(160,320,3))) # cropping image
model.add(Lambda(lambda x: (x/255) - 0.5)) # normalizing
model.add(Conv2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(64,3,3, activation='relu'))
model.add(Conv2D(64,3,3, activation='relu'))
model.add(Flatten())    
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1)) # single output for the steering angle

''' Model compilation and running '''
# callbacks list to prevent overfitting and getting intermediate models for checking best driving
my_callbacks = [EarlyStopping(monitor='val_loss',patience=2),
    ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5')]
# using mean squared error loss model and adam optimizer to get the most accurate steering angle prediction for the image
model.compile(loss='mse', optimizer='adam') 
# model prediction using generators
history_object = model.fit_generator(train_generator,steps_per_epoch=math.ceil(len(X_train)/batch_size),
validation_data=valid_generator,validation_steps=math.ceil(len(X_valid)/batch_size),epochs=20,verbose=1,callbacks=my_callbacks)
# model is saved as model.h5
model.save('model.h5')