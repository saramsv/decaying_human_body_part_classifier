#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dropout, Dense, Input
from keras.models import Sequential
from keras.models import Model
import keras.backend as K
from keras.utils import to_categorical
import pickle
from keras.applications import VGG16
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model

from os import mkdir
import glob
import random
import shutil
import os
import csv
import sys
import pickle


VAL_SIZE = 0.30
TEST_SIZE = 0.15

body_parts = ['arm', 'hand', 'foot',  'legs','fullbody', 'head','butt', 'torso', 'stake', 'plastic']
num_classes = len(body_parts)


data = []
labels = []

clusters = sys.argv[1]#'clean_labeld_clusters'#
#img_size = 299
#299 is for inception model
img_size = 224
batch_size = 32

inp = Input((img_size, img_size, 3))

#inceptionV3_model = InceptionV3(include_top = False, weights='imagenet', 
#                                input_tensor = inp, input_shape = (img_size, img_size, 3), pooling = 'avg')
model = VGG16(include_top = False, weights='imagenet', 
                                input_tensor = inp, input_shape = (img_size, img_size, 3), pooling = 'avg')
x = model.output
x = Dense(256, activation='relu')(x)
x = Dropout(0.1)(x)
out = Dense(num_classes, activation='softmax')(x)

model = Model(inp, out)
model.summary()

model = multi_gpu_model(model, gpus = 4)

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipvalue = 0.5)

#inceptionV3_model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics=['acc'])

'''
if os.path.isfile('data.pkl') and os.path.isfile('labels.pkl'):
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('labels.pkl', 'rb') as f:
        labels = pickle.load(f)
else:
'''
not_found = 0
with open(clusters, 'r') as file_:
    csv_reader = csv.reader(file_, delimiter = ":")
    for row in csv_reader:
        tag = row[1].strip()
        if tag in body_parts:
            try:
                img = image.load_img(row[0].strip(), target_size = (img_size, img_size, 3), grayscale = False)
                img = image.img_to_array(img)
                img = img/255
                data.append(img)
                labels.append(body_parts.index(tag))
                

            except:
                not_found += 1

data = np.array(data)
labels = to_categorical(np.array(labels), num_classes = num_classes)
'''
try:
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)
    with open('labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
except:
    print("Could not write the data in files")
'''
#labels= np.array(labels)

print("Done with reading in the data")
print("Data size is {} and {}".format(data.shape, labels.shape))
X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=42, test_size=0.3)


#earlyStopping = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='min')

checkpoint = ModelCheckpoint('test_model_epoch_-{epoch:03d}-_acc_{acc:03f}-_val_acc_{val_acc:.3f}.h5', verbose=1, monitor='val_acc', save_best_only=True, mode='auto')  
 

'''

#data generator for train and val data
train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.2,
                                   height_shift_range=0.2, 
                                   fill_mode='nearest', horizontal_flip=True,
                                   vertical_flip=True, rescale=1./255, 
                                   preprocessing_function=None)

'''
train_datagen = ImageDataGenerator()#rescale=1./255)
val_datagen = ImageDataGenerator()#rescale=1./255)
#train_datagen.fit(X_train)
#val_datagen.fit(X_test)



history = model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size), 
                                                    steps_per_epoch = len(X_train) // batch_size,
                                                    validation_data=val_datagen.flow(X_test, y_test, batch_size=batch_size), 
                                                    validation_steps=(len(X_test))//batch_size,
                                                    callbacks=[checkpoint],
                                                    epochs = 100, verbose = 1) 

'''
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    steps_per_epoch = len(X_train) // batch_size,
                    validation_steps=(len(X_test)*0.2)//batch_size,
                    callbacks=[checkpoint],
                    epochs = 2, verbose = 1) 

'''
'''
#history = inceptionV3_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save('VGG16model_retrained_moredata.h5')
with open('history', 'wb') as file_:
    pickle.dump(history.history, file_)

fig1 = plt.figure()
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves :CNN',fontsize=16)
fig1.savefig('more_data_loss_cnn.png')
plt.show()

fig2=plt.figure()
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves : CNN',fontsize=16)
fig2.savefig('more_data_accuracy_cnn.png')
plt.show()

'''

