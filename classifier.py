#!/usr/bin/env python
# run python3 classifier.py very_clean_label_data
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
from keras.applications import ResNet50
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
#from keras.utils import multi_gpu_model

from os import mkdir
import glob
import random
import shutil
import os
import csv
import sys
import pickle


train_data = sys.argv[1] 
#model_names = ['resnet', 'vgg', 'inception'] 
model_names = ['inception'] 

import bpython
bpython.embed(locals())

VAL_SIZE = 0.30
TEST_SIZE = 0.15

body_parts = ['arm', 'hand', 'foot',  'legs','fullbody', 
        'head','backside', 'torso', 'stake', 'plastic']
num_classes = len(body_parts)


data = []
labels = []


vgg_resnet_img_size = 224
inception_img_size = 299

batch_size = 32

for model_name in model_names:

    if model_name == 'vgg':
        inp = Input((vgg_resnet_img_size, vgg_resnet_img_size, 3))
        model = VGG16(include_top = False, weights='imagenet', 
                                        input_tensor = inp, input_shape = (vgg_resnet_img_size, vgg_resnet_img_size, 3),
                                        pooling = 'avg')

    if model_name == 'resnet':
        inp = Input((vgg_resnet_img_size, vgg_resnet_img_size, 3))
        model = ResNet50(include_top = False, weights='imagenet', 
                                        input_tensor = inp, input_shape = (vgg_resnet_img_size, vgg_resnet_img_size, 3),
                                        pooling = 'avg')
    if model_name == 'inception':
        inp = Input((inception_img_size, inception_img_size, 3))
        model = InceptionV3(include_top = False, weights='imagenet', 
                                        input_tensor = inp, input_shape = (inception_img_size, inception_img_size, 3), 
                                        pooling = 'avg')

    x = model.output
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inp, out)

    #model = multi_gpu_model(model, gpus = 4)

    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipvalue = 0.5)

    #inceptionV3_model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics=['acc'])

    not_found = 0
    #if model_name != 'vgg': # becase for vgg the same data that was used for resnet will work
    data = []
    labels = []

    with open(train_data, 'r') as file_:
        csv_reader = csv.reader(file_, delimiter = ":")
        for row in csv_reader:
            tag = row[1].strip()
            if tag in body_parts:
                try:
                    if model_name == 'inception':
                        img = image.load_img(row[0].strip(), 
                                target_size = (inception_img_size, inception_img_size, 3), grayscale = False)
                    else:
                        img = image.load_img(row[0].strip(), 
                                target_size = (vgg_resnet_img_size, vgg_resnet_img_size, 3), grayscale = False)

                    img = image.img_to_array(img)
                    img = img/255
                    data.append(img)
                    labels.append(body_parts.index(tag))

                except:
                    not_found += 1

    data = np.array(data)
    labels = to_categorical(np.array(labels), num_classes = num_classes)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, 
             test_size=0.3)



    checkpoint = ModelCheckpoint(model_name + '_5000_random_epoch_-{epoch:03d}-_acc_{acc:03f}-_val_acc_{val_acc:.5f}.h5', verbose=1, monitor='val_acc', save_best_only=True, mode='auto')  
     
    train_datagen = ImageDataGenerator()#rescale=1./255)
    val_datagen = ImageDataGenerator()#rescale=1./255)


    history = model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size), 
                                                        steps_per_epoch = len(X_train) // batch_size,
                                                        validation_data=val_datagen.flow(X_test,
                                                            y_test, batch_size=batch_size), 
                                                        validation_steps=(len(X_test))//batch_size,
                                                        callbacks=[checkpoint],
                                                        epochs = 100, verbose = 1) 


