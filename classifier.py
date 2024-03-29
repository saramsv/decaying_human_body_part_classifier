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
from keras.applications import VGG16
from keras.applications import ResNet50
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import os
import csv
import sys


train_data = sys.argv[1] 



body_parts = ['arm', 'hand', 'foot',  'legs','fullbody', 
        'head','backside', 'torso', 'stake', 'plastic']
num_classes = len(body_parts)


data = []
labels = []


vgg_resnet_img_size = 224
inception_img_size = 299

batch_size = 32

model_names = ['inception']#['resnet', 'vgg', 'inception'] 
for model_name in model_names:

    not_found = 0
    data = []
    labels = []
    #os.mkdir(model_name)
    with open(train_data, 'r') as file_:
        csv_reader = csv.reader(file_, delimiter = ":")
        for row in csv_reader:
            tag = row[1].strip()
            if tag in body_parts:
                try:
                    if model_name == 'inception':
                        img = image.load_img(row[0].strip(), 
                                target_size = (inception_img_size, 
                                    inception_img_size, 3), grayscale = False)
                    else:
                        img = image.load_img(row[0].strip(), 
                                target_size = (vgg_resnet_img_size, 
                                    vgg_resnet_img_size, 3), grayscale = False)

                    img = image.img_to_array(img)
                    img = img/255
                    data.append(img)
                    labels.append(body_parts.index(tag))

                except:
                    not_found += 1
    sample_sizes = [len(data)]

    for sample_size in sample_sizes:
        if model_name == 'vgg':
            inp = Input((vgg_resnet_img_size, vgg_resnet_img_size, 3))
            model = VGG16(include_top = False, weights='imagenet', 
                                            input_tensor = inp, 
                                            input_shape = (vgg_resnet_img_size, 
                                                vgg_resnet_img_size, 3),
                                            pooling = 'avg')

        if model_name == 'resnet':
            inp = Input((vgg_resnet_img_size, vgg_resnet_img_size, 3))
            model = ResNet50(include_top = False, weights='imagenet', 
                                            input_tensor = inp, 
                                            input_shape = (vgg_resnet_img_size,
                                                vgg_resnet_img_size, 3),
                                            pooling = 'avg')
        if model_name == 'inception':
            inp = Input((inception_img_size, inception_img_size, 3))
            model = InceptionV3(include_top = False, weights='imagenet', 
                                            input_tensor = inp, 
                                            input_shape = (inception_img_size, 
                                                inception_img_size, 3), 
                                            pooling = 'avg')
        x = model.output
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.1)(x)
        out = Dense(num_classes, activation='softmax')(x)

        model = Model(inp, out)
        sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipvalue = 0.5)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

        model.load_weights('models/inception_epoch_-058-_acc_0.999475-_val_acc_0.94976.h5')
        model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics=['acc'])
        d = data[0:sample_size]
        l = labels[0:sample_size]
        data1 = np.array(d)
        labels1 = to_categorical(np.array(l), num_classes = num_classes)

        X_train, X_test, y_train, y_test = train_test_split(data1, labels1, 
                 test_size=0.3)

        #checkpoint = ModelCheckpoint(model_name + '/' + str(sample_size) + 
        #'_epoch_-{epoch:03d}-_acc_{acc:03f}-_val_acc_{val_acc:.5f}.h5', 
        #verbose=1, monitor='val_acc', save_best_only=True, mode='auto')  
        checkpoint = ModelCheckpoint(model_name + '_epoch_-{epoch:03d}-_acc_{acc:03f}-_val_acc_{val_acc:.5f}.h5', 
                verbose=1, monitor='val_acc', save_best_only=True, mode='auto')  
         
        train_datagen = ImageDataGenerator()#rescale=1./255)
        val_datagen = ImageDataGenerator()#rescale=1./255)

        history = model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size)
                ,steps_per_epoch = len(X_train) // batch_size
                ,validation_data=val_datagen.flow(X_test,y_test, batch_size=batch_size), 
                validation_steps=(len(X_test))//batch_size
                ,callbacks=[checkpoint, es]
                ,epochs = 200, verbose = 1) 
        import bpython
        bpython.embed(locals())

        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('acc.png')
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('loss')

        import bpython
        bpython.embed(locals())


