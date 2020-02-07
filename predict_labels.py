#!/usr/bin/env python

## run : python3 predict_labels.py file_name
#(each line in file_name is a path_to_img)

import cv2
import numpy as np
from keras.models import load_model
import keras
import csv
import sys
import os
from keras.preprocessing import image
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


inception_img_size = 299
vgg_resnet_img_size = 224
batch_size = 2

test_csv = sys.argv[1]

body_parts = ['arm', 'hand', 'foot', 'legs','fullbody', 'head','backside',  'torso', 'stake', 'plastic']

model_name = 'inception_10000_epoch_-118-_acc_0.999569-_val_acc_0.98315.h5'
model_type = 'inception'
model = load_model("models/" + model_type + '/' + model_name)

model_copy = keras.models.clone_model(model)
model_copy.set_weights(model.get_weights())
model_copy.layers.pop()
model_copy.outputs = [model_copy.layers[-1].output]
model_copy.layers[-1].outbound_nodes = []


not_found = 0


df = pd.read_csv(test_csv, names = ['path'])

for path in df['path']:
    test_data = []
    #img_names = []
    try:
        if model_type == 'resnet' or model_type == 'vgg':
            img = image.load_img(path.strip(), 
                    target_size = (vgg_resnet_img_size, vgg_resnet_img_size, 3), 
                    grayscale = False)
        elif model_type == 'inception':
            img = image.load_img(path.strip(), 
                    target_size = (inception_img_size, inception_img_size, 3), 
                    grayscale = False)
        img = image.img_to_array(img)
        img = img/255
        test_data.append(img)
        #img_names.append(path)
        test = np.array(test_data)

        prediction = model.predict(test)


        pred_classes = prediction.argmax(axis=-1)
        conf = prediction.max(axis=-1)
        print(conf)
        if conf > 0.99:
            for i, label in enumerate(list(pred_classes)):
                print(path + ":", body_parts[label]+ ":" + str(conf[i]))
        else:
            row = []
            row.append(path)
            prediction = model_copy.predict(test)
            row.extend(list(prediction))
            print(row)
            #print(path + ":"+ str(list(prediction)))

    except:
        not_found += 1
