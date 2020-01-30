#!/usr/bin/env python

## run : python3 predict_bodypart.py file_name
#file name is each line a path and then : the grount truth label

import cv2
import numpy as np
from keras.models import load_model
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

dir_names = ['resnet', 'vgg', 'inception']
model_dir_name = dir_names[2]
dir_info = os.walk(model_dir_name)
model_names = ['inception_10000_epoch_-118-_acc_0.999569-_val_acc_0.98315.h5']
model_name = model_dir_name


not_found = 0

df = pd.read_csv(test_csv, names = ['path'])
model = load_model(model_dir_name + '/' + model_names[0])
for path in df['path']:
    test_data = []
    img_names = []
    try:
        if model_name == 'resnet' or model_name == 'vgg':
            img = image.load_img(path.strip(), 
                    target_size = (vgg_resnet_img_size, vgg_resnet_img_size, 3), 
                    grayscale = False)
        elif model_name == 'inception':
            img = image.load_img(path.strip(), 
                    target_size = (inception_img_size, inception_img_size, 3), 
                    grayscale = False)
        img = image.img_to_array(img)
        img = img/255
        test_data.append(img)
        img_names.append(path)
        test = np.array(test_data)

        prediction = model.predict(test)
        pred_classes = prediction.argmax(axis=-1)
        conf = prediction.max(axis=-1)

        
        for i, label in enumerate(list(pred_classes)):
            print(img_names[i]+":", body_parts[label], conf[i])
    except:
        not_found += 1
        #print(path)
