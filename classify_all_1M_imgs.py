#!/usr/bin/env python

## run : python3 predict_bodypart.py file_name
#each line in file_name is a path 

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import csv
import sys
import os
from tensorflow.keras.preprocessing import image
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


inception_img_size = 299
vgg_resnet_img_size = 224
batch_size = 2

test_csv = sys.argv[1]

body_parts = ['arm', 'hand', 'foot', 'legs','fullbody', 'head','backside',  'torso', 'stake', 'plastic']

model_name = 'inception_epoch_-044-_acc_0.995226-_val_acc_0.96135.h5'#'models/inception_epoch_-058-_acc_0.999475-_val_acc_0.94976.h5'#'inception_10000_epoch_-118-_acc_0.999569-_val_acc_0.98315.h5'
model_type = 'inception'
#model = load_model("models/" + model_type + '/' + model_name)
model = load_model(model_name)

not_found = 0


df = pd.read_csv(test_csv, names = ['path'])
print("The results will be saved in", test_csv + '_preds')
f = open(test_csv + '_preds', 'w')

for path in df['path']:
    test_data = []
    img_names = []
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
        img_names.append(path)
        test = np.array(test_data)

        prediction = model.predict(test)
        pred_classes = prediction.argmax(axis=-1)
        conf = prediction.max(axis=-1)

        
        for i, label in enumerate(list(pred_classes)):
            f.write("{}:{}:{:.2f}\n".format(img_names[i], body_parts[label],conf[i]*100))
            #print(img_names[i]+ ":", body_parts[label],":", conf[i])
    except:
        not_found += 1
f.close()
