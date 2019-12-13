#!/usr/bin/env python
# coding: utf-8

# In[ ]:

## run : python3 predict_bodypart.py file_name
#file name is each line a path and the : alak for the label
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
from keras.models import load_model
import pickle
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import os
import csv
import sys

test_data = []
test_labels = []
img_names = []
#img_size = 299
img_size = 224
batch_size = 2

test_csv = sys.argv[1]

body_parts = ['arm', 'hand', 'foot', 'legs','fullbody', 'head','butt',  'torso', 'stake', 'plastic', 'alak']
num_classes = len(body_parts)

model = load_model('/models/vgg_model_moredata_epoch_-007-_acc_0.982805-_val_acc_0.982.h5')
model.summary()

not_found = 0
with open(test_csv, 'r') as file_:
    csv_reader = csv.reader(file_, delimiter = ":")
    for row in csv_reader:
        tag = row[1].strip()
        if tag in body_parts:
            try:
                img = image.load_img(row[0].strip(), target_size = (img_size, img_size, 3), grayscale = False)
                img = image.img_to_array(img)
                img = img/255
                img_names.append(row[0])
                test_data.append(img)
                test_labels.append(body_parts.index(tag))
            except:
                not_found += 1


test = np.array(test_data)
l = np.array(test_labels)

#model = load_model('model_4donors_epoch_-080-_acc_1.000000-_val_acc_0.931818.h5')

prediction = model.predict(test)
pred_classes = prediction.argmax(axis=-1)
conf = prediction.max(axis=-1)


for i, label in enumerate(list(pred_classes)):
    print(img_names[i]+":", body_parts[label], conf[label])

