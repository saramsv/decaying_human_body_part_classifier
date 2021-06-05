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
dir_info = os.walk('models/' + model_dir_name)
for root, dirs, models in dir_info:
    imodel_names = [model_dir_name for mo in range(len(models))]
    model_name = model_dir_name

    test_data = []
    img_names = []

    not_found = 0

    df = pd.read_csv(test_csv, names = ['path', 'gt'], sep = ':')

    for path in df['path']:
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
        except:
            not_found += 1

    test = np.array(test_data)
    print('done with data')

    for m in model_names:
        #model = load_model('models/'+ model_dir_name + '/' + m)
        model = load_model(model_dir_name + '/' + m)
        '''
        prediction = model.predict(test)
        pred_classes = prediction.argmax(axis=-1)
        conf = prediction.max(axis=-1)


        print(m)
        for i, label in enumerate(list(pred_classes)):
            print(img_names[i]+":", body_parts[label], conf[i])

        '''
        ####### top 1 ######
        print("top 1:")
        #print(m)
        gt = list(df['gt'].values)

        prediction = model.predict(test)
        pred_classes = prediction.argmax(axis=-1)


        pred = list(pred_classes)
        pred = [body_parts[x] for x in pred]
        CM = confusion_matrix(gt, pred, labels= body_parts)
        TP = CM.diagonal()
        FN = np.sum(CM, axis = 1) - TP
        FP = np.sum(CM, axis = 0) - TP
        AP = TP/(TP + FP)
        print("AP: ", AP)
        mAP = np.mean(AP)
        print("mAP: ", mAP)

        recall = TP/(TP + FN)
        print("recall: ", recall)
        mrecall = np.mean(recall)
        print("mrecall: ", mrecall)

        #print(confusion_matrix(gt, pred, labels= body_parts))
        #print(accuracy_score(gt, pred))

        #print("moedel: {}, mAP: {}, mrecall: {}, aac: {}".format(m, mAP, mrecall,
        #                                                accuracy_score(gt, pred)))
        ###### top 3 ######
        print("top k:")
        #print(m)
        pred = []
        k = 3
        for index, p in enumerate(prediction): #p is for each image
            preds = p.argsort()[0-k:] # the top k confident predictions
            added = False
            for p in preds:
                predicted = body_parts[p]
                if predicted == gt[index]: # if one of the top k is the right prediction
                    pred.append(predicted)
                    added = True
                    break
            if added == False:        
                pred.append(body_parts[preds[-1]]) # the most confident

        CM = confusion_matrix(gt, pred, labels= body_parts)
        TP = CM.diagonal()
        FN = np.sum(CM, axis = 1) - TP
        FP = np.sum(CM, axis = 0) - TP
        AP = TP/(TP + FP)
        print("AP: ", AP)
        mAP = np.mean(AP)
        print("mAP: ", mAP)

        recall = TP/(TP + FN)
        print("recall: ", recall)
        mrecall = np.mean(recall)
        print("mrecall: ", mrecall)

        #print(confusion_matrix(gt, pred, labels= body_parts))
        #print(accuracy_score(gt, pred))
        #print("moedel: {}, mAP: {}, mrecall: {}, aac: {}".format(m, mAP, mrecall,
        #                                                accuracy_score(gt, pred)))
