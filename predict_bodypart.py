#!/usr/bin/env python

## run : python3 predict_bodypart.py file_name
#file name is each line a path and then : the grount truth label

import cv2
import numpy as np
from keras.models import load_model
import csv
import sys
from keras.preprocessing import image
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


inception_img_size = 299
vgg_resnet_img_size = 224
batch_size = 2

test_csv = sys.argv[1]

body_parts = ['arm', 'hand', 'foot', 'legs','fullbody', 'head','backside',  'torso', 'stake', 'plastic']

'''
models = ['models/resnet_model_epoch_-042-_acc_0.999860-_val_acc_0.98716.h5',
        'models/vgg_model_epoch_-006-_acc_0.860008-_val_acc_0.96641.h5', 
        'models/inception_model_epoch_-131-_acc_0.999860-_val_acc_0.98848.h5']
model_names = ['resnet', 'vgg', 'inception']
## after self training and more data
models = ['models/self_trained_resnet_model_epoch_-098-_acc_0.999346-_val_acc_0.98339.h5',
        'models/self_trained_vgg_model_epoch_-091-_acc_0.999782-_val_acc_0.98039.h5', 
        'models/self_trained_inception_model_epoch_-050-_acc_0.999565-_val_acc_0.98141.h5']
models = ['models/inception_3000_epoch_-125-_acc_1.000000-_val_acc_0.97430.h5',
        'models/inception_7000_epoch_-148-_acc_0.999791-_val_acc_0.97693.h5',
        'models/inception_model_epoch_-131-_acc_0.999860-_val_acc_0.98848.h5',
        'models/self_trained_inception_model_epoch_-050-_acc_0.999565-_val_acc_0.98141.h5',
        'models/resnet_3000_epoch_-075-_acc_1.000000-_val_acc_0.95444.h5', 
        'models/resnet_7000_epoch_-106-_acc_0.999791-_val_acc_0.96662.h5',
        'models/resnet_model_epoch_-042-_acc_0.999860-_val_acc_0.98716.h5',
        'models/self_trained_resnet_model_epoch_-098-_acc_0.999346-_val_acc_0.98339.h5',
        'models/vgg_3000_epoch_-123-_acc_1.000000-_val_acc_0.95911.h5',
        'models/vgg_7000_epoch_-083-_acc_0.999582-_val_acc_0.96809.h5',
        'models/vgg_model_epoch_-006-_acc_0.860008-_val_acc_0.96641.h5',
        'models/self_trained_vgg_model_epoch_-091-_acc_0.999782-_val_acc_0.98039.h5',
        'models/inception_5000_random_epoch_-034-_acc_1.000000-_val_acc_0.97165.h5',
        'models/inception_ut01-13d_gtplus_hand_epoch_-016-_acc_0.999721-_val_acc_0.98022.h5']

model_names = ['inception', 'inception', 'inception', 'inception', 'resnet', 'resnet','resnet','resnet', 'vgg','vgg','vgg','vgg', 'inception', 'inception'] 
'''

models = ['models/self_trained_resnet_model_epoch_-098-_acc_0.999346-_val_acc_0.98339.h5',
        'models/vgg_3000_epoch_-123-_acc_1.000000-_val_acc_0.95911.h5',
        'models/vgg_7000_epoch_-083-_acc_0.999582-_val_acc_0.96809.h5',
        'models/vgg_model_epoch_-006-_acc_0.860008-_val_acc_0.96641.h5',
        'models/self_trained_vgg_model_epoch_-091-_acc_0.999782-_val_acc_0.98039.h5',
        'models/inception_5000_random_epoch_-034-_acc_1.000000-_val_acc_0.97165.h5',
        'models/inception_ut01-13d_gtplus_hand_epoch_-016-_acc_0.999721-_val_acc_0.98022.h5']
model_names = ['resnet', 'vgg','vgg','vgg','vgg', 'inception', 'inception']

for index, m in enumerate(models):
    model_name = model_names[index]
    model = load_model(m)
    test_data = []
    img_names = []
    #model.summary()

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
    #print("AP: ", AP)
    mAP = np.mean(AP)
    #print("mAP: ", mAP)

    recall = TP/(TP + FN)
    #print("recall: ", recall)
    mrecall = np.mean(recall)
    #print("mrecall: ", mrecall)

    #print(confusion_matrix(gt, pred, labels= body_parts))
    #print(accuracy_score(gt, pred))

    print("moedel: {}, mAP: {}, mrecall: {}, aac: {}".format(m, mAP, mrecall,
                                                    accuracy_score(gt, pred)))
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
    #print("AP: ", AP)
    mAP = np.mean(AP)
    #print("mAP: ", mAP)

    recall = TP/(TP + FN)
    #print("recall: ", recall)
    mrecall = np.mean(recall)
    #print("mrecall: ", mrecall)

    #print(confusion_matrix(gt, pred, labels= body_parts))
    #print(accuracy_score(gt, pred))
    print("moedel: {}, mAP: {}, mrecall: {}, aac: {}".format(m, mAP, mrecall,
                                                    accuracy_score(gt, pred)))
