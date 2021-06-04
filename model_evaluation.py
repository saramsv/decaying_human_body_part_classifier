#!/usr/bin/env python

## run : python3 predict_bodypart.py gt_file_name
#gt_file_name has a path to an image and the grount truth label seperated by ":" in each line

import os
import cv2
import csv
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.models import load_model
from keras.preprocessing import image
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Helper function
def conf_matrix(f, gt, pred, body_parts):
    CM = confusion_matrix(gt, pred, labels=body_parts)
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
    #print(accuracy_score(gt, pred))
    #print("moedel: {}, mAP: {}, mrecall: {}, aac: {}".format(m, mAP, mrecall, accuracy_score(gt, pred)))
    print("mAP: {}, mrecall: {}, aac: {}\n".format(mAP, mrecall, accuracy_score(gt, pred)))
    f.write("mAP: {}, mrecall: {}, aac: {}\n\n".format(mAP, mrecall, accuracy_score(gt, pred)))

    

# MAIN FUNC
if __name__ == "__main__":
    
    inception_img_size = 299
    vgg_resnet_img_size = 224

    test_csv = sys.argv[1]

    body_parts = ['arm', 'hand', 'foot', 'legs','fullbody', 
            'head','backside',  'torso', 'stake', 'plastic']


    dir_names = ['resnet', 'vgg', 'inception']
    model_dir_name = 'models'
    
    f = open("eval_top3.res", "w")

    for name in dir_names:
        model_type = name
        dir_info = os.walk(model_dir_name + "/" + model_type)
        #print("Dir: {}".format(model_dir_name + "/" + model_type))

        for root, dirs, models in dir_info:
            test_data = []
            img_names = []

            not_found = 0

            df = pd.read_csv(test_csv, names = ['path', 'gt'], sep = ':')

            print("loading data")
            for path in tqdm(df['path'], total=len(df)):
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
                    # break
                except:
                    not_found += 1

            test = np.array(test_data)

            for m in models:
                print("\n{}".format(model_dir_name + "/" + model_type + '/' + m))
                f.write("{}\n".format(model_dir_name + "/" + model_type + '/' + m))

                model = load_model(model_dir_name + "/" + model_type + '/' + m, compile=False)
                
                ####### top 1 ######
                f.write("top 1:\n")
                #print("top 1:")
                #print(m)
                gt = list(df['gt'].values)

                prediction = model.predict(test)
                pred_classes = prediction.argmax(axis=-1)

                pred = list(pred_classes)
                pred = [body_parts[x] for x in pred]
                
                conf_matrix(f, gt, pred, body_parts)

                ###### top 3 ######
                #print("top k = 3:")
                f.write("top 3:\n")
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

                conf_matrix(f, gt, pred, body_parts)
    f.close()