# only use it to generate the resnet features then use the other script to convert to pca
#python3 path2embeding.py --img_path data/some_paths --weight_type pt  > resnet_feautres_filename
#then clean the [] and , from the all_embedings.csv
import keras
from keras.applications import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications import VGG19
from keras.applications.resnet50 import preprocess_input
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.callbacks import TensorBoard
from keras.layers import Dropout, Dense, Input
from keras import Model
import numpy as np
from keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import cv2
import sys
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type = str)
parser.add_argument('--weight_type', type = str)

args = parser.parse_args()

imgs_path = args.img_path
weight_type = args.weight_type


img_size = 224
resnet_weigth_path = '../ImageSimilarityMultiMethods/data/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
#resnet_weigth_path = 'resnet_model_epoch_-042-_acc_0.999860-_val_acc_0.98716.h5'
fine_tuned_resnet_weight_path = 'ResNet50/logs/ft-41-0.87.hdf5'

clustering_model = Sequential()

if weight_type == 'pt': # this is for pre_trained
    
    '''
    clustering_model = load_model('models/resnet_model_epoch_-042-_acc_0.999860-_val_acc_0.98716.h5')
    inp = Input((224, 224, 3))
    model = ResNet50(include_top=False, weights='imagenet',
                                        pooling = 'avg')

    x = model.output
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)
    out = Dense(10, activation='softmax')(x)

    clustering_model = Model(inp, out)
    clustering_model.load_weights('models/resnet_model_epoch_-042-_acc_0.999860-_val_acc_0.98716.h5')
    clustering_model.layers.pop()
    clustering_model.layers.pop()
    clustering_model.layers.pop()

    clustering_model.outputs = [clustering_model.layers[-1].output]
    clustering_model.layers[-1].outbound_nodes = []

    clustering_model.layers[0].trainable = False # this would be the base model part

    #clustering_model.outputs = [clustering_model.layers[-1].output]
    #clustering_model.layers[-1].outbound_nodes = []

    #clustering_model.layers[0].trainable = False # this would be the base model part

    #clustering_model.add(ResNet50(include_top = False, pooling='ave', weights = resnet_weigth_path))
    #clustering_model.layers[0].trainable = False

    clustering_model.add(VGG16(weights= 'imagenet' ,include_top= False))
    clustering_model.layers[0].trainable = False
    '''
    clustering_model.add(ResNet50(include_top = False, pooling='ave'))#, weights = resnet_weigth_path))
    clustering_model.layers[0].trainable = False

elif weight_type == 'ft':
    num_classes = 9
    base_model = ResNet50
    base_model = base_model(weights = 'imagenet', include_top = False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation= 'relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    clustering_model = Model(inputs = base_model.input, outputs = predictions)
    clustering_model.load_weights(fine_tuned_resnet_weight_path)

    clustering_model.layers.pop()
    clustering_model.layers.pop()
    clustering_model.outputs = [clustering_model.layers[-1].output]
    clustering_model.layers[-1].outbound_nodes = []

    clustering_model.layers[0].trainable = False # this would be the base model part


clustering_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

missed_imgs = []
rows = []
    
with open(imgs_path) as csv_file:
    paths = csv.reader (csv_file, delimiter='\n')
    img_names = []
    for path in paths:
        row = []
        correct_path = path[0]
        correct_path.replace(' ', '\ ')
        correct_path.replace('(', '\(')
        correct_path.replace(')', '\)')
        try:
            '''
            #######gray########
            img_object = cv2.imread(correct_path, cv2.IMREAD_GRAYSCALE)
            img_object = np.stack((img_object,)*3, axis=-1)
            '''
            img_object = cv2.imread(correct_path)
            img_object = cv2.resize(img_object, (img_size, img_size))
            img_object = np.array(img_object, dtype = np.float64)
            img_object = preprocess_input(np.expand_dims(img_object.copy(), axis = 0))

            resnet_feature = clustering_model.predict(img_object)
            resnet_feature = np.array(resnet_feature)
            row.append(correct_path)
            row.extend(list(resnet_feature.flatten()))
            print(row)


        except: 
            missed_imgs.append(path)
