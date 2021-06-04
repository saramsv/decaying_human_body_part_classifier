#RUN: python path2clustering file_with_paths
import tensorflow.keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import cv2
import sys
import csv
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type = str)

args = parser.parse_args()

imgs_path = args.img_path


img_size = 224

base_model = ResNet50
base_model = base_model(weights = 'imagenet', include_top = False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
d = Dense(1024, activation= 'relu')(x)

clustering_model = Model(inputs = base_model.input, outputs = d)

clustering_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


data = pd.read_csv(imgs_path, sep =":", names = ['path','label','conf'])
labels = data['label'].unique()
for label in labels:
    rows = []
    img_names = []
    df = data[data['label'] == label]
    df = df.reset_index()
    for path in df['path']:
        img_names.append(path)
    
        '''
        #######gray########
        img_object = cv2.imread(correct_path, cv2.IMREAD_GRAYSCALE)
        img_object = np.stack((img_object,)*3, axis=-1)
        '''
        img_object = cv2.imread(path)
        img_object = cv2.resize(img_object, (img_size, img_size))
        img_object = np.array(img_object, dtype = np.float64)
        img_object = preprocess_input(np.expand_dims(img_object.copy(), axis = 0))

        resnet_feature = clustering_model.predict(img_object)
        resnet_feature = np.array(resnet_feature)
        rows.append(list(resnet_feature.flatten()))
    features = np.array(rows)
    pca_model = PCA(n_components = 256)
    PCAed = pca_model.fit_transform(features)
    kmeans = KMeans(n_clusters = 20)
    kmeans.fit(PCAed)
    kmeans_labels = kmeans.predict(PCAed)
    for i,cluster_label in enumerate(kmeans_labels):
        print("{}: {}_{}".format(img_names[i].replace("JPG","icon.JPG"), label,cluster_label))
