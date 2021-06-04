#python3 clustering.py --embeding_file data/pcaUT29-15  --cluster_number 7 > daily_merge_7ClusAll
# the file for this script should be image_name va1l val2.... valn.
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import sys
import numpy as np
import csv
import ast
import argparse
from operator import itemgetter 
from functools import reduce
from scipy.spatial import distance
import math
csv.field_size_limit(sys.maxsize)



def cluster_all(img_names, vectors):
    vectors = np.array(vectors)
    #vectors = vectors / vectors.max(axis=0)
    ## kmeans:
    #pca = PCA(n_components=16)
    #vectors = pca.fit_transform(vectors)
    #print(vectors)
    kmeans = KMeans(n_clusters = num_clusters)
    kmeans.fit(vectors)
    labels = kmeans.predict(vectors)
    #clustering = DBSCAN(eps=0.5, min_samples=5).fit(vectors)
    #labels = clustering.labels_
    '''
    ######### Agglomerative ######
    agglomerative = AgglomerativeClustering(n_clusters = num_clusters, linkage='single')
    agglomerative.fit(list(vectors))
    labels = agglomerative.labels_#predict(vectors)
    '''
    for index, label in enumerate(labels):
        print(img_names[index].replace('JPG', 'icon.JPG:') ,  label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeding_file', type = str)
    parser.add_argument('--cluster_number') # A number

    args = parser.parse_args()

    embedings_file = args.embeding_file #sys.argv[1] # This should be a pca version the embedings
    num_clusters = int(args.cluster_number)

    img_names = []
    vectors = []
    
    with open(embedings_file, 'r') as csv_file:
        data = csv.reader(csv_file,delimiter = '\n')
        for row in data:
            row = row[0]
            row= row.split('JPG')
            img_name = row[0] + 'JPG'
            embeding = row[1].strip()
            embeding = embeding.replace(' ', ',')
            embeding = ast.literal_eval("[" + embeding[1:-1] + "]") # this embeding is a list now
            vectors.append(embeding)
            img_names.append(img_name)
        cluster_all(img_names, vectors)
