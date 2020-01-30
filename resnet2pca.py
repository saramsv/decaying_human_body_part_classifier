#python3 resnet2pca.py --embeding_file 50000resnet.csv > 50000PCAed64
# the input file is image_name var1 var2.....varn. 
#NOTE: No '' round the image_name 
import csv
from sklearn.decomposition import PCA 
import argparse
import ast
import numpy as np
import sys
csv.field_size_limit(sys.maxsize)


parser = argparse.ArgumentParser()
parser.add_argument('--embeding_file', type = str)

args = parser.parse_args()
embedings_file = args.embeding_file 


with open(embedings_file, 'r') as csv_file:
    data = csv.reader(csv_file,delimiter = '\n')
    vectors = []
    img_names = []
    for row in data:
        row = row[0]
        row= row.split('JPG')
        img_name = row[0] + 'JPG'
        img_names.append(img_name)
        embeding = row[1].strip('')
        embeding = embeding.replace(' ', ',')
        embeding = ast.literal_eval("[" + embeding[1:-1] + "]") # this embeding is a list now
        vectors.append(embeding)

    vectors = np.array(vectors)
    model = PCA(n_components = 128) 
    results = model.fit_transform(vectors)
    for index, img in enumerate(img_names):
        print(img, ",", list(results[index,:]))




