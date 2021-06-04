#run python3 decom_sequence_generator.py file_with_the_following_format 
# the file for this script should be image_name va1l val2.... valn.
import sys
import numpy as np
import csv
import ast
import datetime
import argparse
from operator import itemgetter 
from functools import reduce
from scipy.spatial import distance
import math
import sequence



def key_func(x):
    # For some year like 2011 the year is 2 digits so the date format should ne %m%d%y but for others like 2015 it should be %m%d%Y
    try:
        #date = ""
        if '(' in x:     
            date_ = x.split('D_')[-1].split('(')[0].strip()
        else:
            date_ = x.split('D_')[-1].split('.')[0].strip()
        mdy = date_.split('_')
        m = mdy[0]
        d = mdy[1]
        y = mdy[2]
        if len(m) == 1:
            m = '0' + m
        if len(d) == 1:
            d = '0' + d
        date_  = m + d + y
        if len(date_) == 6: #the format that has 2 digits for year
            return datetime.datetime.strptime(date_, '%m%d%y')
        else:
            return datetime.datetime.strptime(date_, '%m%d%Y')
        
    except:
        print(x)
        import bpython
        bpython.embed(locals())
        exit()

def sort_dates(donors2imgs): #sorts the dates by getting a list of img_names for each donor and sorting that
    for key in donors2imgs:
        donors2imgs[key] = sorted(donors2imgs[key], key=key_func)
    return donors2imgs

def convert_to_time(img_name):
    if '(' in img_name:     
        date_ = img_name.split('D_')[-1].split('(')[0].strip()
    else:
        date_ = img_name.split('D_')[-1].split('.')[0].strip()
    mdy = date_.split('_')
    m = mdy[0]
    d = mdy[1]
    y = mdy[2]
    if len(m) == 1:
        m = '0' + m
    if len(d) == 1:
        d = '0' + d
    date_  = m + d + y
    if len(date_) == 6: #the format that has 2 digits for year
        return datetime.datetime.strptime(date_, '%m%d%y')
    else:
        return datetime.datetime.strptime(date_, '%m%d%Y')

def cal_day_from_deth(donors2imgs_sorted):
    for key in donors2imgs_sorted:
        day2imgs = {} 
        first_img = True
        for img in donors2imgs_sorted[key]:
            if first_img == True:
                start_time = convert_to_time(img)
                first_img = False
            img_time = convert_to_time(img)
            time_from_start = (img_time - start_time).days
            if time_from_start not in day2imgs:
                day2imgs[time_from_start] = []
            day2imgs[time_from_start].append(img)
        donors2imgs_sorted[key] = day2imgs 
    return donors2imgs_sorted 
    # this a dictionary with each donor_id as keys and values are another 
    #dictionary with keys being xth days since day one and the values are a 
    #list of images that belong to day xth for that donor.

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeding_file', type = str)

    args = parser.parse_args()

    embedings_file = args.embeding_file #sys.argv[1] # This should be a pca version the embedings

    donors2imgs = {}
    donors2img2embed = {}
    imgname2add = {}
    
    with open(embedings_file, 'r') as csv_file:
        data = csv.reader(csv_file,delimiter = '\n')
        vectors = []
        for row in data:
            row = row[0]
            row= row.split('JPG')
            img_name = row[0] + 'JPG'
            embeding = row[1].strip()
            embeding = embeding.replace(' ', ',')
            embeding = ast.literal_eval("[" + embeding[1:-1] + "]") # this embeding is a list now
            donor_id = img_name.split('/Daily')[0].split('/')[-1]
            if donor_id not in donors2img2embed and donor_id not in donors2imgs:
                donors2img2embed[donor_id] = {} 
                donors2imgs[donor_id] = [] # a list for all of the images belonging to the same donor
            donors2img2embed[donor_id][img_name] = embeding 
            # this a dictionary with each donor_id as keys and values are another dictionary
            # with keys being an image and the values being the feature vector for that imag

            donors2imgs[donor_id].append(img_name)

        donors2imgs_sorted = sort_dates(donors2imgs) # this sorts the images for a donor based on their dates
        donor2day2imgs = cal_day_from_deth(donors2imgs_sorted)

        day2clus2emb = sequence.sequence_finder(donors2img2embed, donor2day2imgs) 

