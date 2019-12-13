#!usr/bin/env python3

import pandas as pd
import operator
import sys

file_name = sys.argv[1]

df = pd.read_csv(file_name, delimiter = ':')
df.columns = ('path', 'seq', 'label', 'conf')
num_seq_uniq = df['seq'].unique().shape[0]

for i in range(num_seq_uniq):

    labels = df.loc[df['seq']==df['seq'].unique()[i]]
    labels = labels.reset_index(drop = True)


    predicted_labels = list(labels['label'].values)
    counts = dict((x, predicted_labels.count(x)) for x in set(predicted_labels))
    frequent_label = max(counts.items(), key = operator.itemgetter(1))[0].strip()

    confs = list(labels['conf'].values)
    counts_conf = dict((x, confs.count(x)) for x in set(confs))
    frequent_conf = max(counts_conf.items(), key = operator.itemgetter(1))[0]

    
    print(labels['seq'].unique()[0], ":",  frequent_label, ":", frequent_conf)
