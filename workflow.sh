#!/bin/bash

#workflow script usage: ./workflow.sh emb_file paths_file output_file_name

if [ "$#" -ne 3 ]; then
  echo "Usage: ./workflow.sh emb_file paths_file output_file_name"
  exit
fi

seq_file="$3_seqs"
pred_file="$3_pred"
comb_file="$3_comb"
votpred_file="$3_votpred"

echo -e "\nRunning: \"python3 decom_sequence_generator.py --embeding_file $1 > $seq_file\""
python3 decom_sequence_generator.py --embeding_file $1 > $seq_file
echo "Done"

echo -e "\nRunning: \"python3 classifier.py $2 > $pred_file\""
python3 classifier.py $2 > $pred_file
echo "Done"

#format for comb_file: "path:seqNum:predLabel:conf"
echo -e "\nRunning: \"join -t: <(sort -t: $seq_file) <(sort -t: $pred_file) > $comb_file\"" 
join -t: <(sort -t: $seq_file) <(sort -t: $pred_file) > $comb_file
echo "Done"

echo -e "\nRunning: \"python3 label_voting.py $comb_file > $votpred_file\""
python3 label_voting.py $comb_file > $votpred_file
echo "Done"

#clean seq_votpred
#g/^.*path.*/d
#:%s/^.*\/h/\/h/g
#and more cleaning
#after cleaning it would be path:label

#join -t: <(sort -t: gt) <(sort -t: votpred) > gt_votpred
