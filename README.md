--------------------------------------------------------------------------- 
Plud: dataset curator, Version 1.0 
Copyright 2020 Sara Mousavi, The University of Tennessee, Knoxville
--------------------------------------------------------------------------- 

We are making our code for curating a human decomposition dataset freely available for reaserch to be used for other datasets that need to be labeled. If you would like to use the dataset for any other purposes please contact 
the authors. 

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.
## Prerequisites
Things that you need to have installed
```
Python3, Opencv, Keras 2.3.1, Tensorflow 2.0.0, 
```
## Running
### Initial step
```
python3 path2embeding.py --img_path paths_file --weight_type pt  > resnet_feautres_filename
```
```
sed -i 's/,//g;  s/\[//g; s/\]//g' resnet_feautres_filename
sed -i "s/'//g" resnet_feautres_filename
```
```
python3 clustering.py --embeding_file resnet_feautres_filename --cluster_number [n] > clusters
```
```
bash make_html.sh clusters
```
This will generate the html file to be viewed. Using python -m SimpleHTTPServer, the clusters can be displayed on port 8888.
After the manual labeling of the clusters and excluding the misclustered ones, the cluster ids in the file 'clusters' should be replaced with the new labels gained from the manual evaluation (The format of the 'cluster' file should be image_path: label).
### Iterations
This step envolves classification, classification and evaluation.
```
python3 classifier.py cluster 
```
This step generates VGG16, ResNet and inception models and placed in the directory with their names. In each directory there will be all of the models resulted from different epochs. model_evaluation is to evaluate the performance of the models on the test data. Only the one with highest accuracy is needed.
```
python3 predict_bodypart.py gt_file_name
```
gt_file_name has a path to an image and the grount truth label seperated by ":" in each line
```
python3 predict_labels.py file_name > predictions
```
Each line in file_name is a path_to_img
Will generates two different outputs. 1. image_name, label, confidence_value if the classifier is confident about label for the image_name, otherwise outputs image_name: image_embedding. The data can be separated as follows:
```
cat predictions | grep ":0.99" > high_confs 
cat predictions | grep "]" > low_confs
sed -i 's/,//g;  s/\[//g; s/\]//g' low_confs
sed -i "s/'//g" low_confs
python3 clustering.py --embeding_file low_confs --cluster_number [n] > clusters
```
To display:
```
bash make_html.sh clusters
```
And for the high confidence predictions to see both the label and the conf_value in localhost:8888:
```
bash make_classe_labeled_html.sh.sh clusters
```





