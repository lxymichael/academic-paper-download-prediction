# Predicting paper download number and time using LSTM and DTW
This is the code for using a LSTM for predicting the download number and time until the next download. The user clustering is based on hierarchical clustering on the distance between users, measured by DTW (Dynamic time warping). The data is proprietary in the archive and will not be shared.

## Getting started
You need a working java environment from version 1.7 upwards.

### Prerequisites
You need to download/install the following packages by clicking into the links:

* [Deep learning for Java](https://deeplearning4j.org/index.html)--the deep learning framework used
* [Fast Dynamic Time Warping](https://github.com/rmaestre/FastDTW)--The time warping library to compare similarity between sequences
* [Hierarchical clustering](https://github.com/lbehnke/hierarchical-clustering-java/tree/master/src/main/java/com/apporiented/algorithm/clustering)--the clustering algorithms

## Data
1. For LSTM input data, it is in the format specified by deeplearning4j (https://deeplearning4j.org/lstm.html#code). For training and testing, each has 2 folders: features and labels, where each file represents an instance (user). Each line in a feature file represents a feature vector for that user, and each line in a label file represents a label. There should be equal numbers of feature files and label files.
The model outputs accuracy, F1, as well as individual predictions. It is optional to output weighted F1 (weight decided by class proportions).

2. For DTW clustering data, to get the input ready make a csv file in the format each line in format [user_index, feature_1, feature_2.....]
Example: Consider there are 2 users each with 3 sequences, then the input file looks like the following:
```
0,1,1,2,0,1,0,0
0,3,4,4,0,2,0,2
0,1,1,2,0,0,0,1
1,2,2,1,0,1,1,0
1,3,0,3,0,0,0,0
1,1,0,1,0,0,0,0
```
One-step warping is applied to get optimal performance. In terms of very large input data, multiple-step warping can be tried to accelerate. The output is by default the top and second level clusteres from hierarchical clustering.

## Run
Use the DTW clustering to divide your instances, and then run LSTM on the clusters. Search hyperparametrs in LSTM such as layer size, learning rate, minibatch size etc.
