# -*- coding: utf-8 -*-
"""
This is the script used to train an activity recognition 
classifier on accelerometer data.

"""

import os
import sys
import numpy as np
from numpy.lib.function_base import average
from sklearn.tree import export_graphviz
from features import extract_features
from util import slidingWindow, reorient, reset_vars
import pickle
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = 'combined-data(upper combined).csv'
data = np.genfromtxt(data_file, delimiter=',', skip_header=1)
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()
# %%---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------

print("Reorienting accelerometer data...")
sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data[i,0], data[i,1], data[i,2]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:,0:1],reoriented,axis=1)
data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)

# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 20
step_size = 20

# sampling rate should be about 25 Hz; you can take a brief window to confirm this
n_samples = 1000
time_elapsed_seconds = (data[n_samples,0] - data[0,0]) / 1000
sampling_rate = n_samples / time_elapsed_seconds

class_names = ["minimal movement", "legs moving", "posture change/sitting up"] #...

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

X = []
Y = []

for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:,1:-1]   
    feature_names, x = extract_features(window)
    X.append(x)
    Y.append(window_with_timestamp_and_label[10, -1])
    
X = np.asarray(X)
Y = np.asarray(Y)
n_features = len(X)
    
print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(Y)))
print("\n")
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------

cv = KFold(n_splits=10, random_state=None, shuffle=True)

tree = DecisionTreeClassifier(criterion="entropy", max_depth=4)

def _calc_precision(label_index, conf):
    TP = conf[label_index][label_index]
    FPandTP = 0
    for i in range(len(class_names)):
        FPandTP += conf[i][label_index] 
    return TP / FPandTP

def _calc_recall(label_index, conf):
    TP = conf[label_index][label_index]
    FNandTP = 0
    for i in range(len(class_names)):
        FNandTP += conf[label_index][i] 
    return TP / FNandTP

sum_acc = 0
sum_pre = [0, 0, 0, 0]
sum_rec = [0, 0, 0, 0]
for train_index, test_index in cv.split(X):
    print("~~~~~~~~~~~~~~ START FOLD ~~~~~~~~~~~~~~")

    train_data, test_data = X[train_index], X[test_index]
    train_labels, test_labels = Y[train_index], Y[test_index]
    tree.fit(train_data, train_labels)

    predicted_labels = tree.predict(test_data)
    conf = confusion_matrix(test_labels, predicted_labels)
    print(conf)
    accuracy = (conf[0][0] + conf[1][1] + conf[2][2] + conf[3][3]) / np.sum(conf)
    sum_acc += accuracy
    print(f'{accuracy = }')

    # Minimal Movement Stats
    precision_little_movement = _calc_precision(0, conf)
    sum_pre[0] += precision_little_movement
    print(f'{precision_little_movement = }')
    recall_little_movement = _calc_recall(0, conf)
    sum_rec[0] += recall_little_movement
    print(f'{recall_little_movement = }')
    
    # Leg Movement Stats 
    precision_legs = _calc_precision(1, conf)
    sum_pre[1] += precision_legs
    print(f'{precision_legs = }')
    recall_legs = _calc_recall(1, conf)
    sum_rec[1] += recall_legs
    print(f'{recall_legs = }')

    # Posture Change Stats
    precision_turning = _calc_precision(2, conf)
    sum_pre[2] += precision_turning
    print(f'{precision_turning = }')
    recall_turning = _calc_recall(2, conf)
    sum_rec[2] += recall_turning
    print(f'{recall_turning = }')

    # # Sitting Up Stats
    # precision_sitting = _calc_precision(3, conf)
    # sum_pre[3] += precision_sitting
    # print(f'{precision_sitting = }')
    # recall_sitting = _calc_recall(3, conf)
    # sum_rec[3] += recall_sitting
    # print(f'{recall_sitting = }')



    print("~~~~~~~~~~~~~~ END FOLD ~~~~~~~~~~~~~~")
    print("\n")

print("~~~~~~~~~~~~~~ AVERAGES ~~~~~~~~~~~~~~")
average_accuracy = sum_acc / 10
print(f'{average_accuracy = }')
for i in range(4):
    average_precision = sum_pre[i] / 10
    print(f'For {class_names[i]}: {average_precision = }')
    average_recall = sum_rec[i] / 10
    print(f'For {class_names[i]}: {average_recall = }')

tree.fit(X, Y)

export_graphviz(tree, out_file='tree.dot', feature_names = feature_names)

# with open('classifier.pickle', 'wb') as f:
#     pickle.dump(tree, f)