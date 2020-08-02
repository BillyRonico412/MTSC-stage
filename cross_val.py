# JAFFUER Pierre
# TEST

from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import os
import sktime
import time


# -------------- SETUP ------------------------------------------

# Dataset path folder
DATA_PATH = os.path.join(os.path.dirname(__file__), "Datasets")

# Setup classifier
# [ [classifier, classifier name], ...]
classifiers = [
    [KNeighborsTimeSeriesClassifier(1, 'uniform', 'brute', 'dtw', None), "DTW-1NN"],
    [KNeighborsTimeSeriesClassifier(4, 'uniform', 'brute', 'dtw', None), "DTW-4NN"]
]
# Number of cores to use (-1 -> all)
nb_jobs = -1

# Number of data split for cross validation
nb_split = 10

# --------------- MAIN PROGRAM ---------------------------------

# Loads data for cross validation
# [ [data, class, data name], ...]
data = []
for name in os.listdir(DATA_PATH):
    print("Loading %s dataset..." % name)
    start_time = time.perf_counter()
    # Load train data + class
    d, c = load_from_tsfile_to_dataframe(os.path.join(DATA_PATH, name+"/"+name+"_TRAIN.ts"))
    # Load test data + class
    dd, cc = load_from_tsfile_to_dataframe(os.path.join(DATA_PATH, name+"/"+name+"_TEST.ts"))
    # Store all data, all class and dataset name
    data+=[[d.append(dd),np.concatenate((c, cc)), name]]

    elapsed_time = time.perf_counter() - start_time
    print("Finished in: %f seconds" % elapsed_time)

# Evaluates classifiers using cross validation per datasets
for classifier, classifier_name in classifiers:
    print("|---Classifier: "+classifier_name)
    for (d, c, name) in data:
        print("    |---Dataset: "+name)
        start_time = time.perf_counter()
        scores = cross_val_score(classifier, d, c, cv=nb_split, n_jobs=nb_jobs)
        elapsed_time = time.perf_counter() - start_time
        print("        |---Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print("        |---Finished in: %f seconds" % elapsed_time)

print("Done.")
