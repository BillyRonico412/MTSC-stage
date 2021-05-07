# JAFFUER Pierre

from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics.cluster import contingency_matrix
import numpy as np
import os
import sktime
import time


# -------------- SETUP -----------------

# Datasets path folder
DATA_PATH = os.path.join(os.path.dirname(__file__), "Datasets")

# Classifiers
# [ [classifier, classifier name], ...]
classifiers = [
    [KNeighborsTimeSeriesClassifier(1, 'uniform', 'dtw'), "DTW-1NN"],
    [KNeighborsTimeSeriesClassifier(4, 'uniform', 'dtw'), "DTW-4NN"]
]

# Number of cores to use (-1 -> all)
nb_jobs = -1

# Split strategy
nb_split = 10
cv = KFold(n_splits=nb_split)

# --------------- MAIN PROGRAM --------------------

# Display some info
nbDatasets = 0
for  dataset in os.listdir(DATA_PATH): nbDatasets+=1
print("Evaluating on %s cores %d classifiers on %d datasets with a %d fold cross-validation..." % 
     ("all" if nb_jobs == -1 else str(nb_jobs), len(classifiers), nbDatasets, nb_split))

start_global_time = time.perf_counter()

# Evaluates all classifiers using cross validation per dataset
for dataset in os.listdir(DATA_PATH):
    # Loads dataset for cross validation
    print("\nLoading %s dataset..." % dataset)
    start_load_time = time.perf_counter()

    filepath = dataset+"/"+dataset+"_" # dataset/dataset_TEST.ts and dataset/dataset_TRAIN.ts
    # Load train data + class
    d, c = load_from_tsfile_to_dataframe(os.path.join(DATA_PATH, filepath+"TRAIN.ts"))
    # Load test data + class
    dd, cc = load_from_tsfile_to_dataframe(os.path.join(DATA_PATH, filepath+"TEST.ts"))

    # Store all data and all class (concatenate train and test)
    data, classes = d.append(dd), np.concatenate((c, cc))

    elapsed_load_time = time.perf_counter() - start_load_time
    print("Loading took: %f seconds" % elapsed_load_time)

    # Now we will do all cross-validations on this dataset
    for classifier, classifier_name in classifiers:
        print("Classifier: "+classifier_name)
        start_time = time.perf_counter()
        # cross-validation
        scores = cross_val_score(classifier, data, classes, cv=cv, n_jobs=nb_jobs)
        elapsed_time = time.perf_counter() - start_time
        print("    |Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print("    |_Cross validation took: %f seconds" % elapsed_time)

elapsed_global_time = time.perf_counter() - start_global_time
print("\nDone in %f seconds." % elapsed_global_time)
