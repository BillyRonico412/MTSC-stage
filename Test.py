# JAFFUER Pierre


#This is a TEST, do not use


from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import contingency_matrix
import os
import sktime
import time

# -------------- SETUP ------------------------------------------

# Dataset path folder
DATA_PATH = os.path.join(os.path.dirname(__file__), "Datasets")

# Datasets paths
# [ [train set path, test set path, dataset name], ...]
datasets_path = [
    ["RacketSports/RacketSports_TRAIN.ts", "RacketSports/RacketSports_TEST.ts", "RacketSport"]
]

# Setup classifier
# [ [classifier, classifier name], ...]
classifiers = [
    [KNeighborsTimeSeriesClassifier(1, 'uniform', 'brute', 'dtw', None), "DTW-1NN"],
    [KNeighborsTimeSeriesClassifier(4, 'uniform', 'brute', 'dtw', None), "DTW-4NN"]
]


# --------------- MAIN PROGRAM ---------------------------------

# Load data
# [ ((train_data, train_class), (test_data, test_class), dataset name), ...]
data = []
for train_path, test_path, name in datasets_path:
    data+=[(load_from_tsfile_to_dataframe(os.path.join(DATA_PATH, train_path)), 
            load_from_tsfile_to_dataframe(os.path.join(DATA_PATH, test_path)),
            name)]

for classifier, classifier_name in classifiers:
    print("|---"+classifier_name)
    for ((train_data, train_class), (test_data, test_class), name) in data:
        # Training
        classifier.fit(train_data, train_class)

        # Predicting class
        prediction = classifier.predict(test_data)

        # Computing accuracy
        accuracy = accuracy_score(test_class, prediction)

        # Computing contingency matrix
        cm = contingency_matrix(test_class, prediction)

        # Show results
        print("    |---"+name+": ")
        print("        Accuracy :"+str(accuracy))