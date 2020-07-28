# JAFFUER Pierre
# TEST

from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import contingency_matrix
import os
import sktime
import time

DATA_PATH = os.path.join(os.path.dirname(__file__), "Datasets")

train_x, train_y = load_from_tsfile_to_dataframe(os.path.join(DATA_PATH, "RacketSports/RacketSports_TRAIN.ts"))
test_x, test_y = load_from_tsfile_to_dataframe(os.path.join(DATA_PATH, "RacketSports/RacketSports_TEST.ts"))

classifier1 = KNeighborsTimeSeriesClassifier(1, 'uniform', 'brute', 'dtw', None)
classifier2 = KNeighborsTimeSeriesClassifier(4, 'uniform', 'brute', 'dtw', None)

time.clock()
classifier1.fit(train_x, train_y)
classifier2.fit(train_x, train_y)


pred_y_1 = classifier1.predict(test_x)
pred_y_2 = classifier2.predict(test_x)

print("DTW 1: ", accuracy_score(test_y, pred_y_1))
print("DTW 4: ", accuracy_score(test_y, pred_y_2))

tableau_convergence1 = contingency_matrix(test_y, pred_y_1)
tableau_convergence2 = contingency_matrix(test_y, pred_y_2)

print(tableau_convergence1)
print(tableau_convergence2)


#to do: merger les 2 bases (test+train) avant de faire la cross validation

