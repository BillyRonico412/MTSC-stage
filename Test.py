# JAFFUER Pierre
# TEST

from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.metrics import accuracy_score
import os
import sktime

DATA_PATH = os.path.join(os.path.dirname(__file__), "Datasets")
train_x, train_y = load_from_tsfile_to_dataframe(os.path.join(DATA_PATH, "RacketSports/RacketSports_TRAIN.ts"))
test_x, test_y = load_from_tsfile_to_dataframe(os.path.join(DATA_PATH, "RacketSports/RacketSports_TEST.ts"))

classifier1 = KNeighborsTimeSeriesClassifier(1, 'uniform', 'brute', 'dtw', None)
classifier2 = KNeighborsTimeSeriesClassifier(4, 'uniform', 'brute', 'dtw', None)

classifier1.fit(train_x, train_y)
classifier2.fit(train_x, train_y)

pred_y_1 = classifier1.predict(test_x)
pred_y_2 = classifier2.predict(test_x)
print("DTW 1: ", accuracy_score(test_y, pred_y_1))
print("DTW 4: ", accuracy_score(test_y, pred_y_2))
