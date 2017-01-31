cat 1pipeline.py
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('/home/mg800/teapot/class_train_Xy.csv', delimiter=',', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
#training_features, testing_features, training_classes, testing_classes = \
#    train_test_split(features, tpot_data['class'], random_state=42)

training_features=features
training_classes=tpot_data['class']

exported_pipeline = make_pipeline(
    make_union(VotingClassifier([("est", LogisticRegression(C=50.0, dual=False, penalty="l1"))]), FunctionTransformer(lambda X: X)),
    PCA(iterated_power=10, svd_solver="randomized"),
    SelectPercentile(percentile=11, score_func=f_classif),
    KNeighborsClassifier(n_neighbors=5, weights="distance")
)


exported_pipeline.fit(training_features, training_classes)
#results = exported_pipeline.predict(testing_features)


tpot_data = np.recfromcsv('/home/mg800/teapot/class_test_in.csv', delimiter=',', dtype=np.float64)
testing_features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
results = exported_pipeline.predict(testing_features)
i=1
print('Point_ID, Output')
for res in results:
	print(str(i)+','+str(int(res)))
	i+=1
