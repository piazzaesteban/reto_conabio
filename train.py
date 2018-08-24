import numpy as np
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split,KFold
import utils as ut
import pickle

NDVI = True
image,gt = ut.read_raster_data(NDVI)
values, ground_truth = ut.eliminate_lacking_data(image,gt)

#Models
clf = LinearSVC(C=0.5,random_state=0,max_iter=1000)
#clf = KNeighborsClassifier(n_neighbors=7)
#clf = GaussianNB()

#Reduce array so that SVC training doesn't take hours
values,X_test, ground_truth, y_test = train_test_split(values, ground_truth, test_size=0.7, random_state=0)
values = values[:500000]
ground_truth = ground_truth[:500000]

#5 folds cross validation
kf = KFold(n_splits=5)
cross_val = [clf.fit(values[train], ground_truth[train]).score(values[test], ground_truth[test])
	for train, test in kf.split(ground_truth)]

print(cross_val)
print(reduce(lambda x, y: x + y, cross_val) / len(cross_val))

filename = 'SVM_NDVI2.sav'
pickle.dump(clf, open(filename, 'wb'))






	
