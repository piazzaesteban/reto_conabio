import numpy as np
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
'''
We need a scaler.
'''
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,KFold
import utils as ut
import pickle

NDVI = False

'''
We adapt the changes in utils.
'''
image,gt, origin_shape = ut.read_raster_data(NDVI)
values, ground_truth, cond = ut.eliminate_lacking_data(image,gt)

#Models
clf = LinearSVC(C=0.5,random_state=0,max_iter=1000)
#clf = KNeighborsClassifier(n_neighbors=7)
#clf = GaussianNB()

#Reduce array so that SVC training doesn't take hours
values,X_test, ground_truth, y_test = train_test_split(values, ground_truth, test_size=0.999, random_state=0)

'''
values = values[:500000]
ground_truth = ground_truth[:500000]
'''

#5 folds cross validation
kf = KFold(n_splits=5)
'''
cross_val = [clf.fit(values[train], ground_truth[train]).score(values[test], ground_truth[test])
	for train, test in kf.split(ground_truth)]
'''
'''
One-liners are very cool, but in some cases they make the code unreadable.
'''
scaler = StandardScaler()
cross_val = []
for train, test in kf.split(ground_truth):
    '''
    We scale the train data for this fold before training the model.
    '''
    sl = scaler.fit(values[train])
    X = sl.transform(values[train])
    '''
    We use the same scaler to scale the test data before prediction.
    '''
    X_test = sl.transform(values[test])

    cross_val.append(clf.fit(X, ground_truth[train]).score(X_test, ground_truth[test]))


print(cross_val)
print("Overall train acurracy:")
print(reduce(lambda x, y: x + y, cross_val) / len(cross_val))

sl = scaler.fit(values)
clf.fit(sl.transform(values), ground_truth)

filename = 'SVM_FIXED.sav'
pickle.dump(clf, open(filename, 'wb'))
filename_scaler = 'SCALER.sav'
pickle.dump(sl, open(filename_scaler, 'wb'))






	
