"""
Module for cloud classification training and application

It uses sklearn for machine learning
"""

import numpy as np
import csv
import os
import time
import h5py
from operator import itemgetter

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, cross_val_score, ShuffleSplit, train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

import pickle
import lzma

def pickle_to_array(obj):
   return np.fromstring(lzma.compress(pickle.dumps(obj)), dtype=np.uint8)

def unpickle_from_array(array):
   return pickle.loads(lzma.decompress(array.data))



class classificator:

    def __init__(self,ini):

        self.names = "" # Feature names
        self.x = [] # Feature data
        self.y = [] # Label data
        self.cloud_class = []
        self.class_prob = []
        self.indices = None
        self.time = []
        self.modelfile = ini.cloud_class_model
        self.label_names = ['Cu','Ci & Cs','Cc & Ac','Clear','Sc','St & As','Cb & Ns']

    # Collect training data
    def get_training(self, ini, contour_flag=False, exclude=[]):

        if contour_flag:
            self.feats = np.arange(6,33)
        else:
            self.feats = np.arange(6,26)

        for i in range(1,8):
            filename = ini.path_training + os.sep + str(i) + os.sep + "features.dat"
            print(("read file: %s" % filename))
            reader = csv.reader(open(filename, "r"),delimiter=" ")
            names =  next(reader)
            self.names = np.array(names)[self.feats]
            find = []
            for j in range(0, len(self.names)):
                if self.names[j] not in exclude:
                    find.append(j)
                else:
                    print(self.names[j])
            find = np.array(find)
            self.names = self.names[find]
            cnt = 0
            for row in reader:
                    if contour_flag:
                        tmp = np.array([float(x) for x in row[6:]])
                        self.x.append(tmp[find])
                    else:
                        tmp = np.array([ float(row[x]) for x in self.feats ])
                        self.x.append(tmp[find])
                    self.y.append(i)
                    cnt += 1
            print("Number of images for class %d: %d" % (i, cnt))
        print("Number of images for classification: %d" % len(self.x))

        self.x = np.array(self.x)
        self.y = np.float16(self.y)

        # Preprocess feature "overall RBR"
        q = np.where(self.names == "Overall_RB_ratio")[0]
        if len(q) > 0:
            q = q[0]
            ind = np.where(self.x[:,q] > 1)
            if len(ind) > 0:
                for h in ind:
                    self.x[h,:] = np.nan
                    self.y[h] = np.nan

        ind = np.isfinite(self.y) & ( np.all(np.isfinite(self.x),axis=1) )

        self.x = self.x[ind,:]
        self.y = np.int32(self.y[ind])

        self.max_feat = self.x.shape[1]


    # Learn model ( with grid search and cross validation )
    def fit(self, modelfile, model="SVC", rank_feat=False, print_feat=False,
            plot_feat=False, grid_search=False):

        st = time.time()

        # train normalizer
        self.normalize()

        # defaults
        params = {}
        params['gamma'] = 'auto'
        params['C'] = 1
        params['n_neighbors'] = 100

        # define initial model
        if model == "SVC":
            self.predictor = SVC(gamma=params['gamma'], C=params['C'],probability=True)
        elif model == "kNN":
            self.predictor = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
        elif model == "RandomForest":
            self.predictor = RandomForestClassifier(n_estimators=10, max_depth=None, \
                min_samples_split=10, random_state=0)
        elif model == "Tree":
            self.predictor = tree.DecisionTreeClassifier()


        # Apply feature ranking
        if rank_feat:
            self.indices = self.feature_selection(self.x,self.y,self.max_feat,
                                                  print_flag=print_feat,plot_flag=plot_feat)
            self.max_feat = self.test_feature_ranking(params)
            self.indices = self.indices[:self.max_feat]

            self.x = self.reduce_features(self.x, self.indices, max_feat=self.max_feat)
            self.normalize() # repeat normalization on reduced set
        else:
            self.indices = np.arange(0,self.x.shape[1])
            self.max_feat = self.x.shape[1]


        # normalize feature data
        self.x = self.scaler.transform(self.x)

        if grid_search:

            # specify parameters for parameter grid search
            if model == "SVC":
                param_grid = {"gamma": [2**-10,2**-9,2**-8,2**-7,2**-6,2**-5,2**-4,2**-3,2**-2,2**-1,2**0,2**1,2**2], \
                    "C": [2**6,2**7,2**8,2**9,2**10]}
            elif model == "kNN":
                param_grid = {"n_neighbors": [1, 10, 50, 100, 200, 500, 1000]}

            grid_search = GridSearchCV(self.predictor, param_grid=param_grid,cv=4)
            grid_search.fit(self.x, self.y)

            # get best parameters
            params, performance = self.report(grid_search.grid_scores_,print_flag=True)
            print("Grid search...finished with a performance of %.2f and parameter %s in %.2f seconds." % \
                 ( performance, params, round(time.time() - st,1)))


        # define new model
        if model == "SVC":
            self.predictor = SVC(gamma=params['gamma'], C=params['C'],probability=True)
        elif model == "kNN":
            self.predictor = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
        elif model == "RandomForest":
            self.predictor = RandomForestClassifier(n_estimators=10, max_depth=None, \
                min_samples_split=1, random_state=0)
        elif model == "Tree":
            self.predictor = tree.DecisionTreeClassifier()


        X_train, X_test, y_train, y_test = train_test_split(
            self.x[:,:], self.y, test_size=0.33, random_state=42)
        scores = self.crossval(self.predictor, X_train, y_train)
        self.predictor.fit(X_train, y_train)
        y_pred = self.predictor.predict(X_test)
        r = accuracy_score(y_test, y_pred)

        print(r, scores)

        self.predictor.fit(self.x[:,:],self.y)


        # Export model
        with h5py.File(self.modelfile,'w') as h:
            h.create_dataset('scaler', data=pickle_to_array(self.scaler))
            h.create_dataset('model', data=pickle_to_array(self.predictor))
            h.create_dataset('names', data=pickle_to_array(self.names[self.indices]) )
            h.create_dataset('accuracy', data=r)



    # Normalize training data set and store transformation function
    def normalize(self):
        self.scaler = preprocessing.StandardScaler().fit(self.x)

    def crossval(self, model, x, y):
        cv = ShuffleSplit(n_splits=3, test_size=0.7, random_state=0)
        scores = cross_val_score(model, x, y, cv=cv)
        return scores

    # imgeature Selection
    def feature_selection(self,x,y,nfeatures,plot_flag=False,print_flag=False):

        # Build a forest and compute the feature importances
        forest = ExtraTreesClassifier(n_estimators=250,  \
            random_state=0)

        forest.fit(self.scaler.transform(x) ,y )
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], \
                axis=0)
        indices = np.argsort(importances)[::-1]
        indices = indices[0:nfeatures]

        if print_flag == True:
            # Print the feature ranking
            print("Feature ranking:")
            for f in range(nfeatures):
                print(("%d. feature %s (%f)" % (f + 1, self.names[indices[f]], importances[indices[f]])))

        if plot_flag == True:
            # Plot the feature importances of the forest
            from matplotlib import pyplot as plt
            plt.figure()
            plt.title("Feature importances")
            plt.bar(list(range(nfeatures)), importances[indices], \
                color="r", yerr=std[indices], align="center")
            plt.xticks(list(range(nfeatures)), indices)
            plt.xlim([-1,nfeatures])
            plt.show()

        return indices


    def reduce_features(self,x,indices,max_feat=10):
        x_sel = np.empty([x.shape[0],max_feat])
        for f in range(max_feat):
            x_sel[:,f] = x[:,indices[f]]

        return x_sel



    def test_feature_ranking(self, params):

        from matplotlib import pyplot as plt

        acc = []

        X_train, X_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=0.7, random_state=0)

        # Start feature selection
        for i in range(1, self.x.shape[1]+1, 1):

            # train model
            self.predictor.fit(X_train[:,self.indices][:,:i], y_train)
            y_pred = self.predictor.predict(X_test[:,self.indices][:,:i])
            r = accuracy_score(y_test, y_pred)
            acc.append(r)
            print('Features = %d - Extra feature: %s - Accuracy = %f ' % (i, \
            self.names[self.indices][i-1], r))

        acc = np.array(acc)
        return np.argwhere(acc>0.99)[0][0] + 1


    # Utility function to report best scores
    def report(self, grid_scores, n_top=10, print_flag=True):
        top_scores = sorted(grid_scores, key=itemgetter(1),reverse=True)[:n_top]
        if print_flag:
            for i, score in enumerate(top_scores):
                print(("Model with rank: {0}".format(i + 1)))
                print(("Mean validation score: {0:.3f} (std: {1:.3f})".format( \
                        score.mean_validation_score, \
                        np.std(score.cv_validation_scores))))
                print(("Parameters: {0}".format(score.parameters)))
                print("")
        return top_scores[0].parameters, top_scores[0].mean_validation_score



    # Apply classification model
    def apply_model(self,features,modelfile,rank_feat=False,contour_flag=False):
        # Load model, scaler and featurelist
        with h5py.File(self.modelfile, 'r') as infile:
            self.predictor = unpickle_from_array(infile['model'][:])
            self.scaler = unpickle_from_array(infile['scaler'][:])
            names = unpickle_from_array(infile['names'][:])
        vec = []
        for key in names:
            vec.append(features[key])
        vec = np.array(vec)
        if np.any(np.isnan(vec)):
            return -1, np.nan
        else:
            # normalization
            vec = np.array(vec).reshape(1,-1)
            vec = self.scaler.transform(vec)
            # predicted class
            cloudClass = self.predictor.predict(vec)[0]
            # predicted probabilities for each class
            prob = self.predictor.predict_proba(vec)[0]
            return cloudClass, np.round(np.array(prob),2)
