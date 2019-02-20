import random
import sys, os
import numpy as np
import matplotlib.colors
from pylab import rcParams
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, validation_curve, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from scipy.spatial import procrustes
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.externals.joblib import parallel_backend
from sklearn.utils import shuffle
from sklearn.ensemble import BaggingClassifier
from matplotlib.colors import ListedColormap
from ipywidgets import interact, interactive, fixed, interact_manual
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from PIL import Image
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

"""
    Notices:
    1) For PCA, the total number of components corresponds to the dimensionality of the samples.
    2) Explained variance for PLS is different than for PCA (usually does not add up to 100% , see stackexchange answers)
        More information in: chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/https://www.utdallas.edu/~herve/Abdi-PLS-pretty.pdf
    3) No additional de-meaning and scaling is necessary after procrustes (already aligned).
            X_train_scaled = preprocessing.scale(X_train, axis=0,
                            with_mean = True, with_std = False)
    4)
"""

# ==============================================================================
# shape processing functions ..
# ==============================================================================
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def truncate_landmarks(landmarks_all):
    # input: landmarks_all are all the landmarks for the 2ch or 4ch views
    # output: ensures that all individual 2ch/4ch views have the same number of landmarks/rows
    lens = np.zeros((len(landmarks_all), 1), dtype=int)
    for i in range(len(landmarks_all)):
        lens[i] = len(landmarks_all[i])

    min_len = np.min(lens)

    landmarks_all_trunc = np.zeros((len(landmarks_all), min_len, 2), dtype=np.float64)

    print('min len = ', min_len)

    # remove points if it has more than min_len ..
    for i in range(len(landmarks_all)):  # i = 1 to all 2ch views
        if len(landmarks_all[i]) > min_len:
            # remove points but make sure we dont remove the points that belong to the special landmarks (top pts and low pt)
            lowest_pt_idx = int(np.floor((len(landmarks_all[i])-1)/2.0))
            arr = np.arange(len(landmarks_all[i]))
            arr = np.delete(arr, lowest_pt_idx)  # delete index for top point1
            arr = np.delete(arr, 0)  # delete index for top point 2
            arr = np.delete(arr, len(arr)-1)  # delete index for lowest point
            np.random.shuffle(arr)
            num_pts_to_remove = len(landmarks_all[i]) - min_len
            arr = arr[:num_pts_to_remove]
            landmarks_all_trunc[i] = np.delete(
                landmarks_all[i], arr, axis=0)  # just remove the 3rd point
        else:
            landmarks_all_trunc[i] = landmarks_all[i]

    return landmarks_all_trunc

def check_lengths_of_arrays(list_of_arrays):
    n = len(list_of_arrays[0])
    if all(len(x) == n for x in list_of_arrays):
        print('All shapes have the same number of landmarks.')
        trunc_array = list_of_arrays
    else:
        print('Truncating some shapes so that number of landmarks match..')
        trunc_array = truncate_landmarks(list_of_arrays)

    return trunc_array

def process_data(landmarks_trunc):
    n_shapes = len(landmarks_trunc)
    num_pts = len(landmarks_trunc[0])
    processed_landmarks = np.zeros((2*num_pts, n_shapes), dtype=np.float64)

    for n_shape in range(n_shapes):
        xs = landmarks_trunc[n_shape][:, 0]
        ys = landmarks_trunc[n_shape][:, 1]
        processed_landmarks[0::2, n_shape] = xs
        processed_landmarks[1::2, n_shape] = ys

    return processed_landmarks

# ==============================================================================
# alignShapes
# ==============================================================================

def my_procrustes(X, Y, scaling, reflection):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        X : (2*num_landmarks x 1) vector
        Y : (2*num_landmarks x 1) vector

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the vector of transformed Y-values (2*num_landmarks x1 )

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y
    """
    # convert to 2D arrays (1st column is x, 2nd column is y)
    X = np.column_stack((X[0::2], X[1::2]))
    Y = np.column_stack((Y[0::2], Y[1::2]))

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    # convert 2D matrix of coordinates into 1d:
    Z = np.vstack((Z[:,0], Z[:,1])).reshape((-1,),order='F')

    return d, Z, tform

def translate_to_origin(shape_vec):
    """
    Translates landmark to origin.

    lm : vector of length 2*num_landmarks
    """
    centroid_x = np.mean(shape_vec[0::2])
    centroid_y = np.mean(shape_vec[1::2])

    centroid_vec = np.tile([centroid_x, centroid_y], int(len(shape_vec)/2))

    translated_lm = shape_vec - centroid_vec

    return translated_lm

def alignShapes(allShapes, scaling, reflection):
    """
    ALIGNSHAPES uses Procrustes analysis to align a set of shapes (with or without scaling).
    INPUT
        allShapes: [2*n_landmarks x n_shapes] --> ndarray!, n_shapes is basically n_subjects
           Collected from the HD scan semi-automated segmentation
           for each row: 20 points (40 elements): x1, y1, x2, y2, ..., x20, y20%
        scaling: (bool) Do you want to scale your images or not? (default = 0)

    OUTPUT
        alignedShapes: The realigned shapes. Same shape as totalShapes
        meanShape: The mean shape of allShapes, same shape as totalShapes

    Shape analysis techniques based on this paper:
    Cootes, T. F., Taylor, C. J., Cooper, D. H., & Graham, J. (1995).
       "Active Shape Models-Their Training and Application. Computer Vision and
       Image Understanding."

    See also PROCRUSTES, PLACELANDMARKS, PLOTLANDMARKS

    Woo-Jin Cho
    15-May-2018
    """
    # Pre-allocate
    n_shapes = allShapes.shape[1]  # the number of columns
    alignedShapes = np.zeros(allShapes.shape)

    # first translate all shapes to origin:
    for i in range(allShapes.shape[1]):
        allShapes[:,i] = translate_to_origin(allShapes[:,i])

    # set first mean shape as the reference shape ..
    meanShape = normalize(translate_to_origin(allShapes[:, 0])) # use initial estimate as first shape ..

    it = 0

    while it < 500:

        print('iteration : ', it)

        ds = []

        # start loop now that we have optimal refShape ..
        for n_shape in range(n_shapes):

            # current shape ..
            iShape = allShapes[:, n_shape]  # current shape

            # alignment ..
            disparity, Z, _ = my_procrustes(meanShape, iShape, scaling, reflection)

            # project Z (the aligned shape) onto tangent space ..
            c = 1.0/np.dot(Z, meanShape)
            Z = Z*c

            # append error ..
            ds.append(disparity)

            # store alignedShapes to output ..
            alignedShapes[:, n_shape] = Z

        # mean_aligned_shape ..
        new_meanShape = np.mean(alignedShapes, axis=1)  # x1, y1, x2, y2, ..., x20, y20
        new_meanShape = normalize(translate_to_origin(new_meanShape))

        # compute tolerance with previous meanShape ..
        if ((meanShape - new_meanShape) < 1e-10).all():
            break

        # set new meanshape to the old
        meanShape = new_meanShape

        it = it + 1

    return meanShape, alignedShapes, np.asarray(ds)

# ==============================================================================
# classification functions ..
# ==============================================================================

def split_data_meshwise(X, y, split_perc, num_meshes, num_views):
    """
        This function splits the data meshwise.
        That is, 33 views of the same mesh are either in train, test
        or cv dataset. They cannot be split or we introduce bias to the model.
        This is because a slice of the anatomy can tell info about the other
        views which can affect its classification.
        Run this function after alignShapes ..

        INPUT:
            X : ALIGNED shapes ..  (n_shapes x n_features)
                The first 225 are HC and last 50 are HF
            y : labels
            split_perc : percentage of train + cv
                e.g. [0.6, 0.4]
            num_meshes : fore (275) + orient (275)

        OUTPUT:
            x_data = [x_train_cv, x_test]
            y_data = [y_train_cv, y_test]
    """
    # split data into specified %/%/% ..
    num_train_cv = int(num_meshes * split_perc)
    num_test = num_meshes - num_train_cv

    # generate random indices for indexing ..
    r = np.random.RandomState(42)
    train_cv_idxs = r.choice(num_meshes, size=num_train_cv, replace=False)
    test_idxs = np.delete(np.arange(num_meshes),
                np.argmax(np.arange(num_meshes) == train_cv_idxs[:, np.newaxis],
                axis=1))

    # declare arrays ..
    num_feats = X.shape[1]
    x_train_cv = np.zeros((num_train_cv * num_views, num_feats), dtype=float)
    x_test = np.zeros((num_test * num_views, num_feats), dtype=float)
    y_train_cv = np.zeros((num_train_cv * num_views, 1), dtype=int)
    y_test = np.zeros((num_test * num_views, 1), dtype=int)

    # split data meshwise.. xsplit[0] has all views from mesh number 0 (from 275 meshes, HF + HC combined)
    x_split = np.asarray(np.split(X, num_meshes, axis=0)) # np splits equally (works since each mesh generates 33 views)
    y_split = np.asarray(np.split(y, num_meshes, axis=0))

    x_train_cv = x_split[train_cv_idxs]
    x_test = x_split[test_idxs]

    y_train_cv = y_split[train_cv_idxs]
    y_test = y_split[test_idxs]

    # return colors for tsne viewing 9classify by HC/HF and mesh number
    y_trcv_colors = np.zeros((x_train_cv.shape[0], x_train_cv.shape[1], 4)) # RGBA
    y_test_colors = np.zeros((x_test.shape[0], x_test.shape[1], 4))

    cmap_hc = plt.cm.RdPu
    norm_hc = matplotlib.colors.Normalize(vmin=-0.3*(225-1), vmax=225-1)
    cmap_hf = plt.cm.BuGn
    norm_hf = matplotlib.colors.Normalize(vmin=-0.1*(num_meshes-1), vmax=num_meshes-1)

    # for train idxs
    for i, idx in enumerate(train_cv_idxs):
        if idx < 225-1: # hc
            y_trcv_colors[i] = np.tile(cmap_hc(norm_hc(idx)), (y_trcv_colors.shape[1], 1))
        else: # from 225-275-1 is hf
            y_trcv_colors[i] = np.tile(cmap_hf(norm_hf(idx)), (y_trcv_colors.shape[1], 1))

    # for test idxs
    for i, idx in enumerate(test_idxs):
        if idx < 225-1: # hc
            y_test_colors[i] = np.tile(cmap_hc(norm_hc(idx)), (y_test_colors.shape[1], 1))
        else: # from 225-275-1 is hf
            y_test_colors[i] = np.tile(cmap_hf(norm_hf(idx)), (y_test_colors.shape[1], 1))

    # store into output ..
    x_data = [x_train_cv, x_test]
    y_data = [y_train_cv, y_test]

    return x_data, y_data, y_trcv_colors, y_test_colors

def main_SVC(alignedShapes, y, n_components, k, plot_curves):
    """
    1) Align shapes using procrustes analysis
    2) Performs PCA on the aligned shapes to reduce dimensionality
    3) Performs SVC on the dim. reduced data
    4) Plots validation and learning curves for validation.

    INPUT:
        alignedShapes : (n_features x n_shapes)
        y : (1 x n_shapes)
        n_components = number of components to keep after PCA dimensionality reduction
        scaling : scaling or not on the initial dataset (Procrustes alignment stage)
        k : k-fold cross validation
            if k<2 then it must be a vector of C and gamma parameters. (no cv needed)
        kernel : kernel used for svd (rbf or linear)

    OUTPUT:
        clf.best_estimator_ : the best estimator from the gridsearchCV
        clf.best_params_ : hyperparameters for the best_estimator
        accuracy : accuracy of the best model

    Notes:
        1) CV is always always performed on the training set only!
        2) Test set should never be touched!

    """
    # split into train/test MESHWISE ..
    num_meshes = 275
    num_views = 33
    split_perc = 0.75 # (50/25)/25 for (train, cv) - test
    x_data, y_data, _, _ = split_data_meshwise(alignedShapes.T, y, split_perc,
                                                num_meshes, num_views)

    # assign data .. (need flattening)
    x_train = np.reshape(x_data[0], (x_data[0].shape[0] * x_data[0].shape[1], x_data[0].shape[2]))
    y_train = y_data[0].ravel()

    x_test = np.reshape(x_data[1], (x_data[1].shape[0] * x_data[1].shape[1], x_data[1].shape[2]))
    y_test = y_data[1].ravel()

    # shuffle within data just to be sure ..
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_test, y_test = shuffle(x_test, y_test, random_state=42)

    # fit pca ..
    pca = PCA(n_components = n_components)
    pca.fit(x_train)

    # project data onto pcs ..
    X_train_pca = pca.transform(x_train)  # project onto the k components ..
    X_test_pca = pca.transform(x_test)  # do the same with the test data ..

    # run cv with k-fold ..
    if not isinstance(k, list):
        print('running k-fold ..')
        pipeline = Pipeline(steps=[('clf', SVC(kernel='rbf', gamma=0.1))])
        n_estimators = 20

        # parameter grid range ..
        param_range1 = np.asarray([1e-3, 1e-1, 1, 1e1, 1e3, 1e5, 1e8, 1e11])
        param_range2 = np.asarray([0.0001, 0.001, 0.01, 0.1, 1])
        # param_range1 = np.asarray([100, 500, 1000, 1500, 2000])
        # param_range2 = np.asarray([0.001, 0.005, 0.010, 0.050, 0.10, 0.5])
        param_grid = {'clf__C': param_range1,
                      'clf__gamma': param_range2
                     }

        # setting up grid-search ..
        cv = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
        print('Entering Grid Search CV .. ')
        clf_cv = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)

        # fit ..
        print('Fitting Best Estimator ..')
        with parallel_backend('threading'):
            clf_cv.fit(X_train_pca, y_train)

        print("Tuned rg best params: {}".format(clf_cv.best_params_))

        # learning curves ..
        if plot_curves:
            print('Plotting Learning Curve..')
            train_sizes, train_scores, test_scores = learning_curve(estimator=clf_cv.best_estimator_,
                                                                    X=X_train_pca, y=y_train,
                                                                    train_sizes=np.linspace(.1, 1.0, 5),
                                                                    cv=cv, scoring='accuracy', n_jobs=4)

            plot_learning_curve(train_sizes, train_scores, test_scores, title='Learning curve for SVC')


            # plot validation curve for C parameter ..
            plt.figure(figsize=(9, 6))

            print('Plotting Validation Curve..')
            train_scores, test_scores = validation_curve(estimator=clf_cv.best_estimator_,
                                                         X=X_train_pca, y=y_train,
                                                         param_name="clf__C",
                                                         param_range=param_range1,
                                                         cv=cv, scoring='accuracy', n_jobs=4)

            plot_validation_curve(param_range1, train_scores, test_scores, title="Validation Curve for C", alpha=0.1)


            # plot validation curve for gamma parameter ..
            train_scores, test_scores = validation_curve(estimator=clf_cv.best_estimator_,
                                                         X=X_train_pca, y=y_train,
                                                         param_name="clf__gamma",
                                                         param_range=param_range2,
                                                         cv=cv, scoring='accuracy', n_jobs=4)

            plot_validation_curve(param_range2, train_scores, test_scores, title="Validation Curve for gamma", alpha=0.1)

    else: # no kfold needed, parameters given ..
        print('providing cv parameters (no kfold) ..')
        _C = k[0]
        _gamma = k[1]
        gamma = 1.0
        C = 1000.0
        clf_cv = SVC(kernel='rbf', gamma=_gamma, C = _C)

        # fit ..
        print('Fitting Best Estimator ..')
        with parallel_backend('threading'):
            clf_cv.fit(X_train_pca, y_train)

        print("Tuned rg best params: {}".format(clf_cv.get_params()))

    # classify/predict ..
    y_pred = clf_cv.predict(X_test_pca)

    # performance measure ..
    accuracy = np.sum(y_pred == y_test)/len(y_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = float(tn) / float((tn+fp))
    sensitivity = float(tp) / float((tp+fn))
    print(classification_report(y_test, y_pred))
    print('accuracy : ', accuracy)
    print('sensitivity : ', specificity)
    print('specificity : ', sensitivity)

    decision_boundary = True

    if decision_boundary:

        clf = SVC()
        clf.set_params(**clf_cv.get_params())
        print(clf.get_params())
        clf.fit(X_train_pca[:, :2], y_train)

        plot_svm_db(X_train_pca[:, :2], X_test_pca[:, :2], y_train, y_test, clf)

    if not isinstance(k, list):
        return clf_cv.best_estimator_, clf_cv.best_params_
    else:
        return [],[]

def plot_contour(ax, x, y, clf_cv, title):
    """
    Plotting function for svc ..
    """
    # create meshgrid ..
    x_min, x_max = x[:,0].min() - 0.5, x[:,0].max() + 0.5
    y_min, y_max = x[:,1].min() - 0.5, x[:,1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                        np.arange(y_min, y_max, .02))

    # decision function
    Z = clf_cv.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plot ..
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(x[:,0], x[:,1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

def plot_svm_db(x_train, x_test, y_train, y_test, clf_cv):
    """
    Plot classifier boundary decision of svm in training and testing.
    """
    fig, axs = plt.subplots(nrows = 1, ncols = 2)
    plot_contour(axs[0], x_train, y_train, clf_cv, 'training')
    plot_contour(axs[1], x_test, y_test, clf_cv, 'testing')
    plt.show()

# ==============================================================================
# PCA functions ..
# ==============================================================================

from matplotlib.widgets import Slider, Button, RadioButtons

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return [x, y]

def cart2pol(x, y):
    """
    This computes radius and angle from the origin!
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return [rho, phi]

def performance_print(y_pred, y_test):
    accuracy = np.sum(y_pred == y_test)/len(y_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = float(tn) / float((tn+fp))
    sensitivity = float(tp) / float((tp+fn))
    print(classification_report(y_test, y_pred))
    print('accuracy : ', accuracy)
    print('sensitivity : ', specificity)
    print('specificity : ', sensitivity)

def PCA_features(alignedShapes, y, n_components, interactive_plot, whiten):
    """
    1) Align shapes using procrustes analysis
    2) Performs PCA on the aligned shapes to reduce dimensionality
    3) Plots the interpretable features!

    INPUT:
        X : (n_features x n_shapes)
        y : (1 x n_shapes)
        n_components = number of components to keep after PCA dimensionality reduction
        scaling : scaling all shapes to reference shape (not the same as unit normalization)
        k : k-fold cross validation
    """

    # split into train/test MESHWISE ..
    num_meshes = 275
    num_views = 33
    split_perc = 0.75 # (50% + 25%) for train + cv and 25% for tests
    x_data, y_data, _ , _ = split_data_meshwise(alignedShapes.T, y, split_perc,
                                                num_meshes, num_views)

    # assign data .. (need flattening)
    x_train = np.reshape(x_data[0], (x_data[0].shape[0] * x_data[0].shape[1], x_data[0].shape[2]))
    y_train = y_data[0].ravel()

    x_test = np.reshape(x_data[1], (x_data[1].shape[0] * x_data[1].shape[1], x_data[1].shape[2]))
    y_test = y_data[1].ravel()

    # shuffle within data just to be sure ..
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_test, y_test = shuffle(x_test, y_test, random_state=42)

    # fit pca ..
    pca = PCA(n_components=n_components, whiten=whiten)
    pca.fit(x_train)

#     # reconstruct the data with only the first 2 PCs ..
#     X_trans = pca.transform(x_train)
#     X_recons = pca.inverse_transform(X_trans)  # reconstructed using only n_components
#     loss = ((x_train - X_recons) ** 2).mean()
#     print('reconstruction error = ', loss)

    # generate new shapes by adding a certain amount in the direction of the PCs
    eigenvalues = pca.explained_variance_ratio_ # PERCENTAGE of variance
    variance = pca.explained_variance_ # AMOUNT of variance
    eigenvectors = pca.components_

    # plot eigen-variance ..
    eigen_var_plot = False
    if eigen_var_plot:
        plt.figure(figsize=(16, 10))
        plt.plot(np.arange(1,len(eigenvalues)+1), np.cumsum(eigenvalues), linewidth=2)
        plt.axis('tight')
        plt.xlabel('# of PCs')
        plt.ylabel('Explained Variance')
        plt.axvline(5, linestyle=':', label='n_components chosen')
        plt.legend(prop=dict(size=12))
        plt.show()

    # compute meanShape ..
    meanShape = np.mean(x_train, axis=0)

    # display data ..
    cols =  ['0', '1', '2', '3']
    rows = ['eig', 'std', 'cumsum']
    arr = np.array([eigenvalues[:4], np.sqrt(eigenvalues[:4]), 100.0*np.cumsum(eigenvalues[:4])])
    df = pd.DataFrame(arr, index = rows, columns = cols)
    display(df)

    # interactive plotting ..
    if interactive_plot:
        # plot meanShape first ..
        fig, ax = plt.subplots(figsize=(7,7))
        ax.set_xlim([-0.4, 0.4])
        ax.set_ylim([-0.4, 0.4])
        plt.subplots_adjust(left=0.25, bottom=0.25)
        l, = plt.plot(meanShape[1::2], -meanShape[0::2], linestyle='-', marker='o')

        # display variances of pcs ..
        eig1 = 'pc1 var = ' + str(np.around(eigenvalues[0]*100, 2)) + '%'
        eig2 = 'pc2 var = ' + str(np.around(eigenvalues[1]*100, 2)) + '%'
        eig3 = 'pc3 var = ' + str(np.around(eigenvalues[2]*100, 2)) + '%'
        eig4 = 'pc4 var = ' + str(np.around(eigenvalues[3]*100, 2)) + '%'

        plt.gcf().text(0.05, 0.75, eig1, fontsize=10)
        plt.gcf().text(0.05, 0.7, eig2, fontsize=10)
        plt.gcf().text(0.05, 0.65, eig3, fontsize=10)
        plt.gcf().text(0.05, 0.6, eig4, fontsize=10)

        # create sliders..
        axcolor = 'lightgoldenrodyellow'
        ax1 = plt.axes([0.25, 0.18, 0.65, 0.03], facecolor=axcolor)
        ax2 = plt.axes([0.25, 0.13, 0.65, 0.03], facecolor=axcolor)
        ax3 = plt.axes([0.25, 0.08, 0.65, 0.03], facecolor=axcolor)
        ax4 = plt.axes([0.25, 0.03, 0.65, 0.03], facecolor=axcolor)

        slider1 = Slider(ax1, 'pc1 weight', valmin=-2, valmax=2, valinit=0, valstep=0.05)
        slider2 = Slider(ax2, 'pc2 weight', valmin=-2, valmax=2, valinit=0, valstep=0.05)
        slider3 = Slider(ax3, 'pc3 weight', valmin=-2, valmax=2, valinit=0, valstep=0.05)
        slider4 = Slider(ax4, 'pc4 weight', valmin=-2, valmax=2, valinit=0, valstep=0.05)


        # update the plot on weights of slider ..
        def update(val):
            w1 = slider1.val
            w2 = slider2.val
            w3 = slider3.val
            w4 = slider4.val

            update_pc1 = w1*np.sqrt(variance[0])*eigenvectors[0].ravel()
            update_pc2 = w2*np.sqrt(variance[1])*eigenvectors[1].ravel()
            update_pc3 = w3*np.sqrt(variance[2])*eigenvectors[2].ravel()
            update_pc4 = w4*np.sqrt(variance[3])*eigenvectors[3].ravel()

            # add all updates to the mean shape ..
            finalShape = meanShape + update_pc1 + update_pc2 + update_pc3 + update_pc4

            l.set_xdata(finalShape[1::2])
            l.set_ydata(-finalShape[0::2])
            fig.canvas.draw_idle()

        slider1.on_changed(update)
        slider2.on_changed(update)
        slider3.on_changed(update)
        slider4.on_changed(update)

        # reset button ..
        resetax = plt.axes([0.8, 0.75, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

        # reset event ..
        def reset(event):
            slider1.reset()
            slider2.reset()
            slider3.reset()
            slider4.reset()

            l.set_xdata(meanShape[1::2])
            l.set_ydata(-meanShape[0::2])

            fig.canvas.draw_idle()

        button.on_clicked(reset)

        plt.show()

    # non interactive plotting ..
    non_inter_plot = True
    if non_inter_plot:
            # initialize newShapes ..
            newShapes_pc1 = np.zeros((4+1, meanShape.shape[0], ), dtype=np.float64)
            newShapes_pc2 = np.zeros((4+1, meanShape.shape[0], ), dtype=np.float64)
            newShapes_pc3 = np.zeros((4+1, meanShape.shape[0], ), dtype=np.float64)
            newShapes_pc4 = np.zeros((4+1, meanShape.shape[0], ), dtype=np.float64)

            # add mean shapes first ..
            newShapes_pc1[0] = meanShape
            newShapes_pc2[0] = meanShape
            newShapes_pc3[0] = meanShape
            newShapes_pc4[0] = meanShape

            param_range = [-2, -1, 1, 2]
            for i, weight in enumerate(param_range):
                newShapes_pc1[i+1] = meanShape + weight * np.sqrt(variance[0]) * eigenvectors[0].ravel()
                newShapes_pc2[i+1] = meanShape + weight * np.sqrt(variance[1]) * eigenvectors[1].ravel()
                newShapes_pc3[i+1] = meanShape + weight * np.sqrt(variance[2]) * eigenvectors[2].ravel()
                newShapes_pc4[i+1] = meanShape + weight * np.sqrt(variance[3]) * eigenvectors[3].ravel()

            plt.figure()
            # plt.figure(figsize=(16,8))     # now figure with 4 subplots
            colors = ['k', 'b', 'm', 'r', 'g']

            for i in range(5):
                plt.subplot(141)
                plt.axis('equal')
                plt.plot(newShapes_pc1[i, 1::2], -newShapes_pc1[i, 0::2],
                         linestyle='-', marker='o', color=colors[i])
                plt.title('mean + PC_1', size=10)
                plt.xticks(size=10)
                plt.yticks(size=10)

                plt.subplot(142)
                plt.axis('equal')
                plt.plot(newShapes_pc2[i, 1::2], -newShapes_pc2[i, 0::2],
                         linestyle='-', marker='o', color=colors[i])
                plt.title('mean + PC_2', size=10)
                plt.xticks(size=10)
                plt.yticks(size=10)

                plt.subplot(143)
                plt.axis('equal')
                plt.plot(newShapes_pc3[i, 1::2], -newShapes_pc3[i, 0::2],
                         linestyle='-', marker='o', color=colors[i])
                plt.title('mean + PC_3', size=10)
                plt.xticks(size=10)
                plt.yticks(size=10)

                plt.subplot(144)
                plt.axis('equal')
                plt.plot(newShapes_pc4[i, 1::2], -newShapes_pc4[i, 0::2],
                         linestyle='-', marker='o', color=colors[i])
                plt.title('mean + PC_4', size=10)
                plt.xticks(size=10)
                plt.yticks(size=10)

            # caption
            txt="Variation of the landmarks from the mean shape (black) along the direction of the first 4 PCs."
            plt.figtext(0.5, 0.05, txt, wrap=True, horizontalalignment='center', fontsize=10)
            plt.show()

    score_plot = False

    if score_plot:
        # project data onto pcs ..
        X_train_pca = pca.transform(x_train)  # project onto the k components ..
        X_test_pca = pca.transform(x_test)  # do the same with the test data ..
#                np.transpose(pca.components_[0:2, :]),

        text =  ['pc1', 'pc2', 'pc1 vs pc2']
        biplot(np.column_stack((X_test_pca[:,0], X_test_pca[:,1])),
               np.column_stack((eigenvectors[0].T, eigenvectors[1].T)),
               y_test, text)

        text =  ['pc1', 'pc3', 'pc1 vs pc3']
        biplot(np.column_stack((X_test_pca[:,0], X_test_pca[:,2])),
               np.column_stack((eigenvectors[0].T, eigenvectors[2].T)),
               y_test, text)

        text =  ['pc2', 'pc3', 'pc2 vs pc3']
        biplot(np.column_stack((X_test_pca[:,1], X_test_pca[:,2])),
               np.column_stack((eigenvectors[1].T, eigenvectors[2].T)),
               y_test, text)


        text =  ['pc1', 'pc4', 'pc1 vs pc4']
        biplot(np.column_stack((X_test_pca[:,0], X_test_pca[:,3])),
               np.column_stack((eigenvectors[0].T, eigenvectors[3].T)),
               y_test, text)

    return pca

def biplot(score, coeff, labels, text):
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(text[0], fontsize = 15)
    ax.set_ylabel(text[1], fontsize = 15)
    ax.set_title(text[2], fontsize = 20)
    targets = [1, 0]
    colors = ['r', 'b']

    arr = np.column_stack((score, labels))

    finalDf = pd.DataFrame(data = arr
             , columns = [text[0], text[1], 'target'])

    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, text[0]]
                   , finalDf.loc[indicesToKeep, text[1]]
                   , c = color
                   , s = 50
                   , alpha=0.5)
    ax.legend(['F', 'NF'])
    ax.grid()
    plt.show()

def PLS_features(alignedShapes, y, num_cs, interactive_plot):
    """ Performs PLS rather than PCA to learn features! (better for clasisfication)
        https://www.quora.com/What-is-the-difference-between-PLS-and-PCA
    """

    # split into train/test MESHWISE ..
    num_meshes = 275
    num_views = 33
    split_perc = 0.75 # (50% + 25%) for train + cv and 25% for tests
    x_data, y_data,_,_ = split_data_meshwise(alignedShapes.T, y, split_perc,
                                                num_meshes, num_views)

    # assign data .. (need flattening)
    x_train = np.reshape(x_data[0], (x_data[0].shape[0] * x_data[0].shape[1], x_data[0].shape[2]))
    y_train = y_data[0].ravel()

    x_test = np.reshape(x_data[1], (x_data[1].shape[0] * x_data[1].shape[1], x_data[1].shape[2]))
    y_test = y_data[1].ravel()

    # shuffle within data just to be sure ..
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_test, y_test = shuffle(x_test, y_test, random_state=42)

    # tecnically , this doesnt work because it doesnt order the eigenvalues in order

    # if num_cs == 0: # choosing the optimal number of components via kfold mse error ..
    #     print('finding optimal number of components ..')
    #     mse = []
    #     total_ncs = x_train.shape[1]
    #     kf_10 = KFold(n_splits=10, shuffle=True, random_state=1)
    #     for i in np.arange(1, total_ncs):
    #         pls = PLSRegression(n_components=i)
    #         score = cross_val_score(pls, x_train, y_train,
    #                 cv=kf_10, scoring='neg_mean_squared_error').mean()
    #         mse.append(-score)
    #     optimal_ncs = np.argmin(mse)
    #     print('optimal # of PCs = ', optimal_ncs)
    # else:

    pls = PLSRegression(max_iter=5000, n_components = num_cs,
                        scale=False)
    pls.fit(x_train, y_train)
    y_pred = pls.predict(x_test)

    # binarize y_pred [0->1] to (0,1) ..
    y_pred = np.where(y_pred > 0.5, 1, 0).flatten()

    # performance measure ..
    accuracy = np.sum(y_pred == y_test)/len(y_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = float(tn) / float((tn+fp))
    sensitivity = float(tp) / float((tp+fn))

    print(classification_report(y_test, y_pred))
    print('accuracy : ', accuracy)
    print('sensitivity : ', specificity)
    print('specificity : ', sensitivity)

    # compute meanShape ..
    meanShape = np.mean(x_train, axis=0)

    # explain variance by components of X ..
    expl_var = True
    if expl_var:
        # components are the normalized loadings of X ..
        T = pls.x_scores_
        P = pls.x_loadings_
        C = pls.y_scores_
        B = pls.y_loadings_.ravel()

        # explained variance in pls ..
        expl_var_xs = np.zeros((P.shape[1],))
        expl_var_ys = np.zeros((P.shape[1],))

        SSx = np.zeros((P.shape[1], ))
        SSy = np.zeros((P.shape[1], ))

        varX = np.trace(np.dot(x_train.T, x_train))

        for i in range(P.shape[1]):
            tTt = np.dot(T[:,i] , T[:,i])
            ssx = tTt * np.dot(P[:,i] , P[:,i])
            ssy = tTt * np.dot(C[:,i] , C[:,i])
            SSx[i] = ssx
            SSy[i] = ssy

        # # the variance is actually 0.3% (i.e. 0.0003) --> correct it
        # expl_var_xs = 100.0 * SSx / varX
        # cum_expl_var_xs = 100.0 * np.cumsum(SSx) / varX
        eigenvectors = P.T

        # preprocessing.normalize(P.T, axis=0)
        total_variance_in_x = np.var(x_train[:,:num_cs], axis = 0)

        # variance in transformed X data for each latent vector:
        variance_in_x = np.var(pls.x_scores_, axis = 0)
        sort_idxs = np.argsort(-variance_in_x) # descending order
        sorted_variancex = variance_in_x[sort_idxs]
        sorted_eigenvectors = eigenvectors[sort_idxs]

        # normalize variance by total variance:
        expl_var_xs = sorted_variancex / total_variance_in_x
        cum_expl_var_xs = np.cumsum(expl_var_xs)

        plt.subplots(1,2, figsize=(16,10))
        plt.subplot(121)
        plt.plot(np.arange(expl_var_xs.shape[0]), expl_var_xs)
                # plt.plot(np.arange(P.shape[1]), expl_var_xs)
        plt.ylabel('explained variance in X')
        plt.xlabel('PC number')

        plt.subplot(122)
        plt.plot(np.arange(expl_var_xs.shape[0]), cum_expl_var_xs)
        plt.ylabel('cumsum variance in X')
        plt.xlabel('PC number')
        plt.show()



    # interactive plotting ..
    if interactive_plot:
        # plot meanShape first ..
        fig, ax = plt.subplots(figsize=(7,7))
        ax.set_xlim([-0.4, 0.4])
        ax.set_ylim([-0.4, 0.4])
        plt.subplots_adjust(left=0.25, bottom=0.25)
        l, = plt.plot(meanShape[1::2], -meanShape[0::2], linestyle='-', marker='o')

        # display variances of pcs ..
        eig1 = 'pc1 var = ' + str(np.around(expl_var_xs[0], 2)) + '%'
        eig2 = 'pc2 var = ' + str(np.around(expl_var_xs[1], 2)) + '%'
        eig3 = 'pc3 var = ' + str(np.around(expl_var_xs[2], 2)) + '%'
        eig4 = 'pc4 var = ' + str(np.around(expl_var_xs[3], 2)) + '%'

        plt.gcf().text(0.05, 0.75, eig1, fontsize=10)
        plt.gcf().text(0.05, 0.7, eig2, fontsize=10)
        plt.gcf().text(0.05, 0.65, eig3, fontsize=10)
        plt.gcf().text(0.05, 0.6, eig4, fontsize=10)

        # create sliders..
        axcolor = 'lightgoldenrodyellow'
        ax1 = plt.axes([0.25, 0.18, 0.65, 0.03], facecolor=axcolor)
        ax2 = plt.axes([0.25, 0.13, 0.65, 0.03], facecolor=axcolor)
        ax3 = plt.axes([0.25, 0.08, 0.65, 0.03], facecolor=axcolor)
        ax4 = plt.axes([0.25, 0.03, 0.65, 0.03], facecolor=axcolor)

        slider1 = Slider(ax1, 'pc1 weight', valmin=-2, valmax=2, valinit=0, valstep=0.05)
        slider2 = Slider(ax2, 'pc2 weight', valmin=-2, valmax=2, valinit=0, valstep=0.05)
        slider3 = Slider(ax3, 'pc3 weight', valmin=-2, valmax=2, valinit=0, valstep=0.05)
        slider4 = Slider(ax4, 'pc4 weight', valmin=-2, valmax=2, valinit=0, valstep=0.05)


        # update the plot on weights of slider ..
        def update(val):
            w1 = slider1.val
            w2 = slider2.val
            w3 = slider3.val
            w4 = slider4.val

            update_pc1 = w1*np.sqrt(sorted_variancex[0])*normalize(sorted_eigenvectors[0]).ravel()
            update_pc2 = w2*np.sqrt(sorted_variancex[1])*normalize(sorted_eigenvectors[1]).ravel()
            update_pc3 = w3*np.sqrt(sorted_variancex[2])*normalize(sorted_eigenvectors[2]).ravel()
            update_pc4 = w4*np.sqrt(sorted_variancex[3])*normalize(sorted_eigenvectors[3]).ravel()

            # add all updates to the mean shape ..
            finalShape = meanShape + update_pc1 + update_pc2 + update_pc3 + update_pc4

            l.set_xdata(finalShape[1::2])
            l.set_ydata(-finalShape[0::2])
            fig.canvas.draw_idle()

        slider1.on_changed(update)
        slider2.on_changed(update)
        slider3.on_changed(update)
        slider4.on_changed(update)

        # reset button ..
        resetax = plt.axes([0.8, 0.75, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

        # reset event ..
        def reset(event):
            slider1.reset()
            slider2.reset()
            slider3.reset()
            slider4.reset()

            l.set_xdata(meanShape[1::2])
            l.set_ydata(-meanShape[0::2])

            fig.canvas.draw_idle()

        button.on_clicked(reset)

        plt.show()

    # non-interactive plot ..
    non_inter_plot = False
    if non_inter_plot:
        # declare new shape arrays ..
        newShapes_pc1 = np.zeros((len(param_range)+1, meanShape.shape[0], ), dtype=np.float64)
        newShapes_pc2 = np.zeros((len(param_range)+1, meanShape.shape[0], ), dtype=np.float64)
        newShapes_pc3 = np.zeros((len(param_range)+1, meanShape.shape[0], ), dtype=np.float64)
        newShapes_pc4 = np.zeros((len(param_range)+1, meanShape.shape[0], ), dtype=np.float64)

        # add mean shapes first
        newShapes_pc1[0] = meanShape
        newShapes_pc2[0] = meanShape
        newShapes_pc3[0] = meanShape
        newShapes_pc4[0] = meanShape
        for i, weight in enumerate(param_range):
            newShapes_pc1[i+1] = meanShape + weight*normalize(eigenvectors[:, 0].ravel())
            newShapes_pc2[i+1] = meanShape + weight*normalize(eigenvectors[:, 1].ravel())
            newShapes_pc3[i+1] = meanShape + weight*normalize(eigenvectors[:, 2].ravel())
            newShapes_pc4[i+1] = meanShape + weight*normalize(eigenvectors[:, 3].ravel())

        # now figure with 4 subplots
        plt.figure(figsize=f(30, 16))
        colors = ['k', 'b', 'm', 'r', 'g']

        for i in range(5):
            plt.subplot(141)
            plt.plot(newShapes_pc1[i, 1::2], - newShapes_pc1[i, 0::2],
                     linestyle='-', marker='o', color=colors[i])
            plt.title('mean + PC_1', size=30)
            plt.xticks(size=20)
            plt.yticks(size=20)

            plt.subplot(142)
            plt.plot(newShapes_pc2[i, 1::2], - newShapes_pc2[i, 0::2],
                     linestyle='-', marker='o', color=colors[i])
            plt.title('mean + PC_2', size=30)
            plt.xticks(size=20)
            plt.yticks(size=20)

            plt.subplot(143)
            plt.plot(newShapes_pc3[i, 1::2], - newShapes_pc3[i, 0::2],
                     linestyle='-', marker='o', color=colors[i])
            plt.title('mean + PC_3', size=30)
            plt.xticks(size=20)
            plt.yticks(size=20)

            plt.subplot(144)
            plt.plot(newShapes_pc4[i, 1::2], - newShapes_pc4[i, 0::2],
                     linestyle='-', marker='o', color=colors[i])
            plt.title('mean + PC_4', size=30)
            plt.xticks(size=20)
            plt.yticks(size=20)

        plt.show()

def PLS_SVC(X, y, n_components, scaling, param_range, k):
    """
    Performs PLS rather than PCA to learn features! (better for clasisfication)
    https://www.quora.com/What-is-the-difference-between-PLS-and-PCA
    """
    meanShape, alignedShapes, ds = alignShapes(X, scaling)
    alignedShapes_T = np.transpose(alignedShapes)  # (n_shapes x n_features)

    # split the data into train and test 50% , 50%
    X_train, X_test, y_train, y_test = train_test_split(alignedShapes_T, y, test_size=0.5,
                                                        random_state=42)

    # de-center but do not normalize to unit-variance
    # X_train_scaled = preprocessing.scale(X_train, axis=0,
    #                     with_mean = True, with_std = False)

    # X_test_scaled = preprocessing.scale(X_test, axis=0,
    #                     with_mean = True, with_std = False)

    X_train_scaled = X_train
    X_test_scaled = X_test

    pls = PLSRegression(copy=True, max_iter=500, n_components=n_components,
                        scale=True, tol=1e-06)

    pls.fit(X_train_scaled, y_train)

    # now run the classifier using SVC
    X_train_pls = pls.transform(X_train_scaled)  # project onto the 20 pcs
    X_test_pls = pls.transform(X_test_scaled)  # do the same with the test data

    X_train_pls = np.asmatrix(X_train_pls, dtype='d')
    X_test_pls = np.asmatrix(X_test_pls, dtype='d')
    y_train = np.asarray(y_train, dtype='d')
    y_test = np.asarray(y_test, dtype='d')

    print('Data alignment complete.')

    print('kernel = rbf ..')
    pipeline = Pipeline(steps=[('clf', SVC(kernel='rbf'))])
    n_estimators = 20

    # set parameter grid for search ..
    param_range1 = [1e-3, 1e-1, 1e3, 5e3, 1e4, 1e5]
    param_range2 = [0.0001, 0.001, 0.01, 0.1, 1]
    param_grid = {'clf__C': param_range1,
                  'clf__gamma': param_range2
                 }

    # param_grid = {'clf__C': [1e-3, 1e-1, 1e3, 5e3, 1e4, 1e5],
                  # 'clf__gamma': [0.0001, 0.001, 0.01, 0.1, 1]
                  # }

    cv = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
    print('Entering Grid Search CV .. ')
    clf_cv = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=4)

    print('Fitting Best Estimator ..')
    with parallel_backend('threading'):
        clf_cv.fit(X_train_pls, y_train)

    print("Tuned rg best params: {}".format(clf_cv.best_params_))

    y_pred = clf_cv.predict(X_test_pls)

    data = [y_pred, y_test]

    # performance ..
    accuracy = np.sum(y_pred == y_test)/len(y_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = float(tn) / float((tn+fp))
    sensitivity = float(tp) / float((tp+fn))
    print(classification_report(y_test, y_pred))
    print('accuracy : ', accuracy)
    print('sensitivity : ', specificity)
    print('specificity : ', sensitivity)

    # plot validation and learning curves..
    plot_curves = False
    if plot_curves:
        print('Plotting Learning Curve..')
        train_sizes, train_scores, test_scores = learning_curve(estimator=clf_cv.best_estimator_,
                                                                X=X_train_pca, y=y_train,
                                                                train_sizes=np.linspace(.1, 1.0, 5),
                                                                cv=cv, scoring='f1', n_jobs=4)

        plot_learning_curve(train_sizes, train_scores, test_scores, title='Learning curve for SVC')


        # plot validation curve for C parameter ..
        plt.figure(figsize=(9, 6))

        print('Plotting Validation Curve..')
        train_scores, test_scores = validation_curve(estimator=clf_cv.best_estimator_,
                                                     X=X_train_pca, y=y_train,
                                                     param_name="clf__C",
                                                     param_range=param_range1,
                                                     cv=cv, scoring="f1", n_jobs=4)

        plot_validation_curve(param_range1, train_scores, test_scores, title="Validation Curve for C", alpha=0.1)

        # plot validation curve for gamma parameter ..
        train_scores, test_scores = validation_curve(estimator=clf_cv.best_estimator_,
                                                     X=X_train_pca, y=y_train,
                                                     param_name="clf__gamma",
                                                     param_range=param_range2,
                                                     cv=cv, scoring="f1", n_jobs=4)

        plot_validation_curve(param_range2, train_scores, test_scores, title="Validation Curve for gamma", alpha=0.1)


    return clf_cv.best_estimator_, clf_cv.best_params_

def test_n_components(clf, X, y, scaling, param_range):
    """
    This function tests to see how n_components affects the accuracy
    """
    fig, ax = plt.subplots(figsize=(16, 10))

    _, alignedShapes, _ = alignShapes(X, scaling)
    alignedShapes_T = np.transpose(alignedShapes)  # (n_shapes x n_features)

    # split the data into train and test 50% , 50%
    X_train, X_test, y_train, y_test = train_test_split(
        alignedShapes_T, y, test_size=0.5, random_state=42)

    X_train_scaled = preprocessing.scale(X_train, axis=0)
    X_test_scaled = preprocessing.scale(X_test, axis=0)

    # performance
    num_tests = len(param_range)
    accs = np.zeros((num_tests, 1)).ravel()
    senss = np.zeros((num_tests, 1)).ravel()
    specs = np.zeros((num_tests, 1)).ravel()

    for i, n_cps in enumerate(param_range):

        pca = PCA(n_components=n_cps)
        pca.fit(X_train_scaled)

        X_train_pca = pca.transform(X_train_scaled)  # project onto the 20 pcs
        X_test_pca = pca.transform(X_test_scaled)  # do the same with the test data

        # conver to np.matrix format or cvxopt.solver won't run properly
        X_train_pca = np.asmatrix(X_train_pca, dtype='d')
        X_test_pca = np.asmatrix(X_test_pca, dtype='d')
        y_train = np.asarray(y_train, dtype='d')
        y_test = np.asarray(y_test, dtype='d')

        # training stage (train on the X_train_pca only!)
        clf.fit(X_train_pca, y_train)

        # test it
        y_pred = clf.predict(X_test_pca)

        # performance
        accuracy = np.sum(y_pred == y_test)/len(y_test)
        accs[i] = accuracy

    eigenvalues = pca.explained_variance_ratio_
    print(eigenvalues)

    variances = np.round(np.cumsum(eigenvalues[:num_tests]), 1)

    xtick = np.arange(1, num_tests+1)  # for plotting evenly spaced xticks
    ax.plot(xtick, accs, label='accuracy', color='blue', marker='o')
    ax.xaxis.set_ticks(xtick)
    ax.xaxis.set_ticklabels(param_range)
    ax.set_xlabel('# of PCs.')
    ax.set_ylabel('Accuracy Score')
    # ax.set_ylim([0,1])
    ax.grid(ls='--')

    # for i in range(num_tests):
    #     ax.annotate(str(variances[i]) + '%', xy=(xtick[i], accs[i]-0.04))

    plt.show()

    return eigenvalues

# ==============================================================================
# plotting validation / learning curves ..
# ==============================================================================

def plot_learning_curve(train_sizes, train_scores, test_scores, title, alpha=0.1):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean + train_std,
                     train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(train_sizes, test_mean, label='test score', color='red', marker='o')

    plt.fill_between(train_sizes, test_mean + test_std,
                     test_mean - test_std, color='red', alpha=alpha)
    plt.title(title)
    plt.xlabel('Number of training points')
    plt.ylabel('F-measure')
    plt.grid(ls='--')
    plt.legend(loc='best')
    plt.show()

def plot_validation_curve(param_range, train_scores, test_scores, title, alpha=0.1):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=2)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=2)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=2)
    plt.title(title)
    plt.grid(ls='--')
    plt.ylim(0.0, 1.1)
    plt.xlabel('Parameter Value')
    plt.ylabel('F1-score')
    plt.legend(loc='best')
    plt.show()

def plot_auc_roc(y_test, y_pred):
    plt.figure(figsize=(16, 10))
    auroc = roc_auc_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auroc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  # diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

# ==============================================================================
# t-SNE , high dimensional visualization ..
# ==============================================================================

def tsne(alignedShapes, y, embed_count):
    """
    This function uses tensorbard to visualize high-dimensional data.
    For calling the tensorboard you should be in that drive and call the entire path
    "tensorboard --logdir=path_to_code/mnist-tensorboard/log-1 --port=6006"

    to run:
        tensorboard --logdir=/home/wj17/Documents/phdcoding/tensorboard/log-1 --port=6006
        then type: localhost:6006 in url

    INPUT:
        alignedShapes : (num_features x num_shapes)
        y : (1 x num_shapes)
        embed_count : number of shapes to visualize usin tsne
    """
    # split into train/test MESHWISE ..
    num_meshes = 275
    num_views = 33
    split_perc = 0.75 # 50/25/25 for train/cv/testss

    x_data, y_data, y_trcv_colors, y_test_colors = split_data_meshwise(alignedShapes.T, y, split_perc,
                                                num_meshes, num_views)

    # assign data .. (need flattening)
    x_train = np.reshape(x_data[0], (x_data[0].shape[0] * x_data[0].shape[1], x_data[0].shape[2]))
    y_trcv_colors = np.reshape(y_trcv_colors, (y_trcv_colors.shape[0] * y_trcv_colors.shape[1], y_trcv_colors.shape[2]))
    y_train = y_data[0].ravel()

    x_test = np.reshape(x_data[1], (x_data[1].shape[0] * x_data[1].shape[1], x_data[1].shape[2]))
    y_test_colors = np.reshape(y_test_colors, (y_test_colors.shape[0] * y_test_colors.shape[1], y_test_colors.shape[2]))
    y_test = y_data[1].ravel()

    # shuffle within data just to be sure ..
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_test, y_test = shuffle(x_test, y_test, random_state=42)
    y_trcv_colors = shuffle(y_trcv_colors, random_state=42)
    y_test_colors = shuffle(y_test_colors, random_state=42)

    # embedcount
    x_train_embed = x_train[:embed_count, :]
    y_train_embed = y_train[:embed_count]
    y_trcv_colors_embed = y_trcv_colors[:embed_count, :]

    # log-1 directory ..
    path = '/home/wj17/Documents/phdcoding'
    logdir = path + '/tensorboard/log-1'

    # save labels as meta data for visualization ..
    print('saving metadata labels ..')
    metadata_path = os.path.join(logdir, 'metadata.tsv')
    with open(metadata_path, 'w') as meta:
        meta.write('Index\tLabel\n')
        for index, label in enumerate(y_train_embed):
            meta.write('{}\t{}\n'.format(index, str(label)))

    # define small and big image sizes ..
    num_shapes, num_landmarks = x_train_embed.shape
    rows, cols = 28, 28 # dims of each small image ..

    # create sprite image ..
    print('converting plt.figures to pil images ..')
    images_dir = '/home/wj17/Documents/phdcoding/tensorboard/log-1/images/'
    dpi = 100
    plt.rcParams['figure.dpi']= dpi #set to 300 dpi

    meanShape = np.mean(x_train_embed, axis=0)

    for i in range(num_shapes):
        print('i =', i)
        w = rows/float(dpi)
        h = cols/float(dpi)
        fig = plt.figure(figsize=(w,h))
        # fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x_train_embed[i,1::2], -x_train_embed[i,0::2], color = (0,0,1), linewidth = 1)

        # # plot also mean shape for comparison purposes
        # ax.plot(meanShape[1::2], -meanShape[0::2], color = (0,0,0), linestyle='dashed', linewidth = 1, alpha=0.6)

        # # color based on which mesh every shape belongs to..
        # ax.patch.set_color(tuple(y_trcv_colors_embed[i]))
        # ax.patch.set_alpha(.7)

        ax.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.savefig(images_dir + 'im' + str(i), figure='dpi')
        fig.clf()
        plt.close(fig)

    # create sprite image ..
    print('creating sprite image .. ')
    index = 0
    sprite_dim = int(np.sqrt(embed_count))

    sprite_image = Image.new(mode='RGB',
                    size=(rows * sprite_dim, cols * sprite_dim),
                    color='white')   #white background

    for i in range(sprite_dim):
        for j in range(sprite_dim):
            fnm = images_dir + 'im' + str(i) + '.png'
            img = Image.open(fnm)
            sprite_image.paste(img, (i * cols,j * rows)) # order is reverse due to PIL saving
            index += 1

    # setup the write and embedding tensor ..
    summary_writer = tf.summary.FileWriter(logdir)

    # run the sesion to create the model check point
    print('begin tf session ..')
    with tf.Session() as sesh:
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = 'shapes-embeddings'
        embedding.metadata_path = metadata_path
        embedding.sprite.image_path = os.path.join(logdir, 'sprite.png')
        embedding.sprite.single_image_dim.extend([rows, cols])
        plt.imsave(embedding.sprite.image_path, sprite_image, cmap='gray')
        embedding_var = tf.Variable(x_train_embed, name='shapes-embeddings')
        sesh.run(embedding_var.initializer)
        projector.visualize_embeddings(summary_writer, config)

        saver = tf.train.Saver({'shapes-embeddings': embedding_var})
        saver.save(sesh, os.path.join(logdir, 'model.ckpt'))

def tsne_test():

    path = '/home/wj17/Documents/phdcoding' # import the data and split into X and Y

    test_data = np.array(pd.read_csv(path + '/fashion-mnist_test.csv'), dtype='float32')

    embed_count = 1600
    x_test = test_data[:embed_count, 1:] / 255
    y_test = test_data[:embed_count, 0]

    logdir = r'C:\Users\Fashion_MNIST\logdir'  # you will need to change this!!!
    # setup the write and embedding tensor
    summary_writer = tf.summary.FileWriter(logdir)

    embedding_var = tf.Variable(x_test, name='fmnist_embedding')

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    embedding.metadata_path = os.path.join(logdir, 'metadata.tsv')
    embedding.sprite.image_path = os.path.join(logdir, 'sprite.png')
    embedding.sprite.single_image_dim.extend([28, 28])

    projector.visualize_embeddings(summary_writer, config)
    # run the sesion to create the model check point

    with tf.Session() as sesh:
        sesh.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sesh, os.path.join(logdir, 'model.ckpt'))
    # create the sprite image and the metadata file

    rows = 28
    cols = 28

    label = ['t_shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot']

    sprite_dim = int(np.sqrt(x_test.shape[0]))

    sprite_image = np.ones((cols * sprite_dim, rows * sprite_dim))

    index = 0
    labels = []
    for i in range(sprite_dim):
        for j in range(sprite_dim):

            labels.append(label[int(y_test[index])])

            sprite_image[
                i * cols: (i + 1) * cols,
                j * rows: (j + 1) * rows
            ] = x_test[index].reshape(28, 28) * -1 + 1

            index += 1

    with open(embedding.metadata_path, 'w') as meta:
        meta.write('Index\tLabel\n')
        for index, label in enumerate(labels):
            meta.write('{}\t{}\n'.format(index, label))

    plt.imsave(embedding.sprite.image_path, sprite_image, cmap='gray')
    plt.imshow(sprite_image, cmap='gray')
    plt.show()

def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    return ZCAMatrix

# ==============================================================================
# LDA ..
# ==============================================================================

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import colors

def plot_data(lda, X, y, y_pred):
    plt.title('Linear Discriminant Analysis')

    tp = (y == y_pred)  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]

    alpha = 0.5

    # class 0: dots
    plt.plot(X0_tp[:, 0], X0_tp[:, 1], 'o', alpha=alpha,
             color='red', markeredgecolor='k')
    plt.plot(X0_fp[:, 0], X0_fp[:, 1], '*', alpha=alpha,
             color='#990000', markeredgecolor='k')  # dark red

    # class 1: dots
    plt.plot(X1_tp[:, 0], X1_tp[:, 1], 'o', alpha=alpha,
             color='blue', markeredgecolor='k')
    plt.plot(X1_fp[:, 0], X1_fp[:, 1], '*', alpha=alpha,
             color='#000099', markeredgecolor='k')  # dark blue

    # class 0 and 1 : areas
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.))
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')

    # means
    plt.plot(lda.means_[0][0], lda.means_[0][1],
             'o', color='black', markersize=10, markeredgecolor='k')
    plt.plot(lda.means_[1][0], lda.means_[1][1],
             'o', color='black', markersize=10, markeredgecolor='k')

    return splot

def plot_ellipse(splot, mean, cov, color):
    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    # filled Gaussian at 2 standard deviation
    ell = mpl.patches.Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5,
                              180 + angle, facecolor=color,
                              edgecolor='yellow',
                              linewidth=2, zorder=2)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)
    splot.set_xticks(())
    splot.set_yticks(())

def plot_lda_cov(lda, splot):
    plot_ellipse(splot, lda.means_[0], lda.covariance_, 'red')
    plot_ellipse(splot, lda.means_[1], lda.covariance_, 'blue')

def lda(X, y):
    """
    X : aligned shapes (n_shapes x n_features) / whitened or not whitened
    y : labels

    """
    # split into train/test MESHWISE ..
    num_meshes = 275
    num_views = 33
    split_perc = 0.50 # (50/25)/25 for (train, cv) - test
    x_data, y_data,_,_ = split_data_meshwise(X, y, split_perc,
                                                num_meshes, num_views)

    # assign data .. (need flattening)
    x_train = np.reshape(x_data[0], (x_data[0].shape[0] * x_data[0].shape[1], x_data[0].shape[2]))
    y_train = y_data[0].ravel()

    x_test = np.reshape(x_data[1], (x_data[1].shape[0] * x_data[1].shape[1], x_data[1].shape[2]))
    y_test = y_data[1].ravel()

    # shuffle within data just to be sure ..
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_test, y_test = shuffle(x_test, y_test, random_state=42)

    # classifier lda ..
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # performance measure ..
    performance_print(y_pred, y_test)

    # assign color map for plots ..
    cmap = colors.LinearSegmentedColormap('red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
    plt.cm.register_cmap(cmap=cmap)

    # plot lda ..
    plt.figure(figsize=(6,6))
    splot = plot_data(clf, x_test, y_test, y_pred)
    plot_lda_cov(lda, splot)
    plt.axis('tight')
    plt.show()
