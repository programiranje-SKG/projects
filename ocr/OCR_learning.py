#!/usr/bin/env python
# from __future__ import print_function

import os
# from task2.msg import digits
from os import listdir
from os.path import isfile, join
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from joblib import dump

# roslib.load_manifest('exercise4')

dictm = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# The object that we will pass to the markerDetect function

params = cv2.aruco.DetectorParameters_create()

# To see description of the parameters
# https://docs.opencv.org/3.3.1/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html

# You can set these parameters to get better marker detections
params.adaptiveThreshConstant = 25
adaptiveThreshWinSizeStep = 2

counter = 0


class PCA_learner():
    def __init__(self, PCA, clf, h, w):
        self.PCA = PCA
        self.clf = clf
        self.h = h
        self.w = w

    def predict(self, X_classify):
        assert X_classify.shape[1] == self.h*self.w
        X_test_pca = self.PCA.transform(X_classify)
        y_pred = self.clf.predict(X_test_pca)
        return y_pred


def load_image(source, img_name, size=1, offset=0, filter=None):
    # Create a blank 640x480 black image
    img = np.zeros((480, 640, 3), np.uint8)
    # Fill image with color(set each pixel to some gray color)
    img[:] = (167, 152, 139)

    # convert image to 640 x 480
    img_input = cv2.imread(os.path.join(source, img_name), cv2.IMREAD_COLOR)
    # print(img_input.shape)
    img_input = cv2.resize(img_input, (318, 450))
    # fill with mock image
    img[15:465, 161:479] = img_input

    # cv2.imshow('image', img)
    # cv2.waitKey(30)
    return img


def export_dataset(source_path, dest_path):
    # counter = 0
    for i in range(10):
        for j in range(10):
            img_name = "ring_dig_{}{}.png".format(i, j)
            img = load_image(source_path, img_name)
            learn_one_image(dest_path, img, img_name)


def learn_one_image(destination, cv_image, img_name, size=-1, offset=-1):
    # Set the dimensions of the image
    # dims = cv_image.shape
    global counter

    # Tranform image to gayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Do histogram equlization
    img = cv2.equalizeHist(gray)

    # Binarize the image
    ret, thresh = cv2.threshold(img, 50, 255, 0)

    corners, ids, rejected_corners = cv2.aruco.detectMarkers(cv_image, dictm, parameters=params)

    # Increase proportionally if you want a larger image
    image_size = (351, 248, 3)
    marker_side = 50

    img_out = np.zeros(image_size, np.uint8)
    out_pts = np.array([[marker_side / 2, img_out.shape[0] - marker_side / 2],
                        [img_out.shape[1] - marker_side / 2, img_out.shape[0] - marker_side / 2],
                        [marker_side / 2, marker_side / 2],
                        [img_out.shape[1] - marker_side / 2, marker_side / 2]])

    src_points = np.zeros((4, 2))
    cens_mars = np.zeros((4, 2))

    if not ids is None:
        if len(ids) == 4:
            # print('4 Markers detected')

            for idx in ids:
                # Calculate the center point of all markers
                cors = np.squeeze(corners[idx[0] - 1])
                cen_mar = np.mean(cors, axis=0)
                cens_mars[idx[0] - 1] = cen_mar
                cen_point = np.mean(cens_mars, axis=0)

            for coords in cens_mars:
                #  Map the correct source points
                if coords[0] < cen_point[0] and coords[1] < cen_point[1]:
                    src_points[2] = coords
                elif coords[0] < cen_point[0] and coords[1] > cen_point[1]:
                    src_points[0] = coords
                elif coords[0] > cen_point[0] and coords[1] < cen_point[1]:
                    src_points[3] = coords
                else:
                    src_points[1] = coords

            h, status = cv2.findHomography(src_points, out_pts)
            img_out = cv2.warpPerspective(cv_image, h, (img_out.shape[1], img_out.shape[0]))
            # cv2.imshow('Warped image big', img_out)

            ################################################
            #### Extraction of digits starts here
            ################################################

            # Cut out everything but the numbers
            # print(np.shape(img_out))
            img_out = img_out[125:221, 52:202, :]

            # Convert the image to grayscale
            img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)

            # Option 1 - use ordinairy threshold the image to get a black and white image
            # ret,img_out = cv2.threshold(img_out,100,255,0)

            # Option 1 - use adaptive thresholding
            # img_out = cv2.adaptiveThreshold(img_out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,
            #                                 5)

            ret, img_out = cv2.threshold(img_out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            img_left = img_out[:, :69]
            img_right = img_out[:, 81:]
            for i in range(-10, 10):
                for j in range(-10, 10):
                    rows, cols = img_left.shape

                    M = np.float32([[1, 0, i], [0, 1, j]])
                    img_left_out = cv2.warpAffine(img_left, M, (cols, rows), borderValue=255)

                    left_name = "img_{}_{}.png".format(img_name[9], counter)
                    counter += 1
                    #right_name = "img_{}_{}.png".format(img_name[10], counter)
                    #counter += 1

                    # write both images
                    cv2.imwrite(os.path.join(destination, left_name), img_left_out)
                    #cv2.imwrite(os.path.join(destination, right_name), img_right)
                    print('path: {}'.format(destination))
                    print("imwrite of '{}' succesful".format(left_name))
                    #print("imwrite of '{}' succesful".format(right_name))

            # size of img out: 96x145
            # cv2.imshow('left: {}'.format(left_name), img_left)
            # cv2.imshow('right: {}'.format(right_name), img_right)
            # assert np.shape(img_right) == np.shape(img_left)
            # print(np.shape(img_left))
            # print(np.shape(img_right))
            # print(np.shape(img_left))
            # cv2.waitKey(0)

            # Use Otsu's thresholding
            # ret,img_out = cv2.threshold(img_out,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # Pass some options to tesseract

            # Visualize the image we are passing to Tesseract
            # cv2.imshow('Warped image', img_out)
            # cv2.waitKey(0)

            # Extract text from image

        else:
            print('The number of markers is not ok:', len(ids))
            raise IOError
    else:
        print('No markers found')
        raise IOError


def create_model(path_dataset, path_model, name_model='PCA_learner'):

    # just for zero
    # TODO create model for every number, than classify to model with higher probability
    filenames = [[] for i in range(10)]
    filenames_all = [f for f in listdir(path_dataset) if isfile(join(path_dataset, f))]

    dataset = []
    h,w, channels = cv2.imread(os.path.join(path_dataset, filenames_all[0])).shape
    # print(h, w, channels)
    for image_name in filenames_all:

        # for image_name in digit_filenames:
        img_cv = cv2.imread(os.path.join(path_dataset, image_name))
        temp_h, temp_w, temp_c = img_cv.shape
        assert temp_h == h
        assert temp_w == w
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        img_np = np.array(img_cv)
        img_np = img_np.flatten()
        dataset.append(img_np)
    X = np.array(dataset)
    # X.shape (202, 6624)
    # y.shape (202,)
    y = np.array([name[4] for name in filenames_all])

    n_features = X.shape[1]
    n_samples = X.shape[0]

    # the label to predict is the id of the person
    # y = lfw_people.target
    target_names = [str(i) for i in range(10)]
    n_classes = 10
    # h = 96
    # w = 69

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)

    # #############################################################################
    # Split into a training set and a test set using a stratified k fold

    # split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # #############################################################################
    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 150

    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))
    t0 = time()
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))

    eigenfaces = pca.components_.reshape((n_components, h, w))

    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))

    # #############################################################################
    # Train a SVM classification model

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                       param_grid, cv=5)#, iid=False)
    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    # #############################################################################
    # Quantitative evaluation of the model quality on the test set

    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)

    print("done in %0.3fs" % (time() - t0))

    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred))

    # #############################################################################
    # Qualitative evaluation of the predictions using matplotlib

    def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
        """Helper function to plot a gallery of portraits"""
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())
        plt.savefig('eigendigits.png')

    # plot the result of the prediction on a portion of the test set

    def title(y_pred, y_test, target_names, i):
        pred_name = y_pred[i] # target_names[y_pred[i]].rsplit(' ', 1)[-1]
        true_name = y_test[i] # target_names[y_test[i]].rsplit(' ', 1)[-1]
        return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

    prediction_titles = [title(y_pred, y_test, target_names, i)
                         for i in range(y_pred.shape[0])]

    plot_gallery(X_test, prediction_titles, h, w)

    # plot the gallery of the most significative eigenfaces

    eigenface_titles = ["eigendigit %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)
    plt.show()
    print(X_train.shape)
    # shape: (samples, features)
    learner = PCA_learner(PCA=pca, clf=clf, h=h, w=w)

    dump(learner, os.path.join(path_model, '{}.model'.format(name_model)))


def export_dataset_v2(source_path, dest_path):

    # source_path = "/home/team_eta/ROS/OCR/rings_digits"
    # dest_path = "/home/team_eta/ROS/OCR/test"
    for i in range (10):
        img_name = "ring_dig_{}1.png".format(i)
        img = load_image(source_path, img_name)
        learn_one_image(dest_path, img, img_name)


def test():
    img_name = "ring_dig_11.png"
    source_path = "/home/team_eta/ROS/OCR/rings_digits"
    dest_path = "/home/team_eta/ROS/OCR/test"
    img = load_image(source_path, img_name)
    learn_one_image(dest_path, img, img_name)


def learn(image_source, dataset_path, model_path, model_name):
    export_dataset_v2(image_source, dataset_path)
    create_model(dataset_path, model_path, model_name)


def main():
    source_path = "rings_digits"
    dataset_path = "dataset"
    model_path = "models"

    learn(source_path, dataset_path, model_path, model_name='PCA_learner')
    # create_model(dataset_path, model_path, name_model='PCA_learner')
    # export_dataset_v2()


if __name__ == '__main__':
    main()
