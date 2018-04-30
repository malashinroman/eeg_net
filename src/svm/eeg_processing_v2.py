#!/usr/bin/env python
# # -*- coding: utf-8 -*-
from __future__ import print_function


import csv
import pyedflib
import numpy as np
import itertools
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from pyeeg import *
from pylab import *


EEG_SHIFT = 168
FS = 250 # discretization frequency 250 Hz
TEST_RATIO = 0.3
PATIENTS_NUM = 21
Band = [2*i+1 for i in xrange(0, 43)]		## 0.5~85 Hz

def loadSingleEEG(dataset_folder, eeg_num, excluded, class_division):
    eeg_path = "%s/edf_data/D%07d.EDF" % (dataset_folder, eeg_num + EEG_SHIFT)
    labels_path = "%s/labels/D%07d.TXT" % (dataset_folder, eeg_num + EEG_SHIFT)
    print("EDF File: " + eeg_path)
    # print("labels File: " + labels_path)
    # read EEG data in EDF format
    f = pyedflib.EdfReader(eeg_path)
    n_channels = f.signals_in_file # number of EEG channels
    n_samples = f.getNSamples()[0]
    EEG = np.zeros(shape=(n_channels, n_samples))
    for i in np.arange(n_channels):
        EEG[i][:] = f.readSignal(i)
    f._close()
    del f
    # read labels for EEG data
    times = []
    labels = []
    with open(labels_path) as fp:
        next(fp)
        trial_num = 1
        for line in fp:
            if eeg_num in excluded and trial_num in excluded[eeg_num]:
                trial_num = trial_num + 1
                continue

            values = line.strip().split(' ')
            time = int(float(values[0]) * FS)
            lbl = int(values[2])
            if class_division == "2A": # 2 classes - animate/inanimate objects
                if lbl == 49 or lbl == 51:
                    lbl = 0 # animate
                else:
                    lbl = 1 # inanimate
            elif class_division == "2F": # 2 classes - high/low frequencies
                if lbl == 49 or lbl == 50:
                    lbl = 0 # high frequencies
                else:
                    lbl = 1 # low frequencies
            elif class_division == "4": # 4 classes
                lbl = lbl - 49 # 0 - AHF, 1 - IHF, 2 - ALF, 3 - ILF
            times.append(time)
            labels.append(lbl)
            trial_num = trial_num + 1
    return (EEG, times, labels)


def classifyEEG(dataset_folder, eeg_nums, excluded, features_type, interval, channels, class_division):
    EEG = []
    times = []
    labels = []
    for nums in eeg_nums:
        cur_EEG, cur_times, cur_labels = loadSingleEEG(dataset_folder, nums, excluded, class_division)
        EEG.append(cur_EEG)
        times.append(cur_times)
        labels.append(cur_labels)

    for i in np.arange(len(EEG)):
        # get labels column as y
        cur_y = np.asarray(labels[i])
        cur_y = cur_y.reshape((cur_y.shape[0], 1))

        # get features in matrix as X
        if features_type == "RAW":
            # print("Using raw signal as features")
            cur_X = extractRAWFeatures(EEG[i], times[i], interval, channels)
        elif features_type == "PSI":
            # print("Using spectral entropy as features")
            cur_X = extractPSI(EEG[i], times[i], interval, channels)
        elif features_type == "HJORTH":
            cur_X = extractHjorth(EEG[i], times[i], interval, channels)
        else:
            # print("Using single value for each channel as features")
            cur_X = extractSingleFeature(EEG[i], times[i], interval, channels, features_type)

        if i == 0:
            y = cur_y
            X = cur_X
        else: # concatenate features and labels for multiple EEG records
            y = np.concatenate((y,cur_y))
            X = np.concatenate((X,cur_X))

    # data shuffling
    merged = np.concatenate((y,X),axis=1) # append labels column to the left
    np.random.shuffle(merged)

    # extracting shuffled data
    y = merged[:, [0]].ravel()
    X = merged[:, 1:]

    ntrials = y.shape[0]

    # splitting data into training and testing sets
    TEST_SIZE = int(TEST_RATIO * ntrials)
    X_train = X[0:ntrials-TEST_SIZE]
    X_test = X[ntrials-TEST_SIZE:]
    y_train = y[0:ntrials-TEST_SIZE]
    y_test = y[ntrials-TEST_SIZE:]

    accuracy, cnf_matrix = trainClassifier(X_train, y_train, X_test, y_test, class_division)
    return accuracy, cnf_matrix


def trainClassifier(X_train, y_train, X_test, y_test, class_division):
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
    X_train = scaling.transform(X_train)
    X_test = scaling.transform(X_test)
    if class_division == "4":
        clf = RandomForestClassifier(min_samples_leaf=20).fit(X_train, y_train)
    else:
        C = 1.0  # SVM regularization parameter
        clf = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
    predicted = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    cnf_matrix = confusion_matrix(y_test, predicted)
    return accuracy, cnf_matrix


def getSingleClassification(dataset_folder, eeg_num, excluded, features_type, interval, channels, class_division):
    accuracy, cnf_matrix = classifyEEG(dataset_folder, eeg_num, excluded, features_type, interval, channels, class_division)
    formattedOutput("SINGLE PATIENT VALUE:", accuracy, cnf_matrix, class_division)
    return accuracy, cnf_matrix


def getAverageClassification(dataset_folder, excluded, features_type, interval, channels, class_division):
    mean_acc = 0
    mean_cnf = []
    for i in np.arange(PATIENTS_NUM):
        eeg_num = [i]
        accuracy, cnf_matrix = classifyEEG(dataset_folder, eeg_num, excluded, features_type, interval, channels, class_division)
        mean_acc = mean_acc + accuracy
        if i == 0:
            mean_cnf = cnf_matrix
        else:
            mean_cnf = mean_cnf + cnf_matrix
    mean_acc = float(mean_acc) / PATIENTS_NUM
    mean_cnf = mean_cnf / PATIENTS_NUM
    formattedOutput("AVERAGE VALUE:", mean_acc, mean_cnf, class_division)
    return accuracy, cnf_matrix

def getGlobalClassification(dataset_folder, excluded, features_type, interval, channels, class_division):
    eeg_nums = range(PATIENTS_NUM)
    accuracy, cnf_matrix = classifyEEG(dataset_folder, eeg_nums, excluded, features_type, interval, channels, class_division)
    formattedOutput("GLOBAL VALUE: ", accuracy, cnf_matrix, class_division)
    return accuracy, cnf_matrix

def formattedOutput(message, accuracy, cnf_matrix, class_division):
    print(message)
    if class_division == "4":
        print("4-class score: %.2f" % (accuracy))
        class_names = ["AHF", "IHF", "ALF", "ILF"]
    else:
        print("2-class score: %.2f" % (accuracy))
        class_names = ["Animate", "Inanimate"]

    #plt.figure()
    #plot_confusion_matrix(cnf_matrix, classes=class_names,
    #              title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    #plt.figure()
    #plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #              title='Normalized confusion matrix')

    #plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def extractRAWFeatures(EEG, times, interval, channels):
    nchannels = len(channels)
    nticks = int(interval[1] * FS) - int(interval[0] * FS)
    ntrials = len(times)
    X = np.zeros((ntrials, nchannels*nticks))
    for i in np.arange(ntrials):
        signal = []
        start = times[i] + int(interval[0] * FS)
        end = times[i] + int(interval[1] * FS)
        for ch in channels:
            slice = EEG[ch][start:end]
            signal.extend(slice)
        X[i] = signal
    return X

def extractPSI(EEG, times, interval, channels):
    """
    Power Spectral Intensity (PSI) and Relative Intensity Ratio (RIR)
    """
    nchannels = len(channels)
    ntrials = len(times)

    X = np.zeros((ntrials, nchannels * len(Band) * 2))
    for i in np.arange(ntrials):
        signal = []
        start = times[i] + int(interval[0] * FS)
        end = times[i] + int(interval[1] * FS)
        k = 0
        for ch in channels:
            slice = EEG[ch][start:end]
            result = bin_power(slice, Band, FS)
            for n in range(len(result[0])):
                X[i, k*nchannels + n] = result[0][n]
            for m in range(len(result[1])):
                X[i, k*nchannels + len(Band) + m] = result[1][m]
            k += 1
    return X

def extractHjorth(EEG, times, interval, channels):
    """
    Hjorth mobility and complexity
    """
    nchannels = len(channels)
    ntrials = len(times)

    X = np.zeros((ntrials, nchannels * 2))
    for i in np.arange(ntrials):
        signal = []
        start = times[i] + int(interval[0] * FS)
        end = times[i] + int(interval[1] * FS)
        k = 0
        for ch in channels:
            slice = EEG[ch][start:end]
            result = hjorth(slice)
            X[i, 2 * k] = result[0]
            X[i, 2 * k + 1] = result[1]
            k += 1
    return X

def extractSingleFeature(EEG, times, interval, channels, feat_name):
    """
    SPECT - spectral entropy

    HURST - Hurst exponent

    DFA - Detrented Fluctuation Analysis

    PFD - Petrosian Fractal Dimension

    APEN - approximate entropy

    HFD - Higuchi Fractal Dimension

    FISHER - Fisher information

    SVDEN - SVD ENTROPY
    """

    DIM = 10
    TAU = 4
    Kmax = 5


    nchannels = len(channels)
    ntrials = len(times)

    X = np.zeros((ntrials, nchannels))
    for i in np.arange(ntrials):
        signal = []
        start = times[i] + int(interval[0] * FS)
        end = times[i] + int(interval[1] * FS)
        k = 0
        for ch in channels:
            slice = EEG[ch][start:end]
            if feat_name == "SPECT":
                result = spectral_entropy(slice, Band, FS)
            elif feat_name == "HURST":
                result = hurst(slice)
            elif feat_name == "DFA":
                result = dfa(slice)
            elif feat_name == "PFD":
                result = pfd(slice)
            elif feat_name == "APEN":
                R = np.std(slice) * 0.3
                result = ap_entropy(slice, DIM, R)
            elif feat_name == "HFD":
                result = hfd(slice, Kmax)
            elif feat_name == "FISHER":
                M = embed_seq(slice, TAU, DIM)
                W = svd(M, compute_uv=0)
                W /= np.sum(W)
                result = fisher_info(slice, TAU, DIM, W)
            elif feat_name == "SVDEN":
                M = embed_seq(slice, TAU, DIM)
                W = svd(M, compute_uv=0)
                W /= np.sum(W)
                result = svd_entropy(slice, TAU, DIM, W)
            X[i, k] = result
            k += 1
    return X


def loadExcludedLabels(csvpath):
    # read file with excluded labels
    excluded = dict()
    with open(csvpath, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            subject = int(row[0]) - 1
            trial = int(row[1])
            if subject in excluded:
                excluded[subject].append(trial)
            else:
                excluded[subject] = [trial]
    return excluded


def main():
    dataset_folder = "dataset"
    excluded = loadExcludedLabels(dataset_folder + '/exlabels.csv')
    sample_eeg = [1]
    features = "RAW" # DFA doesn't work, HURST - long to calculate
    interval = [0.3, 1.1]
    ch1 = range(21)
    cdiv = "2A"
    all_features = ["RAW", "SPECT", "PSI", "HURST", "HJORTH", "PFD", "APEN", "HFD", "FISHER", "SVDEN"]
    cdivs = ["2A", "2F", "4"]
    np.set_printoptions(precision=2)

    with open("statistics.txt", "w") as text_file:
        #accuracy, cnf_matrix = getSingleClassification(dataset_folder, sample_eeg, excluded, features, interval, ch1, cdiv)
        #text_file.write("LOCAL\n")
        #text_file.write("accuracy: %.2f\n" % accuracy)
        #text_file.write("confusion_matrix:\n")
        #for i in range(len(cnf_matrix)):
            #for j in range(len(cnf_matrix[i])):
                #text_file.write("%d " % cnf_matrix[i][j])
            #text_file.write("\n")
        for cd in cdivs:
            text_file.write("%s\n" % cd)
            print("%s" % cd)
            for feat in all_features:
                text_file.write("%s\n" % feat)
                print("%s" % feat)
                accuracy, cnf_matrix = getAverageClassification(dataset_folder, excluded, feat, interval, ch1, cd)
                text_file.write("AVERAGE\n")
                text_file.write("accuracy: %.2f\n" % accuracy)
                text_file.write("confusion_matrix:\n")
                for i in range(len(cnf_matrix)):
                    for j in range(len(cnf_matrix[i])):
                        text_file.write("%d " % cnf_matrix[i][j])
                    text_file.write("\n")
        #accuracy, cnf_matrix = getGlobalClassification(dataset_folder, excluded, features, interval, ch1, cdiv)
        #text_file.write("GLOBAL\n")
        #text_file.write("accuracy: %.2f\n" % accuracy)
        #text_file.write("confusion_matrix:\n")
        #for i in range(len(cnf_matrix)):
            #for j in range(len(cnf_matrix[i])):
                #text_file.write("%d " % cnf_matrix[i][j])
            #text_file.write("\n")


    # getAverageClassification(dataset_folder, excluded, features, interval, ch1, cdiv)

    # getGlobalClassification(dataset_folder, excluded, features, interval, ch1, cdiv)


if __name__ == '__main__':
    main()
