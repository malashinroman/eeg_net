#!/usr/bin/env python
# # -*- coding: utf-8 -*-
import csv
import pyedflib
import numpy as np

from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from stacklineplot import stackplot

import pywt
import xlwt

EEG_SHIFT = 168
FS = 250 # discretization frequency 250 Hz
TEST_RATIO = 0.3
C = 1.0  # SVM regularization parameter
PATIENTS_NUM = 21
clf_names = ["svc", "lin_svc", "rfb_svc", "poly_svc"]

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

def wenergy(coeffs, k):
    return np.sqrt(np.sum(np.array(coeffs[len(coeffs) - k - 1]) ** 2) / len(coeffs[len(coeffs) - k - 1]))

def wentropy(coeffs, k):
    Ej = np.sum(np.abs(np.array(coeffs[len(coeffs) - k - 1])))
    Etot = 0
    for i in range(1, len(coeffs)):
        Etot = Etot + np.abs(np.sum(coeffs[i]))
    Pj = Ej / Etot
    spentropy = -1*np.sum(np.multiply(Pj,np.log(Pj)), axis=0)
    return spentropy

def wstds(coeffs, k):
    return np.std(coeffs[len(coeffs) - k - 1], axis=0)

def extractDWTFeatures(EEG, times, interval, channels, mode):
    """
    mode:
    0 - energy
    1 - entropy
    2 - std
    3 - energy & entropy
    4 - energy & std
    5 - entropy & std
    6 - energy & entropy & std
    """
    nchannels = len(channels)
    ntrials = len(times)
    # print("mode: %d" % (mode))
    db2 = pywt.Wavelet('db2')
    LVL_NUM = 3 # levels in wavelet decomposition
    if mode <= 2:
        nfeats = 3
    elif 3 <= mode <= 5:
        nfeats = 6
    else:
        nfeats = 9

    X = np.zeros((ntrials, nchannels*nfeats))
    for i in np.arange(ntrials):
        signal = []
        start = times[i] + int(interval[0] * FS)
        end = times[i] + int(interval[1] * FS)
        for ch in channels:
            slice = EEG[ch][start:end]
            coeffs = pywt.wavedec(slice, db2, mode='constant', level=LVL_NUM)

            if mode == 0 or mode == 3 or mode == 4 or mode == 6:
                feat1 = np.zeros((1, LVL_NUM))
                for j in range(LVL_NUM):
                    feat1[0][j] = wenergy(coeffs, j)
                signal.extend(feat1.flatten().tolist())

            if mode == 1 or mode == 3 or mode == 5 or mode == 6:
                feat2 = np.zeros((1, LVL_NUM))
                for j in range(LVL_NUM):
                    feat2[0][j] = wentropy(coeffs, j)
                signal.extend(feat2.flatten().tolist())

            if mode == 2 or mode == 4 or mode == 5 or mode == 6:
                feat3 = np.zeros((1, LVL_NUM))
                for j in range(LVL_NUM):
                    feat3[0][j] = wstds(coeffs, j)
                signal.extend(feat3.flatten().tolist())

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
        elif features_type.startswith("DWT"):
            # print("Using DWT coeff. as features")
            mode = int(features_type[len(features_type)-1])
            cur_X = extractDWTFeatures(EEG[i], times[i], interval, channels, mode)

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
    print("Size of training set: %d" % (ntrials-TEST_SIZE))
    print("Size of test set: %d" % (TEST_SIZE))
    X_train = X[0:ntrials-TEST_SIZE]
    X_test = X[ntrials-TEST_SIZE:]
    y_train = y[0:ntrials-TEST_SIZE]
    y_test = y[ntrials-TEST_SIZE:]

    accuracies = trainSVM(X_train, y_train, X_test, y_test, class_division)
    return accuracies


def trainSVM(X_train, y_train, X_test, y_test, class_division):
    if class_division == "4":
        clf = RandomForestClassifier(min_samples_leaf=20)
        y_score = clf.fit(X_train, y_train).score(X_test, y_test)
        return y_score
    else:
        scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
        X_train = scaling.transform(X_train)
        X_test = scaling.transform(X_test)
        svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
        rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
        poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)
        lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train)
        accuracies = [0, 0, 0, 0]
        for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
            predicted = clf.predict(X_test)
            # get the accuracy
            accuracies[i] = accuracy_score(y_test, predicted)
            # print ("%s: %.2f" % (clf_names[i], accuracies[i]))
        return accuracies


def formattedOutput(message, accuracies, class_division):
    print(message)
    if class_division == "4":
        print("4-class score: %.2f" % (accuracies))
    else:
        for i in np.arange(len(accuracies)):
            print ("%s: %.2f" % (clf_names[i], accuracies[i]))


def getSingleClassification(dataset_folder, eeg_num, excluded, features_type, interval, channels, class_division):
    if features_type == "MEAN":
        accuracy = classifyEEGMean(dataset_folder, eeg_num, excluded, interval, channels)
        print("Mean value accuracy (SINGLE): %.2f" % (accuracy))
    else:
        accuracies = classifyEEG(dataset_folder, eeg_num, excluded, features_type, interval, channels, class_division)
        formattedOutput("SINGLE PATIENT VALUE:", accuracies, class_division)


def getAverageClassification(dataset_folder, excluded, features_type, interval, channels, class_division):
    output = []
    if features_type == "MEAN":
        mean_acc = 0
        for i in np.arange(PATIENTS_NUM):
            eeg_num = [i]
            accuracy = classifyEEGMean(dataset_folder, eeg_num, excluded, interval, channels)
            if class_division != "4":
                acc =  [accuracy, accuracy, accuracy, accuracy]
            else:
                acc = accuracy
            output.append(acc)
            mean_acc = mean_acc + accuracy
        mean_acc = float(mean_acc) / PATIENTS_NUM
        if class_division != "4":
            mean_acc = [mean_acc, mean_acc, mean_acc, mean_acc]
        print("Mean value accuracy (AVERAGE): %.2f" % (accuracy))
    else:
        if class_division == "4":
            mean_acc = 0
        else:
            mean_acc = [0, 0, 0, 0]
        for i in np.arange(PATIENTS_NUM):
            eeg_num = [i]
            accuracies = classifyEEG(dataset_folder, eeg_num, excluded, features_type, interval, channels, class_division)
            if class_division == "4":
                mean_acc = mean_acc + accuracies
            else:
                mean_acc = [sum(x) for x in zip(mean_acc, accuracies)]
            output.append(accuracies)
        if class_division == "4":
            mean_acc = float(mean_acc) / PATIENTS_NUM
        else:
            mean_acc[:] = [float(x) / PATIENTS_NUM for x in mean_acc]
        formattedOutput("AVERAGE VALUE:", mean_acc, class_division)

    output.append(mean_acc)
    return output


def getGlobalClassification(dataset_folder, excluded, features_type, interval, channels, class_division):
    eeg_nums = range(PATIENTS_NUM)
    if features_type == "MEAN":
        accuracy = classifyEEGMean(dataset_folder, eeg_nums, excluded, interval, channels)
        print("Mean value accuracy (GLOBAL): %.2f" % (accuracy))
        if class_division == "4":
            return accuracy
        else:
            return [accuracy, accuracy, accuracy, accuracy]
    else:
        accuracies = classifyEEG(dataset_folder, eeg_nums, excluded, features_type, interval, channels, class_division)
        formattedOutput("GLOBAL VALUE: ", accuracies, class_division)
        return accuracies


def classifyEEGMean(dataset_folder, eeg_nums, excluded, interval, channels):
    EEG = []
    times = []
    labels = []
    for nums in eeg_nums:
        cur_EEG, cur_times, cur_labels = loadSingleEEG(dataset_folder, nums, excluded, "2A")
        EEG.append(cur_EEG)
        times.append(cur_times)
        labels.append(cur_labels)

    nticks = int(interval[1] * FS) - int(interval[0] * FS)

    for i in np.arange(len(EEG)):
        # get labels column as y
        cur_y = np.asarray(labels[i])
        cur_y = cur_y.reshape((cur_y.shape[0], 1))

        ntrials = cur_y.shape[0]
        cur_X = np.zeros((ntrials, nticks))
        for j in np.arange(ntrials):
            signal = np.zeros((1, nticks))
            start = times[i][j] + int(interval[0] * FS)
            end = times[i][j] + int(interval[1] * FS)
            for ch in channels:
                slice = EEG[i][ch][start:end]
                signal = signal + slice
            cur_X[j] = np.divide(signal, len(channels))

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
    print("Size of training set: %d" % (ntrials-TEST_SIZE))
    print("Size of test set: %d" % (TEST_SIZE))
    X_train = X[0:ntrials-TEST_SIZE]
    X_test = X[ntrials-TEST_SIZE:]
    y_train = y[0:ntrials-TEST_SIZE]
    y_test = y[ntrials-TEST_SIZE:]

    # determining mean value for animate stimuls
    avgAnimCurve = np.zeros((1, nticks))
    k = 0
    for i in np.arange(len(y_train)):
        if y_train[i] == 0: # for animate stimuls
            avgAnimCurve = avgAnimCurve + X_train[i]
            k = k + 1
    avgAnimCurve = np.divide(avgAnimCurve, k)
    meanAnim = avgAnimCurve.mean()

    # classifying test set
    correct = 0
    for i in np.arange(len(y_test)):
        cur_avg = X_test[i].mean()
        label = 0
        if cur_avg > meanAnim:
            label = 0
        else:
            label = 1
        if label == y_test[i]:
            correct = correct + 1

    accuracy = float(correct) / TEST_SIZE
    return accuracy


def conductExperiments(dataset):
    excluded = loadExcludedLabels(dataset + '/exlabels.csv')
    ch1 = [0, 2, 3, 4, 5, 6, 7] # Frontal
    ch2 = [14, 15, 16] # Parietal
    ch3 = [8, 12, 13, 17] # Temporal
    ch4 = [9, 10, 11] # Central
    ch5 = [18, 19, 20] # Occipital
    ch6 = range(21) # all channels

    chnames = ["Frontal", "Parietal", "Temporal", "Central", "Occipital", "ALL"]
    chconts = ["(Fp1,Fp2,F7,F3,Fz,F4,F8)", "(P3,Pz,P4)", "(T3,T4,T5,T6)", "(С3,Сz,С4)", "(O1,Oz,O2)", "(Fp1-O2)"]
    cdivs = ["2A", "2F", "4"]
    intervals = [[0.15, 0.30], [0.30, 0.48], [0.52, 0.65], [0.65, 0.8], [0, 1.1]]
    features = ["RAW", "MEAN", "DWT0", "DWT1", "DWT2", "DWT3", "DWT4", "DWT5", "DWT6"]
    featnames = ["RAW", "MEAN", "ENERGY", "ENTROPY", "STD", "ENERGY & ENTROPY", "ENERGY & STD", "ENTROPY & STD", "ENERGY & ENTROPY & STD"]
    channels = [ch1, ch2, ch3, ch4, ch5, ch6]

    book = xlwt.Workbook(encoding="utf-8")

    for cdiv in cdivs:
        sheet = book.add_sheet(cdiv)
        print(cdiv)
        i = 0
        for interval in intervals:
            print("Interval %.2f - %.2f" % (interval[0], interval[1]))
            sheet.write(i*300, 0, "Interval")
            sheet.write(i*300, 1, "%.2f - %.2f" % (interval[0], interval[1]))
            c = 0
            for chs in channels:
                print("Channels %s" % (chnames[c]))
                sheet.write(i*300 + c*49 + 1, 0, "Channels")
                sheet.write(i*300 + c*49 + 1, 1, chnames[c])
                sheet.write(i*300 + c*49 + 1, 2, chconts[c])
                if cdiv != "4":
                    for k in range(len(clf_names)):
                        sheet.write(i*300 + c*49 + k*11 + 2, 2, "Classifier")
                        sheet.write(i*300 + c*49 + k*11 + 2, 3, clf_names[k])
                        sheet.write(i*300 + c*49 + k*11 + 3, 0, "#")
                        sheet.write(i*300 + c*49 + k*11 + 3, 1, "Features")
                        for m in range(PATIENTS_NUM):
                            sheet.write(i*300 + c*49 + k*11 + 3, 2 + m, "D%d" % (168 + m))
                        sheet.write(i*300 + c*49 + k*11 + 3, 23, "AVG")
                        sheet.write(i*300 + c*49 + k*11 + 3, 24, "GLOBAL")
                        for n in range(len(features)):
                            sheet.write(i*300 + c*49 + k*11 + 4 + n, 0, n)
                            sheet.write(i*300 + c*49 + k*11 + 4 + n, 1, featnames[n])
                else:
                    sheet.write(i*300 + c*49 + 2, 2, "Classifier")
                    sheet.write(i*300 + c*49 + 2, 3, "RandomForest")
                    sheet.write(i*300 + c*49 + 3, 0, "#")
                    sheet.write(i*300 + c*49 + 3, 1, "Features")
                    for m in range(PATIENTS_NUM):
                        sheet.write(i*300 + c*49 + 3, 2 + m, "D%d" % (168 + m))
                    sheet.write(i*300 + c*49 + 3, 23, "AVG")
                    sheet.write(i*300 + c*49 + 3, 24, "GLOBAL")
                    for n in range(len(features)):
                        sheet.write(i*300 + c*49 + 4 + n, 0, n)
                        sheet.write(i*300 + c*49 + 4 + n, 1, featnames[n])
                f = 0
                for feats in features:
                    print("Features: %s" % (feats))
                    output_a = getAverageClassification(dataset, excluded, feats, interval, chs, cdiv)
                    output_g = getGlobalClassification(dataset, excluded, feats, interval, chs, cdiv)
                    if cdiv != "4":
                        for z in range(len(output_a)): # z - number of patient
                            for y in range(len(output_a[z])): # y - number of classifier
                                # print("row: %d, column: %d" % (i*300 + c*49 + y*11 + 4 + f, 2 + z))
                                sheet.write(i*300 + c*49 + y*11 + 4 + f, 2 + z, output_a[z][y])
                        for y in range(len(output_g)):
                            sheet.write(i*300 + c*49 + y*11 + 4 + f, 24, output_g[y])
                    else:
                        for z in range(len(output_a)): # z - number of patient
                            sheet.write(i*300 + c*49 + 4 + f, 2 + z, output_a[z])
                        sheet.write(i*300 + c*49 + 4 + f, 24, output_g)
                    f += 1

                c += 1
            i += 1

    book.save("%s.xls" % (dataset))

def main():
    dataset_folder = "filtered_dataset"
    excluded = loadExcludedLabels(dataset_folder + '/exlabels.csv')
    sample_eeg = [1]
    features = "RAW"
    interval = [0.15, 0.2]
    ch1 = [0, 2, 3, 4, 5, 6, 7]
    cdiv = "2A"
    #getSingleClassification(dataset_folder, sample_eeg, excluded, features, interval, ch1, cdiv)

    # getAverageClassification(dataset_folder, excluded, features, interval, ch1, cdiv)

    # getGlobalClassification(dataset_folder, excluded, features, interval, ch1, cdiv)

    #chnl_names = ["Frontal [Fp1, Fp2, F7, F3, Fz, F4, F8]", "Pariental [P3,Pz,P4]", "Temporal [T3,T4,T5,T6]", "Central [С3,Сz,С4]"]
    #for j, channels in enumerate((ch1, ch2, ch3, ch4)):
        #print("========================================")
        #print(chnl_names[j])

    conductExperiments(dataset_folder)



def plotEDFdata(eeg_path):
    f = pyedflib.EdfReader(eeg_path)
    n_channels = f.signals_in_file # number of EEG channels
    signal_labels = f.getSignalLabels() # names of EEG channels

    n_min = f.getNSamples()[0]
    sigbufs = [np.zeros(f.getNSamples()[i]) for i in np.arange(n_channels)]
    for i in np.arange(n_channels):
        sigbufs[i] = f.readSignal(i)
        if n_min < len(sigbufs[i]):
            n_min = len(sigbufs[i])
    f._close()
    del f
    n_plot = np.min((n_min, 2800))
    sigbufs_plot = np.zeros((n_channels, n_plot))
    for i in np.arange(n_channels):
        sigbufs_plot[i,:] = sigbufs[i][:n_plot]

    stackplot(sigbufs_plot[:, :n_plot], ylabels=signal_labels)


if __name__ == '__main__':
    main()
