#!/usr/bin/env python
# # -*- coding: utf-8 -*- 
""" === eeg_recognition_AN.py: script for EEG recognition (animated/inanimated stimuli) via neural networks === """

from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Activation, Reshape
from keras.optimizers import RMSprop
from keras import backend as K
from keras.layers.convolutional import Conv1D, Conv2D
from sympy.core.numbers import One
import keras.utils

from load_mat_file import load_mat_file
from keras.callbacks import TensorBoard


def create_network(input_dim, num_classes):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,)))
    seq.add(Activation('relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(num_classes, activation='relu'))
    model.add(Activation('softmax'))
    return seq


def create_network_tu(max_words, num_classes):
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


def create_network_eeg_conv1(input_dim, num_classes):
    seq = Sequential()
    seq.add(Reshape((1, input_dim,), input_shape=(input_dim,)))
    seq.add(Conv1D(32, 51, strides=25, padding='same', input_shape=(1, input_dim,), activation='relu'))
    seq.add(Activation('relu'))
    seq.add(Dense(units=128))
    seq.add(Activation('relu'))
    seq.add(Dense(units=num_classes))
    seq.add(Activation('softmax'))
    return seq


def main():
    EXP_NUM = 21 # total number of people
    EXP_SHIFT = 167 # shift in numerations
    CHANNELS_NUM = 21 # number of channels in EEG record
    sel_channels = slice(0, CHANNELS_NUM)
    all_data = np.array([])
    all_Y = np.array([])
    for eeg_num in range(1, EXP_NUM + 1):
        EEG_FILENAME = 'eegmat_selected/D0000' + str(eeg_num + EXP_SHIFT)
        EEG = load_mat_file(EEG_FILENAME, 's')
        X_data = EEG["eeg"][0][0]
        Y_data = EEG["mrk"][0][0]
        OneChannel_data = X_data[:, sel_channels, :]
        OneChannel_data = OneChannel_data.reshape(OneChannel_data.shape[2], OneChannel_data.shape[0]*OneChannel_data.shape[1])
        if eeg_num == 1:
            all_data = np.array([]).reshape(0, OneChannel_data.shape[1])
            all_Y = np.array([]).reshape(0, Y_data.shape[1])
        all_data = np.concatenate([all_data, OneChannel_data])
        all_Y = np.concatenate([all_Y, Y_data])
    
    size_of_sample = all_data.shape[1]
    number_of_samples = all_data.shape[0]
    all_Y = all_Y.astype(int) - 1
    all_Y = all_Y.reshape(all_Y.shape[0])
    num_classes = max(Y_data) + 1
    y_= keras.utils.to_categorical(all_Y, num_classes)
    x_= all_data

    test_prop = 0.1
    test_size= round(test_prop * number_of_samples, 0)
    test_size = int(test_size)
    train_size = number_of_samples - test_size

    y_train = y_[0:train_size]
    x_train = x_[0:train_size]

    y_test= y_[train_size:]
    x_test= x_[train_size:]

    batch_size = 20
    epochs = 100

    model = create_network_tu(size_of_sample, num_classes)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    tensorboard=TensorBoard(log_dir='./logs', write_graph=True)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=[tensorboard])

    scores = model.evaluate(x_test, y_test, batch_size=batch_size)
    for i in range(len(model.metrics_names)):
        print("{}: {}".format(model.metrics_names[i], scores[i]))



if __name__ == '__main__':
    main()
