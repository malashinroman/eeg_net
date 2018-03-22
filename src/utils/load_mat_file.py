import scipy.io
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

def load_mat_file(filename, variablename):
    M = scipy.io.loadmat(filename)
    return M[variablename];

def show_sample(sample, channels, label = -1):
    x = range(0, sample.shape[0]);
    f, ax = plt.subplots(figsize=(8, 8))
    for i in channels:
        plt.plot(x, sample[:,i])
    ax.legend(['ch ' + repr(i) for i in channels])
    ax.set_xlabel('time, ticks')
    ax.set_ylabel('voltage')
    ax.grid(which='both')
    return ax
    
def show_data(X_data, Y_data, indx, channels):
    sample = X_data[indx];
    label = Y_data[indx];
    ax = show_sample(sample, channels, label)
    ax.set_title('stimulus numer: ' + repr(indx) + ', label: ' + repr(Y_data[indx]));
    #f, ax = plt.subplots(figsize=(8, 8))
    #x = range(0, sample.shape[0]);
    #for i in channels:
    #    plt.plot(x, sample[:,i])
        
    #ax.legend(['ch ' + repr(i) for i in channels])
    #ax.set_title('stimulus numer: ' + repr(indx) + ', label: ' + repr(Y_data[indx]));
    #ax.set_xlabel('time, ticks')
    #ax.set_ylabel('voltage')
    #ax.grid(which='both') 
        #x = range(0, sample.shape[0])
        #plt.xlabel('milliseconds');
        #plt.ylabel('volatage');
        
        #ax.plot(acc_levels, acc_perc, color='steelblue', label='perceptron')
        #ax.plot(acc_levels, acc_lstm, color='orangered', label='lstm, exp avg')
        #ax.plot(acc_levels, acc_lstm_2, color='orangered', label='lstm, 0 pred', linestyle='--')
        # ax.plot(acc_levels, acc_merged, color='seagreen')


        #ax.set_xlabel("Distance between predicted and actual dot")
        #ax.set_ylabel("Amount of predictions within the distance")
        # plt.vlines([0.1, 0.2, 0.3], 0, 1, label="within 0.1 of screen size", colors='gray', linestyles='--')
        # plt.hlines([0.25, 0.5, 0.75], 0, 1, label="within 0.1 of screen size", colors='gray', linestyles='--')

        #ax.grid(which='both')                                                            

        # or if you want differnet settings for the grids:                               
        #ax.grid(which='minor', alpha=0.1)                                                
        #ax.grid(which='major', alpha=0.5)   
        #ax.minorticks_on()
        #plt.legend()

def load_data():
    EXP_NUM = 21 # total number of people
    EXP_SHIFT = 167 # shift in numerations
    CHANNELS_NUM = 21 # number of channels in EEG record
    sel_channels = slice(0, 21)
    all_data = np.array([])
    all_Y = np.array([])
    for eeg_num in range(1, EXP_NUM + 1):
        EEG_FILENAME = 'eegmat_selected/D0000' + str(eeg_num + EXP_SHIFT)
        EEG = load_mat_file(EEG_FILENAME, 's')
        X_data = EEG["eeg"][0][0]
        X_data = np.swapaxes(X_data, 2, 0)
        X_data = np.swapaxes(X_data, 1, 2)
        Y_data = EEG["mrk"][0][0]
        OneChannel_data = X_data[:, sel_channels, :]
        OneChannel_data = OneChannel_data.reshape(OneChannel_data.shape[0], OneChannel_data.shape[1]*OneChannel_data.shape[2])
        if eeg_num == 1:
            all_data = np.array([]).reshape(0, OneChannel_data.shape[1])
            all_Y = np.array([]).reshape(0, Y_data.shape[1])
        all_data = np.concatenate([all_data, OneChannel_data])
        all_Y = np.concatenate([all_Y, Y_data])
        
def load_all_data(data_path):
    EXP_NUM = 21 # total number of people
    EXP_SHIFT = 167 # shift in numerations
    CHANNELS_NUM = 21 # number of channels in EEG record
    sel_channels = slice(0, )
    all_data = np.array([])
    all_Y = np.array([])
    for eeg_num in range(1, EXP_NUM + 1):
        EEG_FILENAME = 'eegmat_selected/D0000' + str(eeg_num + EXP_SHIFT)
        EEG = load_mat_file(os.path.join(data_path,EEG_FILENAME), 's')
        X_data = EEG["eeg"][0][0]
        
        X_data = np.swapaxes(X_data, 2, 0)
        X_data = np.swapaxes(X_data, 1, 2)
        Y_data = EEG["mrk"][0][0]
        OneChannel_data = X_data[:, sel_channels, :]
        OneChannel_data = OneChannel_data.reshape(OneChannel_data.shape[0], OneChannel_data.shape[1]*OneChannel_data.shape[2])
        if eeg_num == 1:
            all_data = np.array([]).reshape(0, OneChannel_data.shape[1])
            all_Y = np.array([]).reshape(0, Y_data.shape[1])
        all_data = np.concatenate([all_data, OneChannel_data])
        all_Y = np.concatenate([all_Y, Y_data])
    return all_data, all_Y;


def load_all_data_multichannel(data_path):
    EXP_NUM = 21 # total number of people
    EXP_SHIFT = 167 # shift in numerations
    CHANNELS_NUM = 21 # number of channels in EEG record
    sel_channels = slice(14, 18)
    sel_time = slice(None,None,1) #quanta time
    all_data = np.array([])
    all_Y = np.array([])
    for eeg_num in range(1, EXP_NUM + 1):
        EEG_FILENAME = 'eegmat_selected/D0000' + str(eeg_num + EXP_SHIFT)
        EEG = load_mat_file(os.path.join(data_path,EEG_FILENAME), 's')
        X_data = EEG["eeg"][0][0]
        #print(X_data[:,:,1])
        #print(X_data.shape)
        X_data = np.swapaxes(X_data, 2, 0)
        X_data = np.swapaxes(X_data, 1, 2)
        #print('min ', np.min(X_data), '; max ', np.max(X_data))
        #X_data = normalize_data(X_data)
        #print('min ', np.min(X_data), '; max ', np.max(X_data))
        Y_data = EEG["mrk"][0][0]
        #show_data(X_data, Y_data, 1, [0,2])
        OneChannel_data = X_data[:, sel_time, sel_channels]
        #OneChannel_data = OneChannel_data.reshape(OneChannel_data.shape[0], OneChannel_data.shape[1]*OneChannel_data.shape[2])
        if eeg_num == 1:
            all_data = np.array([]).reshape(0, OneChannel_data.shape[1], OneChannel_data.shape[2])
            all_Y = np.array([]).reshape(0, Y_data.shape[1])
        all_data = np.concatenate([all_data, OneChannel_data])
        all_Y = np.concatenate([all_Y, Y_data])
    return all_data, all_Y;