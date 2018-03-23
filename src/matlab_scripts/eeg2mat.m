% script for converting .txt EEG records to .mat files with specific
% structure
clear all; close all; clc % clear memory & screen
EXP_NUM = 21;
EXP_SHIFT = 167;
% exclude manually selected labels
exlabels = csvread('exlabels.csv');
[h, w] = size(exlabels);
[flab{1:EXP_NUM}] = deal([]);
for y = 1:h
    enum = exlabels(y, 1);
    flab{enum}(end + 1) = exlabels(y, 2);
end
% eeg_num = 1;
 for eeg_num = 1:21 % loop over all eeg records
    infile_labels = sprintf('labels/D%07d.TXT', eeg_num + EXP_SHIFT);
    infile_eeg = sprintf('eeg/D%07d.TXT', eeg_num + EXP_SHIFT); 
    
    % read .txt file with class labels
    data = dlmread(infile_labels,'', 1, 0);
    % remove rows with incorrect measurements
    if ~isempty(flab{eeg_num})
        data = removerows(data, 'ind', flab{eeg_num});
    end
    
    time = data(:,1);
    label = data(:,2);
    
    % convert 4 classes to 2: 49 & 51 = 1 - animate, 50 & 52 = 2 -
    % inanimate
    label(label==49) = 1; label(label==51) = 1;
    label(label==50) = 2; label(label==52) = 2;
    label = int64(label);

    fs = 250; % discretization frequency 250 Hz
    time = time.*fs; % convert time to number of ticks
    time = int64(time); 

    s = struct; % create memory structure for MAMEM toolbox
    s.fs = fs;

    data = dlmread(infile_eeg,'', 0, 0);
    [~, C] = size(data);
    [T, ~] = size(label);
    shift = int64(1.1 * 250); % using timespan = 1100 ms * 250 Hz
    eeg = zeros(shift, C, T);
    for trial = 1:T % for all trials
        for channel = 1:C % for all channels
            ch_data = data(:, channel);
            index = 1;
            for y = time(trial, 1):time(trial, 1)+shift % using selected timespan
                eeg(index, channel, trial) = ch_data(y, 1);
                index = index + 1;
            end
        end
    end
    s.eeg = eeg; 
    s.mrk = label; 
    output_name = sprintf('eegmat_selected/D%07d.mat', eeg_num + EXP_SHIFT);
    save(output_name, 's');
 end