clear all; close all; clc
EXP_NUM = 168;
EXP_FOLDER = "eegmat_selected";
eegpath = sprintf('%s/D%07d.mat', EXP_FOLDER, EXP_NUM);
eeg_struct = load(eegpath);
EEG = eeg_struct.s.eeg;

channel = 3;
chlist = [14 18];
% chlist = [1:21];
eeg_selected = squeeze(EEG(:, channel, :));
[ticks, events] = size(eeg_selected);
f = figure('Name','EEG Curves');
for event = 1:events
    for channel = chlist
       eeg_selected = squeeze(EEG(:, channel, :));
       plot(eeg_selected(:, event));
       hold on;
    end
       xlim([0 ticks]);
       hold off;
        w = waitforbuttonpress;
        switch w 
            case 1 % (keyboard press) 
                key = get(gcf,'currentcharacter'); 
                  switch key
                      case 27 % 27 is the escape key
                          disp('User pressed the escape key. I''m quitting.')
                          close all;
                          break % break out of the while loop
                      otherwise 
                          % Wait for a different command. 
                  end
        end
end