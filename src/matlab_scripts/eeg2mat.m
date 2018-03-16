%  онвертаци€ текстовых файлов с ЁЁ√ данными и метками в единый файл .mat
clear all; close all; clc % очистка пам€ти, очистка консоли
EXP_NUM = 21;
EXP_SHIFT = 167;
% загрузка меток, которые нужно не учитывать
exlabels = csvread('exlabels.csv');
[h, w] = size(exlabels);
[flab{1:EXP_NUM}] = deal([]);
for y = 1:h
    enum = exlabels(y, 1);
    flab{enum}(end + 1) = exlabels(y, 2);
end
% eeg_num = 1;
 for eeg_num = 1:21 % цикл по всем номерам файлов
    infile_labels = sprintf('labels\\D%07d.TXT', eeg_num + EXP_SHIFT);
    infile_eeg = sprintf('eeg\\D%07d.TXT', eeg_num + EXP_SHIFT); 
    
    % загружаем файл с метками
    data = dlmread(infile_labels,'', 1, 0);
    % фильтруем файл с метками
    if ~isempty(flab{eeg_num})
        data = removerows(data, 'ind', flab{eeg_num});
    end
    
    % удал€ем низкочастотные метки
    [h, w] = size(data);
    indices = [];
    for y = 1:h
        if data(y, 2) <= 50
            indices(end + 1) = y;
        end
    end
    data = removerows(data, 'ind', indices);
    
    
    
    time = data(:,1);
    label = data(:,2);
    
    % преобразуем номера меток: 49 и 51 = 1 - живые, 50 и 52 = 2 - неживые
    label(label==49) = 1; label(label==51) = 1;
    label(label==50) = 2; label(label==52) = 2;
    label = int64(label);

    fs = 250; % частота дискретизации 250 √ц
    time = time.*fs; % переводим временные метки в отсчеты сигнала
    time = int64(time); % переходим к целочисленному формату

    s = struct; % создаем структуру дл€ записи в .mat файл
    s.fs = fs;

    data = dlmread(infile_eeg,'', 0, 0);
    [~, C] = size(data);
    [T, ~] = size(label);
    shift = int64(1.1 * 250); % длина одного испытани€ = 1100 мс * 250 √ц
    shift1 = int64(0.15 * 250);
    shift2 = int64(0.42 * 250);
    eeg = zeros(shift, C, T);
    for trial = 1:T % цикл по всем испытани€м
        for channel = 1:C % цикл по всем каналам
            ch_data = data(:, channel);
            index = 1;
            for y = time(trial, 1)+shift1:time(trial, 1)+shift2-1 % цикл по всем отсчетам одного испытани€
                eeg(index, channel, trial) = ch_data(y, 1);
                index = index + 1;
            end
        end
    end
    s.eeg = eeg; % сохран€ем в структуру ээг данные
    s.mrk = label; % сохран€ем в структуру преобразованные метки
    output_name = sprintf('eegmat_selected\\D%07d.mat', eeg_num + EXP_SHIFT);
    save(output_name, 's'); % сохран€ем .mat файл
 end