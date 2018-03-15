function [ X, Y, event_delay, button_delay, labels] = take_training_set_all()

filenames = {'covertShiftsOfAttention_VPgao.mat', 
            'covertShiftsOfAttention_VPiaa.mat',
            'covertShiftsOfAttention_VPiac.mat',
            'covertShiftsOfAttention_VPiae.mat',
            'covertShiftsOfAttention_VPiah.mat',
            'covertShiftsOfAttention_VPiai.mat',
            'covertShiftsOfAttention_VPmk.mat',
            'covertShiftsOfAttention_VPnh.mat'};

experiment_start_delay = 70;
expreiment_end_delay = 450;
nclasses = 6; % number of attention direction
data_size = expreiment_end_delay - experiment_start_delay;
X = zeros(0, data_size, 62);
Y = zeros(0, nclasses);
labels = [];
for i=1:size(filenames,1)
    file = filenames{i};
    load(file);
    hits = (mrk.event.ishit == 1);
    valid = (mrk.event.valid_cue == 1);
    take = valid & hits;
    trials = data.trial(take);
    latencies = mrk.event.latency(take);
    target_latencies = mrk.event.target_latency(take);
    y = data.y(take);
    dataX = data.X;
    coef_to_ticks = data.fs/1000;
    latencies_ticks = int64(latencies.*(coef_to_ticks)); 
    target_latencies_ticks = int64(target_latencies.*(coef_to_ticks));

    time_before_hit = (latencies_ticks + target_latencies_ticks);
    event_delay = latencies_ticks;
    button_delay = time_before_hit;
    min_period = min(time_before_hit);
    max_period = max(time_before_hit);
    n = sum(take);
    X_ = zeros(n, data_size, size(data.X, 2));
    Y_ = zeros(n, nclasses);
    
    for j=1:n
       experiment_start = trials(j);
       experiment_end = trials(j) + data_size;
       record_size = experiment_end - experiment_start;
       experiment_record = dataX(experiment_start:experiment_end-1, :);
       push_data_size = record_size;
       X_(j, 1:push_data_size, :) = experiment_record(1:push_data_size, :);
       Y_(j, y(j)) = 1;
    end
    X = [X; X_];
    Y = [Y; Y_];
    labels = [labels; y(:)];
end












end

