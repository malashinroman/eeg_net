function [ X, Y, event_delay, button_delay] = take_training_set(data, mnt, mrk)

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
data_size = max_period;
n = sum(take);
X = zeros(n, data_size, size(data.X, 2));
nclasses = 6; % number of attention direction
Y = zeros(n, nclasses);

for i=1:n 
   experiment_start = trials(i);
   experiment_end = trials(i) + time_before_hit(i);
   record_size = experiment_end - experiment_start;
   experiment_record = dataX(experiment_start:experiment_end, :);
   push_data_size = record_size;
   if(data_size < push_data_size)
       push_data_size = data_size;
   end 
   X(i, 1:push_data_size, :) = experiment_record(1:push_data_size, :);
   Y(i, y(i)) = 1;
end


end

