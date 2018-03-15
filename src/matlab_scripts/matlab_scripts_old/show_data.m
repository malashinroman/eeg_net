function [] = show_data(X, Y, event_delays, button_delays, indx)

fs = 200;
ms_per_tick = 1000 / fs;
% plot(reshape(X(i,:,:), size(X,2), size(X,3)));
show_data = X(indx,:,:);
show_data = reshape(show_data, size(X,2), size(X,3));
milliseconds = 1:size(show_data,1);
milliseconds = milliseconds.* ms_per_tick;
plot(milliseconds, show_data);
xlabel('milliseconds');
ylabel('volatage');
hold on
plot(event_delays(1)*ms_per_tick, 1.2 * max(max(show_data)), 'ro', button_delays(1)*ms_per_tick, 1.2 * max(max(show_data)), 'bo');

end

