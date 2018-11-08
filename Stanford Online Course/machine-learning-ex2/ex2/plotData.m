function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% so we need to separate the Data based on y values, then plot them
% loop, if y = 1, plotData +, else plotData o

pos = [];
neg = [];

% for i = 1:length(y)
%     %separate columns of X???
%     if y(i) == 1
%         pos = [pos; X(i, 1:2)];
%     else
%         neg = [neg; X(i, 1:2)];
%     end
% end
% 
% plot(pos(:, 1), pos(:, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
% hold on;
% plot(neg(:, 1), neg(:, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

pos = find(y==1);
neg = find(y==0);

plot(X(pos, 1), X(pos, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);



% =========================================================================



hold off;

end
