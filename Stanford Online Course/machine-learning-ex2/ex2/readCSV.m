function [X, y] = readCSV(fileName)

%data = csvread('gender_submission.csv', 1, 0)
clc
data = csvread(fileName, 1, 0);
y = data(:, 2);
X = data(:, [3, 4]);

end