% perform gradient descent on our logistic regression example

% X (m by n) feautures and examples (the first column is x0 = 1)
X = [1, 1, 1; 1, 1, 2; 1, 2, 1; 1, 2, 3; 1, 3, 2; 1, 3, 3];

% theta (n by 1) contains classifyer parameters theta(1) through theta(n)
% this is what we are trying to solve
theta = zeros(3, 1);

% y (m by 1) contains the class
y = [1;1;1;0;0;0];
m = length(y);

% h (m by 1) = g(X*theta)
h = X*theta;
S = transpose(SigmoidFunction(h));

% cost Function
%jVal = (1/m) * (-transpose(y)*log(h) - transpose(1 - y)*log(h))

%gradient descent algorithm
iterations = 5000;
alpha = 0.01;
J_hist = zeros(iterations, 1);

% wait, are the matricies correct?
% plan: go back and try to copy what you did in the preivious Grad Desc
% program

%plot the cost function vs iteration to see if it is working

    %theta_hold = theta;
    %     theta(1) = theta(1) - (alpha / m) * sum(((X * theta_hold) - y) .* X(:, 1));
    %     theta(2) = theta(2) - (alpha / m) * sum(((X * theta_hold) - y) .* X(:, 2));

for i = 1:iterations
    %theta = theta - (alpha/m)*transpose(X)*(S - y);
    theta_hold = theta;
    theta(1) = theta(1) - (alpha / m) * sum((SigmoidFunction(X * theta_hold) - y) .* X(:, 1));
    theta(2) = theta(2) - (alpha / m) * sum((SigmoidFunction(X * theta_hold) - y) .* X(:, 2));
    theta(3) = theta(3) - (alpha / m) * sum((SigmoidFunction(X * theta_hold) - y) .* X(:, 3));
    
    %J_hist(i) = computeCost(X, y, theta);
end

theta;
%descisionBoundary = theta(1) + theta(2)* + theta(3);
DB = zeros(4, 1);

for x = -5:5
    DB(x+6) = (-theta(2) * x - theta(1)) / theta(3);
end

figure;
plot(X(1:3, 2), X(1:3, 3), 'rx', 'MarkerSize', 30);
hold on;
plot(X(4:6, 2), X(4:6, 3), 'bx', 'MarkerSize', 30);
plot(DB,linspace(-5, 5, 11));

% success!!!!!!!!

x1 = input('Enter x1: ');
x2 = input('Enter x2: ');

if (theta(1) + theta(2) * x1 + theta(3) * x2) < 0
    fprintf('predict 0\n');
    plot(x1, x2, 'bx', 'MarkerSize', 30);
else
    fprintf('predict 1\n');
    plot(x1, x2, 'rx', 'MarkerSize', 30);
end
