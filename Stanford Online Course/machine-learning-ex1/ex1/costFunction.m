function [jVal, gradient] = costFunction(theta)

jVal = (theta(1) - 5)^2 + (theta(2) - 5)^2;

 gradient = zeros(2, 1);
gradient(1) = 2 * (theta(1) - 5);
gradient(2) = 2 * (theta(2) - 5);

% jVal = (1 / (1 + exp(-theta(1))) - 1)^2 + (1 / (1 + exp(-theta(2))) - 1)^2;
% 
% gradient(1) = 2 * (1 / (1 + exp(-theta(1))) - 1) * (exp(-theta(1)) / (1 + exp(-theta(1)))^2);
% gradient(2) = 2 * (1 / (1 + exp(-theta(2))) - 1) * (exp(-theta(2)) / (1 + exp(-theta(2)))^2);

end
