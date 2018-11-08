function [h] = SigmoidFunction(Z)

h = 1 ./ (1 + exp(-Z));

end