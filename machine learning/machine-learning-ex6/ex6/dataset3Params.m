function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
try_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
m = length(try_vec);
M = [];
x1 = [1 2 1]; x2 = [0 4 -1];
for i=1:m
    for j=1:m
        M = [M; try_vec(i) try_vec(j)];
    end
end
errors = [M zeros(m*m, 1)];
for i=1:m*m
    model = svmTrain(X, y, M(i,1), @(x1, x2) gaussianKernel(x1, x2, M(i,2))); 
    predictions = svmPredict(model, Xval);
    errors(i,3) = mean(double(predictions ~= yval));
end
[~,I] = min(errors(:,3));
C = M(I, 1);
sigma = M(I, 2);
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%




% =========================================================================

end
