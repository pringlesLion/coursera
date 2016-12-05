function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
z = zeros(size(m));
for i = 1:m
    h = sigmoid(theta'*X(i, :)');
    z(i) = -y(i)*log(h) - (1 - y(i))*log(1-h);
end;
J = 1/m * sum(z) + lambda/(2*m) * sum(theta(2:end, :).^2) ;
grad0 = (1/m*X'*(sigmoid(X*theta) - y));
grad1 = 1/m*X'*(sigmoid(X*theta) - y) + lambda/m * theta;
grad = [grad0(1,:);grad1(2:end,:)];
% =============================================================

end
