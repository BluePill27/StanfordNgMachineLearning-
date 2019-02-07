function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%size(X)%12:2
%size(y)%12:1
%theta% 1 1

%------------Linear Cost-----------
z = X*theta;
%size(z)%12:1 
cost = J;
mistake = (sum((z-y).^2)/(2*m));
penalty = (lambda/(2*m))*(sum((theta(2:end).^2)'));
cost = mistake + penalty;
J = cost;

%------------Linear Gradient-----------
gradient = grad';
for j = 1:length(gradient),
	for i = 1:m,
		gradient(j) = gradient(j) + (z(i) - y(i)) * X(i,j);
	end
end

gradient = gradient/m;

for j = 2:length(gradient),
	gradient(j) = gradient(j) + (lambda/m) * theta(j);
end
grad = gradient';









% =========================================================================

grad = grad(:);

end
