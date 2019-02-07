function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


cost = 0;
weight = 0;

for i = 1:m,
	cost = cost + (-y(i) * log(1/(1+exp(-theta' * X(i,:)'))) - (1-y(i)) * log(1 - (1/(1+exp(-theta' * X(i,:)')))));
end;

for j = 2:length(grad),
	weight = weight + (theta(j))^2;
end;

cost = cost/m;
weight = (weight * lambda)/(2*m);
J = cost + weight;


grad1 = grad
;

%this anc next loop can be combined, you dummy :/
for i = 1:m,
	grad1(1) = grad1(1) + ((1/(1+exp(-theta' * X(i,:)')) - y(i)) * X(i,1));
end

for j = 2:length(grad),
	for i = 1:m,
		grad1(j) = grad1(j) + ((1/(1+exp(-theta' * X(i,:)')) - y(i)) * X(i,j));
end
end

grad1 = grad1/m;

for j = 2:length(grad),
	grad1(j) = grad1(j) + ((lambda*theta(j))/m);
end

grad = grad1;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
