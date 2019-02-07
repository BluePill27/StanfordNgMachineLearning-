function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
g;
q = g;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

for i = 1:length(q(:,1)),
	for j = 1:length(q(1,:)),
		q(i,j) = 1/(1+exp(-z(i,j)));
end
end
%q. = 1/(1+exp(-z.));
g = q;

% =============================================================

end
