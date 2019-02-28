function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);
%size(X) %307:2

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

tmp_mu = mu;
tmp_sigma2 = sigma2;

%calculate the avg value
for j = 1:n,
	for i = 1:m,
		tmp_mu(j) = tmp_mu(j) + X(i,j);
	end
end

tmp_mu = tmp_mu/m;

%calculate sigma value
for j = 1:n,
	for i = 1:m,
		tmp_sigma2(j) = tmp_sigma2(j) + (X(i,j) - tmp_mu(j))^2;
	end
end

tmp_sigma2 = tmp_sigma2/m;

mu = tmp_mu;
sigma2 = tmp_sigma2;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%










% =============================================================


end
