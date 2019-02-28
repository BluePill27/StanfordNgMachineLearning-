function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

%save amount of movies and users into variables
movie_count = size(X, 1);
user_count = size(Theta, 1);


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%---------------TESTING---------------
%X(2,:)%0.78 -0.38 0.52
%R(2,:)%1 0 0 0 
%Theta(1, :)%0.28 -1.68 0.26
%Y(2,:)% 3 0 0 0
%X(2,:) * Theta(1,:)'%1


%---------------UNREG COST VALUE-----------------
%Predictions = calculated predictions
%J = first compute (squared differences of (predictions - real values)) multiply it with R
%so we only sum values that are on the same positions as 1 in the R matrix
%then sum the values so it becomes a 1:n dimencional vector
%then sum again over the result matrix, so we get a cost value

Predictions = X*Theta';
J = sum( (sum( ((Predictions - Y).^2).*R) ) )/2;

%---------------UNREG GRADIENT---------------
%Calculate differences between ((Predictions and Results) .* marked matrix [R[0 1; 1 0]]) * Theta
%=>(([m:u]-[m:u]) .*[m:u]) * [u:n] = [m:n]
X_grad = ((Predictions - Y).*R)*Theta;

%Calculate differences between ((Predictions and Results) .* marked matrix [R[0 1; 1 0]])T * X
%=>(([m:u]-[m:u]) .*[m:u])' * [m*n] = [u:n]
Theta_grad = ((Predictions - Y).*R)'*X;

%---------------REG COST VALUE-----------------
%Calculate the penaltyes for each of the multiplication values (X, Theta) then sum with the cost value
PenaltyX = (lambda/2)*(sum(sum(X.^2)));
PenaltyTheta = (lambda/2)*(sum(sum(Theta.^2)));
J = J + PenaltyX + PenaltyTheta;
		
%---------------REG GRADIENT---------------
%Calculate the penaltyes for each of the values (X, Theta) and sum with the gradient value
X_grad = X_grad + (lambda * X);
Theta_grad = Theta_grad + (lambda * Theta);













% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
