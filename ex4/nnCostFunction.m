function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%-----------------SETUP SETUP SETUP -------------------
% Add ones to the X data matrix
X = [ones(m, 1) X];

%make a tmp_y matrix 5000:10 where 1:10 corresponds to the correct output 
tmp_y = zeros(m, num_labels);
for i = 1:m,
	tmp_y(i, y(i)) = 1;
end


%size(X) %5000:401
%size(y) %5000:1
%size(tmp_y) %5000:10
%y(800) % 10
%tmp_y(800,:) % [0,0,0,0,0,0,0,0,0,1]

%----------------------- Part1 ---------------------
tmp_cost = J;
for i = 1:m,
	tmp_2 = sigmoid(Theta1 * (X(i,:))');%25:401 * 401:1 = 25:1
	tmp_2 = [1;tmp_2];% 26:1
	tmp_3 = sigmoid(Theta2 * tmp_2);%10:26 * 26:1 = 10:1
	for k = 1:num_labels,
		tmp_cost = tmp_cost + (-tmp_y(i,k)*log(tmp_3(k)) - (1 - tmp_y(i,k))*log(1 - tmp_3(k)));
	end
end
J = tmp_cost/m;

%----------------------- Part2 ---------------------
tmp_weight = 0;
tmp_weight1 = 0;
tmp_weight2 = 0;

%Possible to be done by || (lambda/ (2*m)) * (sum((sum(Theta1^2))') + sum((sum(Theta2^2))')) ||?
for i = 1:size(Theta1,1),
	for j = 2:size(Theta1,2),
		tmp_weight1 = tmp_weight1 + Theta1(i,j)^2;
	end
end

for i = 1:size(Theta2,1),
	for j = 2:size(Theta2,2),
		tmp_weight2 = tmp_weight2 + Theta2(i,j)^2;
	end
end

tmp_weight = tmp_weight1 + tmp_weight2;
tmp_weight = (lambda/ (2*m)) * tmp_weight;

J = J + tmp_weight;

%----------------------- Part3 ---------------------

%tmp_delta3 = zeros(size(Theta2), 1)	;
%tmp_delta2 = zeros(size(Theta1), 1);

% a3 = tmp_3, a2 = tmp_2
%for i = 1:m,
%	tmp_2 = sigmoid(Theta1 * (X(i,:))');%25:401 * 401:1 = 25:1
%	tmp_2 = [1;tmp_2];% 26:1
%	tmp_3 = sigmoid(Theta2 * tmp_2);%10:26 * 26:1 = 10:1
%	
%	z2 = Theta1 * (X(i,:))';
%	z2 = [1; z2];
%	tmp_delta3 = tmp_3 - tmp_y(i)';% 10:1
%	tmp_delta2 = ((Theta2)' * tmp_delta3).*(sigmoidGradient(z2));% (26:10 * 10:1). * 26:1
%	
%	
%	Theta2_grad = Theta2_grad + (tmp_delta3 * (tmp_2)');% 10:26 + (10:1 * 1:26)
%	Theta1_grad = Theta1_grad + (tmp_delta2(2:end) * (X(i,:)));% 25:401 + (25:1 * 1:401)
%	end

%Theta1_grad = Theta1_grad./ m;
%Theta2_grad = Theta2_grad./ m;

%Theta1_grad(2:end) = Theta1_grad(2:end) + ((lambda/m) * Theta1(2:end));
%Theta2_grad(2:end) = Theta2_grad(2:end) + ((lambda/m) * Theta2(2:end));


for i = 1:m,
	a1 = X(i,:)'; %401:1
	z2 = Theta1 * a1;
	a2 = sigmoid(z2);
	a2 = [1;a2]; %26:1
	
	z3 = Theta2 * a2;
	a3 = sigmoid(z3); % 10:1
	
	d3 = a3 .- (tmp_y(i,:))';
	z2 = [1;z2];
	d2 = ((Theta2)' * d3).* (sigmoidGradient(z2));
	d2 = d2(2:end);

	Theta2_grad = Theta2_grad + (d3 * (a2)');
	Theta1_grad = Theta1_grad + (d2 * (a1)');
end

%size(Theta1_grad) %25:401
%size(Theta2_grad) %10:26
%Theta1_grad(1:10) %260 150 -30 ...
%Theta2_grad(1:10) %3. 2. 3. ...

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda/(m)) * Theta1(:,2:end));
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda/(m)) * Theta2(:,2:end));









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
