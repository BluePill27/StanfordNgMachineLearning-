function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
%size(X) 5000:400
% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

X = [ones(m, 1) X];
tmp_2 = zeros(size(Theta1,1),1);% 25:1
tmp_3 = zeros(size(Theta2,1),1);% 10:1
tmp_p = zeros(size(X,1), num_labels);% 5000:10
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

for i = 1:m,
	tmp_2 = sigmoid(Theta1 * (X(i,:))');%25:401 * 401:1 = 25:1
	tmp_2 = [1;tmp_2];% 26:1
	tmp_3 = sigmoid(Theta2 * tmp_2);%10:26 * 26:1 = 10:1
	[x, ix] = max(tmp_3);
	p(i) = ix;







% =========================================================================


end
