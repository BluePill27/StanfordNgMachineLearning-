function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);

coord = zeros(n, 1);

%iterate trought centroids id's and if example labled centroid = current iterate centroid, example coord
%are getting added to the coord matrix, then sum of coord in columns gets devided by amount of examples
%belongs to the centroid
for k = 1:K,
	for i = find(idx == k),
		coord = sum(X(i,:));
	end
	coord = coord/length(find(idx == k));
	centroids(k,:) = coord;
	coord = zeros(n, 1);
end
	
		
			
			
	
% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%








% =============================================================


end

