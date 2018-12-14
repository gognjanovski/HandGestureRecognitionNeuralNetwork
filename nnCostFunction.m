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


% Feedforward the neural network and return the cost in the variable J
summary = 0;
for j = 1:m

    y_label = y(j,:);

    a_one = X(j,:);
    a_one = [1 a_one];
    z_one = a_one * Theta1';
    a_two = sigmoid(z_one);
    m_temp = size(a_two, 1);
    a_two = [ones(m_temp, 1) a_two];
    z_two = a_two * Theta2';
    a_three = sigmoid(z_two);
    h = a_three;


    sum_temp = 0;
    sum_temp = (-y_label .* log(h)) - ((1 - y_label) .* log(1 - h));
    summary = summary + sum(sum_temp); 

    
% Implement Backpropagation algorithm
 
    d3 = (a_three .- y_label);
    z_one = [1 z_one];
    d2 = (d3 * Theta2) .* sigmoidGradient(z_one);
    d2 = d2(:, 2:end);

    Theta1_grad = Theta1_grad + d2' * a_one;
    Theta2_grad = Theta2_grad + d3' * a_two;

end;

% Add regularization to the cost
regularization = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end) .^ 2, 2)) + sum(sum(Theta2(:,2:end) .^ 2, 2)));

J = (summary/m) + regularization;

% Add regularization to the gradient
Theta1_grad(:,1) = Theta1_grad(:,1) ./ m;
Theta2_grad(:,1) = Theta2_grad(:,1) ./ m;
Theta1_grad(:,2:end) = (Theta1_grad(:,2:end) ./ m) + ((lambda ./ m) * Theta1(:,2:end));
Theta2_grad(:,2:end) = (Theta2_grad(:,2:end) ./ m) + ((lambda ./ m) * Theta2(:,2:end));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
