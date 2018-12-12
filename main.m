%% Setup the parameters you will use for this exercise
input_layer_size  = 2500;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 4;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

X = dlmread('x_features');
y = dlmread('y_labels');

label_keys = { 'left', 'right', 'palm', 'peace'};

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


pred = predict(Theta1, Theta2, X);

[val idx] = max(y, [], 2);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == idx)) * 100);


test_img = processSkinImage("test/test1.jpg");
pred = predict(Theta1, Theta2, test_img(:)')
fprintf('\nType: %s\n', label_keys{pred});
pause;

test_img = processSkinImage("test/test2.jpg");
pred = predict(Theta1, Theta2, test_img(:)')
fprintf('\nType: %s\n', label_keys{pred});
pause;

test_img = processSkinImage("test/test3.jpg");
pred = predict(Theta1, Theta2, test_img(:)')
fprintf('\nType: %s\n', label_keys{pred});
pause;

test_img = processSkinImage("test/test4.jpg");
pred = predict(Theta1, Theta2, test_img(:)')
fprintf('\nType: %s\n', label_keys{pred});
