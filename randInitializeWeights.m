function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections

% Initialize W randomly so that we break the symmetry while
%               training the neural network.

% Note that W should be set to a matrix of size(L_out, 1 + L_in) as
% the first column of W handles the "bias" terms

epsilon = sqrt(6) / (L_in + L_out);

W = zeros(L_out, 1 + L_in);
W = (rand(L_out, 1 + L_in) * 2 * epsilon) - epsilon;

end
