# L1, L2

#Total number of features (handled by L1 regularization)
#The weights of features (handled by L2 regularization)

# L1
#used to handle sparse vectors which consist of mostly zeroes
#L1 regularization forces the weights of uninformative features to be zero by 
#substracting a small amount from the weight at each iteration and thus making 
#the weight zero, eventually.

# L2
# regularization for simplicity
# y = w_1 x_1 + w_2 x_2 + ... w_n x_n
# L2 regularization terms = w_1^2 + w_2^2 + ... + w_n^2
# L2 regularization forces weights toward zero but it does not make them exactly zero. 
# L2 regularization acts like a force that removes a small percentage of weights at 
# each iteration. Therefore, weights will never be equal to zero.
# For L2 additional tunable parameter = lambda (regularisation rate):
# minimise(Loss(model) + lambda*L2 regularization terms)
# or minimise(Loss(model) + lambda*Complexity(model))
# => high lambda -> simpler model