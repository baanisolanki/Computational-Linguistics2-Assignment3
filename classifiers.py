import numpy as np
from scipy.sparse import csr_matrix

class CustomNaiveBayes:
    def __init__(self): #definign a class
        self.logprior = {}
        self.loglikelihood = {}
        self.classes = []
        self.vocab_size = 0

    def fit(self, X_train, y_train): #naive bayes classifier; X_train is sparse matrix from CountVectorizer adn y_train is list of class labels
       
        self.classes = np.unique(y_train)
        N_doc = X_train.shape[0]
        self.vocab_size = X_train.shape[1]
        
        # storing loglikelihood as a 2D numpy array: [class_idx, word_idx]
        self.loglikelihood = np.zeros((len(self.classes), self.vocab_size))
        
        for c_idx, c in enumerate(self.classes):
            #filter documents belonging to class c
            X_c = X_train[np.array(y_train) == c]
            N_c = X_c.shape[0]
            
            # logprior[c] <- log(Nc / Ndoc)
            self.logprior[c] = np.log(N_c / N_doc)
            
            #count of occurrences of each word in bigdoc[c]
            # X_c.sum(axis=0) returns a matrix of shape (1, vocab_size)
            word_counts = np.array(X_c.sum(axis=0)).flatten()
            
            # total words in class c
            total_words_c = word_counts.sum()
            
            # calculating loglikelihood[w, c] using laplace smoothing
            # log( (count(w,c) + 1) / (sum_w'(count(w',c)) + |V|) )
            self.loglikelihood[c_idx, :] = np.log((word_counts + 1) / (total_words_c + self.vocab_size))

    def predict(self, X_test): #running naive bayes classifer on test set
        predictions = []
        
        for i in range(X_test.shape[0]):
            doc_vector = X_test[i]
            
            #we optimize the sum loop by using dot products on the sparse vector
            #sum[c] = logprior[c] + sum(loglikelihood[w,c] * word_presence)
            best_c = None
            max_sum = -np.inf
            
            for c_idx, c in enumerate(self.classes):
                current_sum = self.logprior[c]
                
                # doc_vector is 1xV sparse matrix multiply by loglikelihoods to sum present words
                #if non-binary bag-of-words, this correctly multiplies loglikelihood by frequency
                likelihood_sum = doc_vector.dot(self.loglikelihood[c_idx, :].T)[0]
                current_sum += likelihood_sum
                
                if current_sum > max_sum:
                    max_sum = current_sum
                    best_c = c
                    
            predictions.append(best_c)
            
        return np.array(predictions)


class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def sigmoid(self, z):
        # we clip z to prevent overflow errors in np.exp
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X_train, y_train): # logistic regression using stochastic gradint descent and cross entropy loss gradients. X_train is sparse matrix from CountVectorizer adn y_train is list of class labels
    
        num_samples, num_features = X_train.shape
        self.weights = np.zeros(num_features)
        self.bias = 0.0
        
        y_train = np.array(y_train)

        #stochastic gradient descent loop
        for epoch in range(self.epochs):
            # shuffle indices at the start of each epoch for true SGD
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            
            for i in indices:
                #extract the single document vector (sparse format)
                x_i = X_train[i]
                
                #forward pass-> compute the linear combination and apply sigmoid
                # x_i.dot(self.weights) returns a 1D array with 1 element
                linear_pred = x_i.dot(self.weights)[0] + self.bias
                y_pred = self.sigmoid(linear_pred)
                
                #compute the gradient of the cross-entropy loss
                error = y_pred - y_train[i]
                
                #backward pass-> update weights and bias
                # x_i.multiply(error) scales the sparse row, .toarray()[0] makes it a dense 1D array
                gradient_w = x_i.multiply(error).toarray()[0]
                
                self.weights -= self.learning_rate * gradient_w
                self.bias -= self.learning_rate * error

    def predict(self, X_test): # predicting class for the logistic regression classifier
        # we can predict the whole test set at once using matrix multiplication
        linear_pred = X_test.dot(self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)
        
        #connvert probabilities to binary class labels (1 if >= 0.5, else 0)
        return (y_pred >= 0.5).astype(int)