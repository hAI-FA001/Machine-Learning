import numpy as np

class MultiLayerNeuralNetwork():
    def __init__(self, num_inputs, num_neurons=[1], num_epochs=100, learning_rate=0.01, last_activation='sigmoid'):
        self.num_layers = len(num_neurons)  # length of list gives number of layers
        self.num_neurons = num_neurons
        
        
        self.weights = []
        self.biases = []
        
        for num_neurons_in_layer, input_shape in zip(num_neurons, [num_inputs] + num_neurons[:-1]):
            weights = np.random.rand(num_neurons_in_layer, input_shape) * 0.01  # initialize to small values
            biases = np.zeros((num_neurons_in_layer, 1)) # initialize to 0s
            
            self.weights.append(weights)
            self.biases.append(biases)
        
        
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.last_activation = last_activation.lower()
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def relu(self, x):
        out = x.copy()
        out[out < 0] = 0
        return out
    
    def relu_derivative(self, x):
        out = x.copy()
        out[out <= 0] = 0
        out[out > 0] = 1
        return out
    
    def predict(self, inputs):
        # z is before applying activation
        self.z = []
        # a is after applying activation
        self.a = []
        input_to_current_layer = inputs
        
        # loop through layers
        for layer in range(self.num_layers):
            weights = self.weights[layer]
            bias = self.biases[layer]
            
            # z = wx + b
            z = np.dot(weights, input_to_current_layer.reshape((-1, 1))) + bias
            
            if layer != self.num_layers-1:  # use relu if not last layer
                a = self.relu(z)
            else:
                a = self.sigmoid(z) if self.last_activation == 'sigmoid' else self.relu(z)  # use relu as default
            
            # store values during forward propagation (used in derivative calculations)
            self.z.append(z)
            self.a.append(a)
            
            # update input for next layer
            input_to_current_layer = a
            
        return self.a[-1]
    
    def train(self, training_inputs, labels):
        errs = []
        for epoch in range(self.num_epochs):
            err = 0
            for inputs, label in zip(training_inputs, labels):
                yhat = self.predict(inputs)
                # for Mean Squared Error (MSE)
                err += (label - yhat) ** 2
                
                # error = (label - yhat) ^ 2
                # derivative of error wrt yhat = 2 * (label - yhat) * derivative of (-yhat) = 2 * (label - yhat) * -1
                yhat_derivative = 2 * (label - yhat) * -1
                
                w_derivatives = [np.zeros(w.shape) for w in self.weights]
                b_derivatives = [np.zeros(b.shape) for b in self.biases]
                z_derivatives = [np.zeros(z.shape) for z in self.z]
                a_derivatives = [np.zeros(a.shape) for a in self.a]
                
                # last activation is same as yhat
                a_derivatives[-1] = yhat_derivative
                
                # chain rules
                # da/dz = derivative of activation function
                activation_derivative = self.sigmoid_derivative(self.z[-1]) if self.last_activation == 'sigmoid' else self.relu_derivative(self.z[-1])
                # dError/da * da/dz
                z_derivatives[-1] = a_derivatives[-1] * activation_derivative
                # z = wi * xi + b -> dz/dwi = xi
                input_to_z = self.a[-2].T if self.num_layers > 1 else np.array(inputs).T
                # da/dz * dz/dwi
                w_derivatives[-1] = z_derivatives[-1] * input_to_z
                # z = wi * xi + b -> dz/db = 1
                # da/dz * dz/db
                b_derivatives[-1] = z_derivatives[-1] * 1
                
                for i in reversed(range(self.num_layers - 1)):
                    # chain rules
                    # z = wi * ai + b -> dz/dai = wi
                    # calculate dz/d(a of layer i) * d(a of layer i+1)/dz
                    a_derivatives[i] = np.dot(self.weights[i+1].T, z_derivatives[i+1])
                    
                    # same chain rules as outside loop for these
                    z_derivatives[i] = a_derivatives[i] * self.relu_derivative(self.z[i])
                    w_derivatives[i] = z_derivatives[i] * self.a[i-1].T
                    b_derivatives[i] = z_derivatives[i] * 1
                    
                for i in range(self.num_layers):
                    self.weights[i] -= self.learning_rate * w_derivatives[i]
                    self.biases[i] -= self.learning_rate * b_derivatives[i]

            err /= len(labels)
            print(epoch, round(err.flatten()[0], 3))                    
            errs.append(err)
        
        errs = np.array(errs)
        min_err = np.min(errs)
        min_idx = np.argmin(errs)
        print(f'{min_err} at {min_idx}')
        
        # plot error against iteration
        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.plot(range(self.num_epochs), errs.reshape(-1))
        # use log scale
        plt.yscale('log')
        plt.xscale('log')
        plt.show()
                    

                    
# training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# labels = np.array([0, 0, 0, 1])
training_inputs = np.array([[i] for i in range(1, 5)])
labels = np.array([2*x[0] + 10 for x in training_inputs])

neural_network = MultiLayerNeuralNetwork(num_inputs=1, num_neurons=[2, 2, 1], num_epochs=5_000, learning_rate=2e-5, last_activation='relu')
neural_network.train(training_inputs, labels)

print(f'{"Inputs":10} {"Actual Output":15} {"Prediction":10}')
predictions = []
for idx in range(len(training_inputs)):
    x = training_inputs[idx]
    y = labels[idx]
    yhat = neural_network.predict(x).flatten()[0]
    predictions.append(yhat)
    
    print(f'{x} {round(y, 2):15} {np.round(yhat, 2)}')


# plot input vs actual output, and input vs predicted output
import matplotlib.pyplot as plt
plt.figure()
plt.plot(training_inputs, labels.reshape((-1,)), 'r')
plt.plot(training_inputs, np.array(predictions).reshape((-1,)), 'g')
plt.ylim(bottom=0)
plt.show()

