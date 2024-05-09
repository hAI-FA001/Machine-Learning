import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=500):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = np.array([])
        self.b = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def calc_z(self, x):
        # yhat = g(w . x + b)
        z = np.dot(self.w, x) + self.b
        return z
    
    def calc_loss(self, y, yhat):
        # -1/m * sum of (y * log(yhat) + (1-y) * log(1-yhat))
        # add 0.00001 to avoid log(0)
        m = 1 if type(y) not in [list, np.ndarray] else len(yhat)
        return np.sum(y * np.log(0.00001 + yhat) + (1-y) * np.log(0.00001 + 1-yhat)) / (-1 * m)
    
    def calc_gradient_descent_aggregate_term(self, z, y, yhat, X=None):
        # -1/m * derivative of loss * x for each w
        # or -1/m * derivative of loss for b
        
        # derivatives
        # y * log(yhat) + (1-y) * log(1-yhat) ---> y * 1/yhat * d(yhat) + (1-y) * 1/(1-yhat) * d(-yhat)
        # yhat = 1/(1 + e^-z) ---> -1 * (1 + e^-z)^-2 * d(e^-z)
        # e^-z = e^-(wx+b) ---> e^-(wx+b) * -x for w and e^-(wx+b) * -1 for b
        
        if X is None:
            deriv_sig_b = np.exp(-z) * -1
            deriv_yhat = -1 * ((1 + np.exp(-z))**-2) * deriv_sig_b
            deriv_b = y * 1/yhat * deriv_yhat + (1 - y) * 1/(1 - yhat) * -deriv_yhat
            
            return np.sum(deriv_b) / -len(y)
        else:
            deriv_term_for_each_w = []
            for col_num in range(len(X[0])):
                deriv_sig_W = np.exp(-z) * -X[:, col_num]
                deriv_yhat = -1 * ((1 + np.exp(-z))**-2) * deriv_sig_W
                deriv_W = y * 1/yhat * deriv_yhat + (1 - y) * 1/(1 - yhat) * -deriv_yhat    
                term = np.sum(deriv_W) / -len(y)
                
                deriv_term_for_each_w.append(term)
                
            return np.array(deriv_term_for_each_w)
        
    def train(self, X, y):
        self.w = np.zeros_like(X[0]).astype(np.float32)
        
        errs = []
        for _ in range(self.iterations):
            yhats = []
            z_values = []
            for i in range(len(y)):
                z = self.calc_z(X[i])
                z_values.append(z)
                yhats.append(self.sigmoid(z))
            
            yhats = np.array(yhats)
            
            err = self.calc_loss(y, yhats)
            errs.append(err)
            
            bias_agg_term = self.calc_gradient_descent_aggregate_term(z, y, yhats)
            weight_agg_term = self.calc_gradient_descent_aggregate_term(z, y, yhats, X)
            
            self.b -= self.learning_rate * bias_agg_term
            self.w -= self.learning_rate * weight_agg_term
        
        plt.plot(list(range(len(errs))), errs)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.xscale('log')
        plt.show()
        
    def predict(self, x):
        return self.sigmoid(self.calc_z(x))
    
    def params(self):
        return self.w, self.b
            
        

total_rows = 50
X = np.array(
    [
        [i, j, k] for i, j, k in zip(range(25, 25 + 50*25, 25), range(2, 2 + 50*2, 2), range(500, 500 + 50*1, 1))
    ]
)

y = np.array([1 if x[0] > 500 and x[1] < 10 and x[2] > 10 else 0 for x in X])

lr = LogisticRegression(learning_rate=0.000001, iterations=2000)
lr.train(X, y)

losses = []
for i in range(len(X)):
    losses.append(lr.calc_loss(y[i], np.array(lr.predict(X[i]))))

plt.plot(list(range(len(y))), losses)
plt.title('Difference in yhat and y for each example')
plt.show()

print('Mean Loss: ', np.array(losses).mean())
print('W and b: ', lr.params())