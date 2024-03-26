import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=500):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = np.array([])
        self.b = 0
        
    def calc_yhat(self, x):
        # yhat = w . x + b
        return np.dot(self.w, x) + self.b
    
    def calc_mse(self, y, yhat):
        # 1/2m * sum of (yhat - y) ^ 2
        return np.sum(np.square(yhat - y)) / (2 * len(y))
    
    def calc_gradient_descent_aggregate_term(self, y, yhat, X=None):
        # 1/m * (yhat - y) * x for each w
        # or just 1/m * (yhat - y) for b
        if X is None:
            return np.sum(yhat - y) / len(y)
        else:
            deriv_term_for_each_w = []
            y_diff = yhat - y
            for col_num in range(len(X[0])):
                term = np.sum(y_diff * X[:, col_num]) / len(y)
                deriv_term_for_each_w.append(term)
            return np.array(deriv_term_for_each_w)
        
    def train(self, X, y):
        self.w = np.zeros_like(X[0]).astype(np.float32)
        
        errs = []
        for _ in range(self.iterations):
            yhats = []
            for i in range(len(y)):
                yhat_i = self.calc_yhat(X[i])
                yhats.append(yhat_i)
            err = self.calc_mse(y, yhats)
            errs.append(err)
            
            bias_agg_term = self.calc_gradient_descent_aggregate_term(y, yhats)
            weight_agg_term = self.calc_gradient_descent_aggregate_term(y, yhats, X)
            
            self.b -= self.learning_rate * bias_agg_term
            self.w -= self.learning_rate * weight_agg_term
        
        plt.plot(list(range(len(errs))), errs)
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.xscale('log')
        plt.show()
        
    def predict(self, x):
        return self.calc_yhat(x)
    
    def params(self):
        return self.w, self.b
            
        

total_rows = 50
X = np.array(
    [
        [i, j, k] for i, j, k in zip(range(25, 25 + 50*25, 25), range(2, 2 + 50*2, 2), range(500, 500 + 50*1, 1))
    ]
)

y = 10 * X[:, 0] + 0.1 * X[:, 1] + 20 * X[:, 2] + 200

lr = LinearRegression(learning_rate=0.000001, iterations=2000)
lr.train(X, y)

yhats = []
for x in X:
    yhats.append(lr.predict(x))
    
plt.plot(list(range(len(y))), yhats - y)
plt.title('Difference in yhat and y for each example')
plt.show()

print('MSE: ', lr.calc_mse(y, yhats))
print('W and b: ', lr.params())