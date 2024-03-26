import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=500):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = 0.0
        self.b = 0.0
        
    def calc_yhat(self, x):
        # yhat = w . x + b
        return self.w * x + self.b
    
    def calc_mse(self, y, yhat):
        # 1/2m * sum of (yhat - y) ^ 2
        return np.sum(np.square(yhat - y)) / (2 * len(y))
    
    def calc_gradient_descent_aggregate_term(self, y, yhat, X=None):
        # 1/m * (yhat - y) * x for w
        # or just 1/m * (yhat - y) for b
        if X is None:
            return np.sum(yhat - y) / len(y)
        else:
            return np.sum((yhat - y) * X) / len(y)
        
    def train(self, X, y):
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
        


X = np.array(
    [
        i for i in range(25, 300, 25)
    ]
)

y = 500 * X + 20

lr = LinearRegression(learning_rate=0.000001, iterations=2000)
lr.train(X, y)

yhats = []
for x in X:
    yhats.append(lr.predict(x))
    
plt.plot(X, y)
plt.plot(X, yhats)
plt.show()

print('MSE: ', lr.calc_mse(y, yhats))
print('w and b: ', lr.params())