import numpy as np
from collections import Counter

def calc_gaussian_dst(mean, std, x):
    # 1/(2*pi*std)^1/2 * e^-((x - mu)^2 / (2 * std^2))
    # add 1e-10 for numerical stability
    return np.exp(-((x - mean)**2 / (2 * std**2 + 1e-10))) / (np.sqrt(2 * np.pi * std**2) + 1e-10)

class NaiveBayesClassifier:
    def __init__(self):
        pass
    
    def train(self, X, y):
        total_count = len(y)
        class_counts = dict(Counter(y))
        # evidence for each class, P(C=c_i)
        self.class_probs = {label:count/total_count for label, count in class_counts.items()}
        
        # calculate mean and variance for each column, separately for each label
        # e.g. mean and variance for col 1 and within those rows with label A
        col_means_stds_label = {}
        for label in class_counts:
            relevant_rows = X[y == label]
            
            col_mean_with_label = np.mean(relevant_rows, axis=0)
            col_std_with_label = np.std(relevant_rows, axis=0)
            
            for col_num in range(len(X[0])):
                col_means_stds_label[(col_num, label)] = (col_mean_with_label[col_num], col_std_with_label[col_num])
        
        self.col_means_stds_label = col_means_stds_label
        
    def predict(self, x):
        # compute P(X=x | C=c_i) = P(C=c_i) * P(x_i | c_i) for each x_i attribute and c_i class
        conditional_probs = {label: prob_label for label, prob_label in self.class_probs.items()}
        for col_num, label in self.col_means_stds_label:
            mean, std = self.col_means_stds_label.get((col_num, label), (0, 0))
            # P(x_i | c_i), calculated using Gaussian Distribution
            conditional_probs[label] *= calc_gaussian_dst(mean, std, x[col_num])
            
        # find index for maximum probability
        index_of_max_prob = np.argmax(np.array(list(conditional_probs.values())))
        # return corresponding label
        return list(conditional_probs.keys())[index_of_max_prob]
            
        
        
data = np.array(
        [
            [np.random.normal(loc=1.0, scale=0.5), np.random.normal(loc=15.0, scale=3.5), 'Cinema'],
            [np.random.normal(loc=1.0, scale=0.5), np.random.normal(loc=15.0, scale=3.5), 'Tennis'],
            [np.random.normal(loc=1.0, scale=0.5), np.random.normal(loc=15.0, scale=3.5), 'Cinema'],
            [np.random.normal(loc=1.0, scale=0.5), np.random.normal(loc=15.0, scale=3.5), 'Cinema'],
            [np.random.normal(loc=1.0, scale=0.5), np.random.normal(loc=15.0, scale=3.5), 'Stay In'],
            [np.random.normal(loc=1.0, scale=0.5), np.random.normal(loc=15.0, scale=3.5), 'Cinema'],
            [np.random.normal(loc=1.0, scale=0.5), np.random.normal(loc=15.0, scale=3.5), 'Cinema'],
            [np.random.normal(loc=1.0, scale=0.5), np.random.normal(loc=15.0, scale=3.5), 'Shopping'],
            [np.random.normal(loc=1.0, scale=0.5), np.random.normal(loc=15.0, scale=3.5), 'Cinema'],
            [np.random.normal(loc=1.0, scale=0.5), np.random.normal(loc=15.0, scale=3.5), 'Tennis']
        ]
    )

X = data[:, :-1].astype(np.float32)
y = data[:, -1]
    
nb = NaiveBayesClassifier()
nb.train(X, y)
preds = []
for row in X:
   pred = nb.predict(row)
   preds.append(pred)
    
print(list(zip(preds, y)))