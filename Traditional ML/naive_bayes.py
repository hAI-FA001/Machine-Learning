import numpy as np
from collections import Counter

def bayes_rule(likelihood, prior, evidence):
    return likelihood * prior / evidence

class NaiveBayes:
    def __init__(self, num_similar_obs=5):
        # number of observations to consider when predicting
        self.num_similar_obs = num_similar_obs
        
    def calc_distance(self, x1, x2):
        # use euclidean distance to find similar observations
        return np.sqrt(np.sum(np.square(x1 - x2)))
    
    def train(self, X, y):
        class_counts = dict(Counter(y))
        self.class_counts = class_counts
        # P(Y) for each label, these are the prior
        self.class_probs = {label:class_counts[label] / y.shape[0] for label in class_counts}
        self.y = y
        self.X = X
        
    
    def predict(self, x):
        # this is marginal likelihood or evidence
        # can ignore this in posterior calculations because every calculation is divided by it
        prob_of_x = self.num_similar_obs / self.X.shape[0]
        
        # similar to K-NN, get similar observations
        distances = [(self.calc_distance(x, row), label) for row, label in zip(self.X, self.y)]
        distances.sort(key=lambda dist_and_label: dist_and_label[0])
        distances = distances[:self.num_similar_obs]
        
        class_counts_in_obs = dict(Counter(map(lambda dist_and_label: dist_and_label[1], distances)))
        
        posteriors = []
        for label in self.class_counts:
            count_in_obs = class_counts_in_obs.get(label, 0)
            count_in_total = self.class_counts[label]
            
            # this is the likelihood
            prob_of_x_given_label = count_in_obs / count_in_total
            # this is the final posterior
            prob_of_label_given_x = bayes_rule(likelihood=prob_of_x_given_label, prior=self.class_probs[label], evidence=prob_of_x)
            
            posteriors.append((prob_of_label_given_x, label))
        
        # find label associated with max posterior probability
        index_of_max_prob = np.argmax(np.array(list(map(lambda prob_and_label: prob_and_label[0], posteriors))))
        prob_and_pred_label = posteriors[index_of_max_prob]
        
        return prob_and_pred_label[1]
    

X = np.array(
        [
            [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
            [[1, 0, 0], [0, 1, 0], [1, 0, 0]],
            [[0, 1, 0], [1, 0, 0], [1, 0, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 1, 0], [1, 0, 0]],
            [[0, 1, 0], [1, 0, 0], [1, 0, 0]],
            [[1, 0, 0], [0, 1, 0], [1, 0, 0]]
        ]
    )
y = np.array([
    'Cinema',
    'Tennis',
    'Cinema',
    'Cinema',
    'Stay In',
    'Cinema',
    'Cinema',
    'Shopping',
    'Cinema',
    'Tennis'
    ])

nb = NaiveBayes(num_similar_obs=3)
nb.train(X, y)
preds = []
for row in X:
    pred = nb.predict(row)
    preds.append(pred)
    
print(list(zip(preds, y)))