import numpy as np
from collections import Counter

class NaiveBayesClassifier:
    def __init__(self):
        pass
    
    def train(self, X, y):
        total_count = len(y)
        class_counts = dict(Counter(y))
        # evidence for each class, P(C=c_i)
        self.class_probs = {label:count/total_count for label, count in class_counts.items()}
        
        # make conditional probability tables for col number x attribute value x label
        # AKA compute P(column = attribute value | label)
        prob_attr_given_label = Counter()
        # count attribute and label pairs
        for row, label in zip(X, y):
            # col is used in case 2 columns have same attribute value
            for col, attr in enumerate(row):
                prob_attr_given_label[(col, attr, label)] += 1
        # turn the counts into probabilities
        for col, attr, label in prob_attr_given_label:
            prob_attr_given_label[(col, attr, label)] /= class_counts.get(label)
        
        self.prob_attr_given_label = prob_attr_given_label
        
    def predict(self, x):
        # compute P(X=x | C=c_i) = P(C=c_i) * P(x_i | c_i) for each x_i attribute and c_i class
        conditional_probs = {label: prob_label for label, prob_label in self.class_probs.items()}
        for label in self.class_probs:
            for col, attr in enumerate(x):
                prob_attr_given_label = self.prob_attr_given_label.get((col, attr, label), 0)
                conditional_probs[label] *= prob_attr_given_label
        
        # find index for maximum probability
        index_of_max_prob = np.argmax(np.array(list(conditional_probs.values())))
        # return corresponding label
        return list(conditional_probs.keys())[index_of_max_prob]
            
        
        
data = np.array(
        [
            ['Sunny', 'Yes', 'Rich', 'Cinema'],
            ['Sunny', 'No', 'Rich', 'Tennis'],
            ['Windy', 'Yes', 'Rich', 'Cinema'],
            ['Rainy', 'Yes', 'Poor', 'Cinema'],
            ['Rainy', 'No', 'Rich', 'Stay In'],
            ['Rainy', 'Yes', 'Poor', 'Cinema'],
            ['Windy', 'No', 'Poor', 'Cinema'],
            ['Windy', 'No', 'Rich', 'Shopping'],
            ['Windy', 'Yes', 'Rich', 'Cinema'],
            ['Sunny', 'No', 'Rich', 'Tennis']
        ]
    )

X = data[:, :-1]
y = data[:, -1]
    
nb = NaiveBayesClassifier()
nb.train(X, y)
preds = []
for row in X:
    pred = nb.predict(row)
    preds.append(pred)
    
print(list(zip(preds, y)))