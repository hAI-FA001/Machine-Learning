import numpy as np
from collections import Counter
from functools import reduce

from decision_tree import DecisionTree

class RandomForest:
    def __init__(self, num_trees=1, criteria='entropy', randomly_drop_columns=False):
        self.num_trees = num_trees
        
        # do NOT initialize as [DecisionTree()] * self.num_trees, otherwise all DecisionTree will point to same object
        self.trees = [DecisionTree(criteria=criteria.lower()) for i in range(num_trees)]
        
        self.drop_cols = randomly_drop_columns
        self.cols_to_keep = [None] * self.num_trees
        
    def train(self, X, y):
        num_rows, num_cols = X.shape
        
        shuffled_datasets = [None] * self.num_trees
        for i in range(self.num_trees):
            # randomly select rows from dataset
            indices_to_pick = [np.random.randint(low=0, high=num_rows) for i in range(num_rows)]

            self.cols_to_keep[i] = list(range(num_cols))
            if self.drop_cols:
                # randomly select which columns to keep
                self.cols_to_keep[i] = [np.random.randint(low=0, high=num_cols) for i in range(num_cols)]
                self.cols_to_keep[i] = list(set(self.cols_to_keep[i]))
            
            # store new X and y
            shuffled_datasets[i] = (X[indices_to_pick][:, self.cols_to_keep[i]], y[indices_to_pick])
            
        for i in range(self.num_trees):
            X_shuffled, y_shuffled = shuffled_datasets[i]
            self.trees[i].train(X_shuffled, y_shuffled)
            
    def predict(self, x):
        # get predictions from all trees
        predictions = []
        for i in range(self.num_trees):
            x_with_cols = x[self.cols_to_keep[i]]
            y_pred = self.trees[i].predict(x_with_cols)
            predictions.append(y_pred)
        
        # find majority label
        counts = Counter(predictions)
        majority = reduce(lambda pred_1, pred_2: pred_1 if counts[pred_1] > counts[pred_2] else pred_2, counts.keys())
        
        return majority
    
    
if __name__ == '__main__':
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

    rf = RandomForest(num_trees=5, criteria='gini', randomly_drop_columns=True)
    rf.train(X, y)
    
    accuracy = 0
    for index in range(X.shape[0]):
        y_pred = rf.predict(X[index])
        print(f'{y_pred}, {y[index]}')
        accuracy += (y_pred == y[index])
    
    print(f'\nAccuracy: {accuracy / len(y) * 100} %')