from collections import Counter
from functools import reduce
import numpy as np

class DecisionNode:
    def __init__(self, col_value_in_this_subtree=None, col_to_split_at=-1, child_nodes=[]):
        self.col_value_in_this_subtree = col_value_in_this_subtree
        self.col_to_split_at = col_to_split_at
        self.child_nodes = child_nodes
        
class LeafNode:
    def __init__(self, col_value_in_this_subtree=None, labels=[]):
        self.col_value_in_this_subtree = col_value_in_this_subtree
        self.labels = labels
        
    def getMajority(self):
        counts = Counter(self.labels)
        
        majority_label = reduce(lambda pred_1, pred_2: pred_1 if counts[pred_1] > counts[pred_2] else pred_2, counts.keys())
        
        return majority_label


class DecisionTree():
    def __init__(self, criteria='entropy'):
        self.criteria = criteria.lower()
        self.tree = None
    
    def calc_entropy(self, y):
        # make y 1-dimensional
        y = y.flatten()
        
        # find fraction p_c of label c in dataset y
        fractions = Counter(y)
        total = len(y)
        for label in fractions.keys():
            fractions[label] /= total
        
        # calculate p_c*log(p_c) for each fraction
        entropy = map(lambda a: a*np.log2(a), fractions.values())
        # sum the fractions
        entropy = reduce(lambda a, b: a+b, entropy)
        # -ve sign
        entropy = -entropy
        
        return entropy
    
    def calc_gini_index(self, y):
        # make y 1-dimensional
        y = y.flatten()
        
        # find fraction p_c of label c in dataset y
        fractions = Counter(y)
        total = len(y)
        for label in fractions.keys():
            fractions[label] /= total
        
        # calculate p_c^2 for each fraction
        gini_index = map(lambda a: a**2, fractions.values())
        # sum the fractions
        gini_index = reduce(lambda a, b: a+b, gini_index)
        # subtract from 1
        gini_index = 1 - gini_index
        
        return gini_index
    
    def find_max_gain_col(self, X, y, cols_to_ignore):
        # entropy/gini index at root node
        criteria_value_at_root = self.calc_gini_index(y) if self.criteria == 'gini' else self.calc_entropy(y)
        
        num_cols = X.shape[1]
        total_rows = X.shape[0]
        info_gain_for_each_col = {}
        
        # calculate info gain for each column
        for col_number in range(num_cols):
            if col_number in cols_to_ignore: continue
            
            current_feature_col = X[:, col_number]
            
            # count # of rows for each value of feature/column
            counts = Counter(current_feature_col)
            
            # weights are used when calculating weighted sum
            weights = {}
            for feature_value in counts.keys():
                weights[feature_value] = counts[feature_value] / total_rows
            
            # calculate 2nd term in info gain forumula: entropy at root - weighted sum of (entropies for this column)
            criteria_values = {}
            for feature_value in counts.keys():
                # get rows that have current value of feature & their labels
                index_of_rows_with_this_value = np.argwhere(current_feature_col == feature_value)

                corresponding_labels = y[index_of_rows_with_this_value]
                # calculate entropy/gini index for this feature value
                criteria_value = self.calc_gini_index(corresponding_labels) if self.criteria == 'gini' else self.calc_entropy(corresponding_labels)
                criteria_values[feature_value] = criteria_value
                
            gain = 0
            # weighted sum/2nd term in info gain forumula: entropy at root - weighted sum of (entropies for this column)
            for feature_value in counts.keys():
                gain += (weights[feature_value] * criteria_values[feature_value])
            # info gain = entropy at root - above weighted sum
            gain = criteria_value_at_root - gain
            
            info_gain_for_each_col[col_number] = gain
            
        # find column index that gives maximum info gain
        index_of_col_giving_max_gain = reduce(lambda col_a, col_b: col_a if info_gain_for_each_col[col_a] > info_gain_for_each_col[col_b] else col_b, info_gain_for_each_col.keys())

        return index_of_col_giving_max_gain
        
    def split_dataset(self, X, y, col_number):
        selected_column = X[:, col_number]
        unique_values_in_that_column = set(selected_column)
    
        # split current dataset based on values of selected column
        splitted_datasets = []
        for value in unique_values_in_that_column:
            indices_with_this_value = np.argwhere(selected_column == value).flatten()
            splitted_datasets.append((X[indices_with_this_value], y[indices_with_this_value], value))
    
        return splitted_datasets
    
    def create_decision_tree(self, X, y, branch_value=None, cols_to_ignore=[]):
        # no more columns
        if X.shape[1] == 0 or X.shape[1] == len(cols_to_ignore):
            return LeafNode(col_value_in_this_subtree=branch_value, labels=y)
        # no entropy
        if len(set(y.flatten())) == 1:
            return LeafNode(col_value_in_this_subtree=branch_value, labels=y)
        
        index_of_max_gain = self.find_max_gain_col(X, y, cols_to_ignore)
        sub_datasets = self.split_dataset(X, y, index_of_max_gain)
        sub_trees = []
        cols_to_ignore_next = cols_to_ignore + [index_of_max_gain]
        
        # recursively make decision tree
        for dataset in sub_datasets:
            current_X, current_y, feature_value = dataset
            sub_tree = self.create_decision_tree(current_X, current_y, feature_value, cols_to_ignore_next)
            sub_trees.append(sub_tree)

        node = DecisionNode(col_value_in_this_subtree=branch_value, col_to_split_at=index_of_max_gain, child_nodes=sub_trees)
        return node
        
        
    def train(self, X, y):
        self.tree = self.create_decision_tree(X, y)
        
    def predict(self, x):
        if self.tree is None:
            return None

        # traverse tree
        queue = [self.tree]
        while len(queue) > 0:
            cur_node = queue.pop(0)
            
            if isinstance(cur_node, LeafNode):
                return cur_node.getMajority()
            
            for child in cur_node.child_nodes:
                if child.col_value_in_this_subtree == x[cur_node.col_to_split_at]:
                    queue.append(child)


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

    model = DecisionTree(criteria='gini')
    model.train(X, y)

    accuracy = 0
    for row_number in range(X.shape[0]):
        y_pred = model.predict(X[row_number])
        y_actual = y[row_number]
        print(f'{y_actual}, {y_pred}')
        accuracy += (y_pred == y_actual)
        
    print(f'\nAccuracy: {accuracy / len(y) * 100} %')