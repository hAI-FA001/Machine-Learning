import numpy as np

class KNearestNeighbors:
    def __init__(self, k=5, distance='euclidean'):
        self.k = k
        self.distance = distance.lower()
        self.x = []
        self.y = []
        self.total_classes = 0
        
    def calc_euclidean(self, x1, x2):
        return np.sqrt(np.sum((x2 - x1) ** 2))
    
    def calc_manhattan(self, x1, x2):
        return np.sum(np.abs(x2 - x1))
    
    def calc_chessboard(self, x1, x2):
        return np.max(np.abs(x2 - x1))

    def normalize_x(self):
        min_value = np.min(self.x, axis=0)
        max_value = np.max(self.x, axis=0)
        
        # makes values of x range from 0 to 1
        self.x = (self.x - min_value) / (max_value - min_value)
        
    def train(self, X, y):
        # no actual training, just save X & y
        self.x = X
        self.y = y
        # get total classes by getting unique values in y
        self.total_classes = set(y)
        
        self.normalize_x()
    
    def predict(self, x):
        # return if no training data stored
        if len(self.x) == 0:
            return None
        
        differences = []
        for index, each_row in enumerate(self.x):
            diff = 0
            if self.distance == 'chessboard':
                diff = self.calc_chessboard(each_row, x)
            elif self.distance == 'manhattan':
                diff = self.calc_manhattan(each_row, x)
            else:
                diff = self.calc_euclidean(each_row, x)
            
            # store difference and corresponding label as tuple
            diff_with_label = (diff, self.y[index])
            differences.append(diff_with_label)
            
        # sort in descending order based on difference (first value in tuple)
        differences.sort(reverse=True, key=lambda x: x[0])
        # get kth closest rows
        top_k = differences[:self.k]
        
        # store count of labels
        counts = {}
        for difference, label in top_k:
            try:
                counts[label] += 1
            except:
                counts[label] = 1
        
        # find majority class/label
        majority = None
        current_max_count = -1
        for label in counts.keys():
            if counts[label] > current_max_count:
                current_max_count = counts[label]
                majority = label
        
        return majority
        

X = np.array([
    # name, sweetness, crunchiness
    ['apple',     10,  9],
    ['bacon',      1,  4],
    ['banana',    10,  1],
    ['carrot',     7, 10],
    ['celery',     3, 10],
    ['cheese',     1,  1],
    ['green bean', 3,  7],
    ['grape',      8,  5],
    ['nuts',       3,  6],
    ['orange',     7,  3]
])
# ignore "name" in training data, only use sweetness and crunchiness
X = X[:, 1:].astype(np.int32)

y = np.array([
    'fruit', 'protein', 'fruit', 'vegetable', 'vegetable', 'protein', 'vegetable', 'fruit', 'protein', 'fruit'
])



# find accuracy of classifier for different values of k and different distances
errors = []
for k in range(1, len(X)):
    knn_classifiers = [KNearestNeighbors(k=k, distance='euclidean'), KNearestNeighbors(k=k, distance='manhattan'), KNearestNeighbors(k=k, distance='chessboard')]
    
    for knn_classifier in knn_classifiers:
        knn_classifier.train(X, y)
    
    total_examples = len(X)
    total_misclassified_examples = np.array([0, 0, 0])
    
    for idx, each_training_example in enumerate(X):
        yhats = [knn_classifier.predict(each_training_example) for knn_classifier in knn_classifiers]
        actual = y[idx]
        
        for classifier_idx, yhat in enumerate(yhats):
            if yhat != actual:
                total_misclassified_examples[classifier_idx] += 1
    
    error = total_misclassified_examples / total_examples
    errors.append(error)


# create plot of value of k against error for each distance
import matplotlib.pyplot as plt
plt.figure()

color_of_lines = ['r', 'g', 'b']
name_of_lines = ['euclidean', 'manhattan', 'chessboard']
errors = np.array(errors)
for ith_classifier in range(len(color_of_lines)):
    plt.plot(list(range(1, len(X))), errors[:, ith_classifier], color_of_lines[ith_classifier], label=name_of_lines[ith_classifier])

plt.legend()
plt.ylim(bottom=0, top=1)
plt.show()
