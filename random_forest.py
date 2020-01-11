import numpy as np
import scipy.io
from scipy import stats
import random

class Node:

    def __init__(self, split_rule, left, right, depth, label=None):
        """
        TODO: initialization of a decision tree
        """
        self.split_rule = split_rule
        self.left = left
        self.right = right
        self.depth = depth
        self.label = label
    
    def is_leaf(self):
        return self.label is not None
    
    def __repr__(self):
        if self.is_leaf():
            return "label: " + str(class_names[self.label])
        else:
            return "(word: {}, thresh: {})".format(features[self.split_rule[0]], 
                                                   self.split_rule[1]) + \
                   "\n" + "      " * self.depth + \
                   "  └---" + self.left.__repr__() + \
                   "\n" + "      " * self.depth + \
                   "  └---" + self.right.__repr__() 

class DecisionTree:

    def __init__(self, max_depth, impurity_thresh):
        """
        TODO: initialization of a decision tree
        """
        self.max_depth = max_depth
        self.impurity_thresh = impurity_thresh
        self.node = None
        
    @staticmethod
    def entropy(y):
        """
        TODO: implement a method that calculates the entropy given all the labels
        """
        # assume we have "0" or "1", compute proportions by dividing frequencies by length.
        y_prob = np.array([sum(y==0), sum(y==1)]) / len(y)
        return -(y_prob) @ np.log(y_prob)

    @staticmethod
    def information_gain(X, y, thresh):
        """
        TODO: implement a method that calculates information gain given a vector of features
        and a split threshold
        """
        info_curr = DecisionTree.entropy(y)

        X_left_prop = sum(X >= thresh) / len(X)
        y_left = y[np.where(X >= thresh)]
        X_right_prop = sum(X < thresh) / len(X)
        y_right = y[np.where(X < thresh)]

        info_new = X_left_prop * DecisionTree.entropy(y_left) + \
                   X_right_prop * DecisionTree.entropy(y_right)
        
        return info_curr - info_new

    @staticmethod
    def gini_impurity(y):
        """
        TODO: implement a method that calculates the gini impurity given all the labels
        """
        # assume we have "0" or "1", compute proportions by dividing frequencies by length.
        y_prob = np.array([sum(y==0), sum(y==1)]) / len(y)
        return 1 - y_prob @ y_prob

    @staticmethod
    def gini_purification(X, y, thresh):
        """
        TODO: implement a method that calculates reduction in impurity gain given a vector of features
        and a split threshold
        """
        # Compute
        gini_curr = DecisionTree.gini_impurity(y)

        X_left_prop = sum(X >= thresh) / len(X)
        y_left = y[np.where(X >= thresh)]
        X_right_prop = sum(X < thresh) / len(X)
        y_right = y[np.where(X < thresh)]

        gini_new = X_left_prop * DecisionTree.gini_impurity(y_left) + \
                   X_right_prop * DecisionTree.gini_impurity(y_right)
        
        return gini_curr - gini_new

    def split(self, X, y, idx, thresh):
        """
        TODO: implement a method that return a split of the dataset given an index of the feature and
        a threshold for it
        """
        X_left = X[np.where(X[:, idx] >= thresh)]
        y_left = y[np.where(X[:, idx] >= thresh)]
        X_right = X[np.where(X[:, idx] < thresh)]
        y_right = y[np.where(X[:, idx] < thresh)]

        return X_left, y_left, X_right, y_right
    
    def segmenter(self, X, y):
        """
        TODO: compute entropy gain for all single-dimension splits,
        return the feature and the threshold for the split that
        has maximum gain
        """
        # initialize features
        best_idx, best_thresh, best_gain = 0, 0, 0

        # iterate through feature indices to find the best feature.
        for idx in range(X.shape[1]):

            vals = np.unique(X[:, idx])

            # below is a naive method where 
            # for j in range(1, len(vals) - 1):
            #     thresh = (vals[j] + vals[j+1]) / 2
            #     X_left, y_left, X_right, y_right = self.split(X, y, idx, thresh)

            #     if len(y_left) == 0 or len(y_right) == 0:
            #         continue

            #     # compute the gain with the current feature and threshold.
            #     gain = DecisionTree.gini_purification(X[:, idx], y, thresh)

            #     # update if the gain is the highest so far.
            #     if gain > best_gain:
            #         best_idx, best_thresh, best_gain = idx, thresh, gain
            

            # now, in order to raise the computation speed, we use 
            # the median of the unique values in the selected feature vector
    
            thresh = np.median(vals)
            X_left, y_left, X_right, y_right = self.split(X, y, idx, thresh)

            # skips if one of the partition set is empty.
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            # compute the gain with the current feature and threshold.
            gain = DecisionTree.gini_purification(X[:, idx], y, thresh)

            # update if the gain is the highest so far.
            if gain > best_gain:
                best_idx, best_thresh, best_gain = idx, thresh, gain
        return best_idx, best_thresh
    
    def train(self, X, y, depth):
        """
        TODO: fit the model to a training set. Think about what would be 
        your stopping criteria
        """
        # check stopping criterion (max depth and impurity threshold).
        if depth > self.max_depth or DecisionTree.gini_impurity(y) < self.impurity_thresh:
            label = int(sum(y==1) > sum(y==0))
            return Node(None, None, None, depth, label=label)
        else:
            split_rule = self.segmenter(X, y)
            X_left, y_left, X_right, y_right = self.split(X, y, split_rule[0], split_rule[1])

            # if there is only one label, make it a leaf.
            if len(y_left) == 0 or len(y_right) == 0:
                label = int(sum(y==1) > sum(y==0))
                if depth == 0:
                    self.node = Node(None, None, None, depth, label=label)
                else:
                    return Node(None, None, None, depth, label=label)
            else:

                # recursively add left and right nodes.
                left = self.train(X_left, y_left, depth+1)
                right = self.train(X_right, y_right, depth+1) 

                # if it's at the root, set it as self.node .
                if depth == 0:
                    self.node = Node(split_rule, left, right, depth)
                else:
                    return Node(split_rule, left, right, depth)

    def predict(self, X, suppress_print=True):
        """
        TODO: predict the labels for input data 
        """
        num_samples = X.shape[0]
        predictions = np.empty(num_samples)
        for i in range(num_samples):
            sample = X[i]
            curr_node = self.node

            # iterate until current node is a leaf.
            while not curr_node.is_leaf():
                if X[i, curr_node.split_rule[0]] >= curr_node.split_rule[1]:
                    if not suppress_print:
                        print("({}) >= {}".format(features[curr_node.split_rule[0]],
                                                  curr_node.split_rule[1]))
                    curr_node = curr_node.left
                else:
                    if not suppress_print:
                        print("({}) < {}".format(features[curr_node.split_rule[0]],
                                                 curr_node.split_rule[1]))
                    curr_node = curr_node.right
            predictions[i] = curr_node.label
            if not suppress_print:
                print("Therfore this email was {}".format(class_names[curr_node.label]))
        return predictions

    def __repr__(self):
        """
        TODO: one way to visualize the decision tree is to write out a __repr__ method
        that returns the string representation of a tree. Think about how to visualize 
        a tree structure. You might have seen this before in CS61A.
        """
        return self.node.__repr__()


class RandomForest():
    
    def __init__(self, 
                 bagging_rate, 
                 feature_rate, 
                 num_trees, 
                 max_depth, 
                 impurity_thresh):
        """
        TODO: initialization of a random forest
        """
        self.bagging_rate = bagging_rate
        self.feature_rate = feature_rate
        self.num_trees = num_trees
        self.trees = [DecisionTree(max_depth, impurity_thresh) for _ in range(num_trees)]
        self.feature_i_list = []

    def fit(self, X, y):
        """
        TODO: fit the model to a training set.
        """
        for tree in self.trees:
            data_indices = np.random.choice(range(X.shape[0]), 
                                            int(X.shape[0] * self.bagging_rate),
                                            replace=False)
            feature_indices = np.random.choice(range(X.shape[1]),
                                               int(X.shape[1] * self.feature_rate),
                                               replace=False)
            self.feature_i_list.append(feature_indices)
            tree.train(X[data_indices][:, feature_indices], y[data_indices], 0)
    
    def predict(self, X):
        """
        TODO: predict the labels for input data 
        """
        num_samples = X.shape[0]
        predictions = np.empty(num_samples)
        all_predictions = np.array([self.trees[i].predict(X[:, self.feature_i_list[i]])  
                                    for i in range(self.num_trees)])

        # return the po
        for i in range(num_samples):
            y_hat = all_predictions[:, i]
            predictions[i] = int(sum(y_hat==1) > sum(y_hat==0))
        return predictions        

features = ["pain", "private", "bank", "money", "drug", "spam", "prescription",
        "creative", "height", "featured", "differ", "width", "other",
        "energy", "business", "message", "volumes", "revision", "path",
        "meter", "memo", "planning", "pleased", "record", "out",
        "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
        "square_bracket", "ampersand"]

class_names = ["Ham", "Spam"]

if __name__ == "__main__":

    features = [
        "pain", "private", "bank", "money", "drug", "spam", "prescription",
        "creative", "height", "featured", "differ", "width", "other",
        "energy", "business", "message", "volumes", "revision", "path",
        "meter", "memo", "planning", "pleased", "record", "out",
        "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
        "square_bracket", "ampersand"
    ]
    assert len(features) == 32

    # Load spam data
    path_train = 'datasets/spam-dataset/spam_data.mat'
    data = scipy.io.loadmat(path_train)
    X = data['training_data']
    y = np.squeeze(data['training_labels'])
    class_names = ["Ham", "Spam"]
     

    """
    TODO: train decision tree/random forest on different datasets and perform the tasks 
    in the problem
    """

    # 80 / 20 random split
    train_indices = np.random.choice(range(X.shape[0]), 
                                 int(X.shape[0] * 0.8),
                                 replace=False)

    train_mask = np.zeros(X.shape[0], dtype=bool)
    train_mask[train_indices] = True

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_validation = X[~train_mask]
    y_validation = y[~train_mask]

    # 3.3 Performance Evaluation

    # Decision Tree
    classifier = DecisionTree(max_depth=20, impurity_thresh=0.01)
    classifier.train(X_train, y_train, 0)

    y_train_hat = classifier.predict(X_train)
    y_validation_hat = classifier.predict(X_validation)

    train_acc = sum(y_train==y_train_hat) / len(y_train_hat)
    valid_acc = sum(y_validation==y_validation_hat) / len(y_validation_hat)

    print("train accuracy: ", train_acc)
    print("validation accuracy: ", valid_acc)

    # Random Forest
    classifier = RandomForest(bagging_rate=0.7,
                              feature_rate=0.7,
                              num_trees=9, 
                              max_depth=20, 
                              impurity_thresh=0.01)
    classifier.fit(X_train, y_train)

    y_train_hat = classifier.predict(X_train)
    y_validation_hat = classifier.predict(X_validation)

    train_acc = sum(y_train==y_train_hat) / len(y_train_hat)
    valid_acc = sum(y_validation==y_validation_hat) / len(y_validation_hat)

    print("train accuracy: ", train_acc)
    print("validation accuracy: ", valid_acc)

    # 3.4.2 Stating the splits
    classifier = DecisionTree(max_depth=3, impurity_thresh=0.01)
    classifier.train(X_train, y_train, 0)
    classifier.predict(X_validation[0:1, :], suppress_print=False)

    # 3.4.3 Varying maximum depths

    # 80 / 20 random split
    train_indices = np.random.choice(range(X.shape[0]), 
                                    int(X.shape[0] * 0.8),
                                    replace=False)

    train_mask = np.zeros(X.shape[0], dtype=bool)
    train_mask[train_indices] = True

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_validation = X[~train_mask]
    y_validation = y[~train_mask]

    tree_accs = []
    depths = np.arange(1, 41)
    for max_depth in depths:
        classifier = DecisionTree(max_depth=max_depth, impurity_thresh=0.01)
        classifier.train(X_train, y_train, 0)
        y_validation_hat = classifier.predict(X_validation)
        acc = sum(y_validation==y_validation_hat) / len(y_validation_hat)
        tree_accs.append(acc)
    
    plt.plot(depths, tree_accs)

    # 3.4.4 Printing the Decision Tree
    classifier = DecisionTree(max_depth=3, impurity_thresh=0.01)
    classifier.train(X_train, y_train, 0)
    print(classifier)
        
