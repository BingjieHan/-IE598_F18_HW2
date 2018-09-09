import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

#training set & test set

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.3, random_state=1, stratify=y)

# Building a decision tree
tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=4, 
                              random_state=1)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, 
                      y_combined, 
                      clf=tree, 
                      filler_feature_ranges=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()

from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

dot_data = export_graphviz(tree,
                            filled=True, 
                            rounded=True,
                            class_names=['Setosa', 
                                         'Versicolor',
                                         'Virginica'],
                            feature_names=['petal length', 
                                           'petal width'],
                            out_file=None) 
graph = graph_from_dot_data(dot_data) 
graph.write_png('tree.png')

print("My name is Bingjie Han ")
print("My NetID is: bingjie5 ")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

