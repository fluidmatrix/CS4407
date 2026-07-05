# Importing Necessary Libraries and Modules
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from utils import print_heatmap

# Loading the Iris Dataset
iris = load_iris()

# Assigning the Variables
X = iris.data 
y = iris.target 

# Splitting the dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Determining our Custom Rules based on petal length, petal width
def rule_based_classifier(X):
    predictions = []

    for sample in X:

        sepal_length, sepal_width, petal_length, petal_width = sample

        if petal_length < 2.4:
            predictions.append(0)

        elif petal_width < 1.85:
            predictions.append(1)

        else:
            predictions.append(2)

    return np.array(predictions)

# Making the predicion
rule_pred = rule_based_classifier(X_test)

# Plotting the graph along with classification report
print_heatmap(y_test=y_test, y_predict=rule_pred)