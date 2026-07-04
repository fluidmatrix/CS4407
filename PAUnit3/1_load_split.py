# Imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

"""
1  Loading the Dataset 
"""
# Loading the Iris Dataset
iris = load_iris()

# assigning the variabels
X = iris.data 
y = iris.target
"""
2 Describing the Characterstics
"""
# Printing the desired metrics 
print("Number of Samples:", X.shape[0])
print("Number of Features:", X.shape[1])

# Optional: distribution Density
unique, counts = np.unique(y, return_counts=True)

print("\nClass Distribution")
for label, count in zip(unique, counts):
    print(iris.target_names[label], ":", count)

""" 
3 Split the dataset into Test and Train 
"""
# Using our Import to split the datset into 80/20 for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrainig Samples: ", len(X_train))
print("\nTesting Samples: ", len(X_test))