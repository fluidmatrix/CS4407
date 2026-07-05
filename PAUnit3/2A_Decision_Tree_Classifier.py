# # Importing Necessary Libraries and Modules
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utils import print_heatmap

# # Loading the Iris Dataset
iris = load_iris()

# # Assigning the Variables
X = iris.data 
y = iris.target

# # Splitting the dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# # Loading the Decision Tree Classifier
DecisionTree = DecisionTreeClassifier(random_state=42)

# # Fitting the Model on our Dataset
DecisionTree.fit(X_train, y_train)

# # Making the Prediction
DecisionTree_Prediction = DecisionTree.predict(X_test)

# # Printing the Metrics
print_heatmap( y_test=y_test, y_predict=DecisionTree_Prediction)

