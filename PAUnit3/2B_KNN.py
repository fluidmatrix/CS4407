# Importing Necessary Libraries and Modules
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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


# Assigning the Model to our Variable
k_nn = KNeighborsClassifier(n_neighbors=5)

# Fitting the model to our dataset
k_nn.fit(X_train, y_train)

# Making the Prediction
k_nn_prediction = k_nn.predict(X_test)

# Printing and Comparing the Results
print_heatmap(y_test, k_nn_prediction)