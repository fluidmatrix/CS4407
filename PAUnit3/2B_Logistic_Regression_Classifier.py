# Importing Necessary Libraries and Modules
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
l_regression = LogisticRegression()

# Fitting the model to our dataset
l_regression.fit(X_train, y_train)

# Making the Prediction
l_prediction = l_regression.predict(X_test)

print_heatmap(y_test=y_test, y_predict=l_prediction)