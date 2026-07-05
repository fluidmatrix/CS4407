# Importing Necessary Libraries and Modules
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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

# Loading the Model with linear kernel
svm = SVC(kernel='linear')

# Fitting the Model on our Dataset
svm.fit(X_train, y_train)

# Making the Prediction
svm_prediction = svm.predict(X_test)

# Using a utility function to predict Results separately for every model
print_heatmap(y_test, svm_prediction)