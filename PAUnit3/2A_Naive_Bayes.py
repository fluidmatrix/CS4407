# Importing Necessary Libraries and Modules
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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

# Assiging our Model 
naive_bayes = GaussianNB()

# Fitting the model on the training sets
naive_bayes.fit(X_train, y_train)

# Making the Prediction
naive_bayes_predicted = naive_bayes.predict(X_test)

# Printing the heatmap and classification Report
print_heatmap(y_test, naive_bayes_predicted)

