import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Dataset
data = {
    "Advertising_Spend": [2000, 3000, 2500, 4000, 3500, None, 5000, 4500, 3000, 3800],
    "Store_Size": [1500, 2000, 1800, 2200, None, 2100, 2500, 2400, 2000, 2300],
    "Customers": [200, 250, 230, 300, 280, 260, 320, 310, 270, None],
    "Promotion": ["Yes", "No", "Yes", "Yes", "No", "No", "Yes", "Yes", "No", "Yes"],
    "Sales": [40000, 50000, 45000, 60000, 52000, 48000, 65000, 63000, 51000, 59000]
}

df = pd.DataFrame(data)

# Preprocessing
imputer = SimpleImputer(strategy='mean')
df[['Advertising_Spend','Store_Size','Customers']] = \
imputer.fit_transform(df[['Advertising_Spend','Store_Size','Customers']])

df['Promotion'] = df['Promotion'].map({'Yes':1,'No':0})

X = df[['Advertising_Spend','Store_Size','Customers','Promotion']]
y = df['Sales']

# Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

models = {
    "Multiple Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0)
}

# Polynomial
poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly,y_train)

poly_pred = poly_model.predict(X_test_poly)

print("Polynomial Regression")
print("RMSE:", mean_squared_error(y_test,poly_pred)**0.5)
print("MAE :", mean_absolute_error(y_test,poly_pred))
print("R²  :", r2_score(y_test,poly_pred))
print()

for name, model in models.items():

    model.fit(X_train,y_train)
    pred = model.predict(X_test)

    rmse = mean_squared_error(y_test,pred)**0.5
    mae = mean_absolute_error(y_test,pred)
    r2 = r2_score(y_test,pred)

    print(name)
    print("RMSE:", rmse)
    print("MAE :", mae)
    print("R²  :", r2)
    print()