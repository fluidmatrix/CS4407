import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential # type:ignore
from tensorflow.keras.layers import Dense # type:ignore

from utils import print_heatmap

df = pd.read_csv("customer_churn.csv")

df.head()

df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],
                                   errors="coerce")

df["TotalCharges"].fillna(df["TotalCharges"].median(),
                          inplace=True)

encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = encoder.fit_transform(df[col])
        
X = df.drop("Churn", axis=1)

y = df["Churn"]

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = Sequential()

model.add(Dense(32,
                input_dim=X_train.shape[1],
                activation="relu"))

model.add(Dense(16,
                activation="relu"))

model.add(Dense(1,
                activation="sigmoid"))


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

pred = model.predict(X_test)

pred = (pred > 0.5).astype(int)

acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)

cm = confusion_matrix(y_test, pred)

print_heatmap(y_test, pred)

print(cm)

print(classification_report(y_test,pred))

