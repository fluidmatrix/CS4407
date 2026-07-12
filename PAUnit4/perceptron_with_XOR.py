from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense #type: ignore
import numpy as np

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([0,1,1,0])

model = Sequential()

model.add(Dense(4,
                input_dim=2,
                activation='relu'))

model.add(Dense(1,
                activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X,y,
          epochs=1000,
          verbose=0)

pred = model.predict(X)

print("Predictions")

for i in range(len(X)):
    print(X[i], round(float(pred[i])))