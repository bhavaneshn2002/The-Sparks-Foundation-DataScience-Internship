import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_squared_error, r2_score


x = file.Hours
y = file.Scores
#Train test split
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size= 0.25)



model = LinearRegression()
model.fit(X_train.values.reshape((len(X_train), 1)), y_train)
pred = model.predict(X_test.values.reshape((len(X_test), 1)))
#Score and error
print(mean_squared_error(pred, y_test))
r2 = r2_score(y_test, pred)
print("R2 score:", r2)
#Plot line
plt.scatter(x, y, color='b')
plt.plot(X_test, pred, color='r')
plt.show()
