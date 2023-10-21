import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


file = pd.read_csv("Iris.csv")
file= file.drop(['Id'], axis = 1)
file.head()


#checking for Null values
file.isnull().sum()
SepalLengthCm    0
SepalWidthCm     0
PetalLengthCm    0
PetalWidthCm     0
Species          0
dtype: int64


#Label Encoding - for encoding categorical features into numerical ones
encoder = LabelEncoder()
file['Species'] = encoder.fit_transform(file['Species'])
file.head()
#Label Encoding - for encoding categorical features into numerical ones
encoder = LabelEncoder()
file['Species'] = encoder.fit_transform(file['Species'])
file.head()


SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm	Species
0	5.1	3.5	1.4	0.2	0
1	4.9	3.0	1.4	0.2	0
2	4.7	3.2	1.3	0.2	0
3	4.6	3.1	1.5	0.2	0
4	5.0	3.6	1.4	0.2	0
ax = file[file.Species==0].plot.scatter(x='SepalLengthCm', y='SepalWidthCm', 
                                                    color='red', label='Iris - Setosa')
file[file.Species==1].plot.scatter(x='SepalLengthCm', y='SepalWidthCm', 
                                                color='green', label='Iris - Versicolor', ax=ax)
file[file.Species==2].plot.scatter(x='SepalLengthCm', y='SepalWidthCm', 
                                                color='blue', label='Iris - Virginica', ax=ax)


ax.set_title("Scatter Plot")
Text(0.5, 1.0, 'Scatter Plot')



model = KMeans(n_clusters=3)
model.fit(X_train, y_train)

pred = model.predict(X_test)


