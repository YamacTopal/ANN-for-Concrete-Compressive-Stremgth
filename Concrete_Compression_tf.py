from ucimlrepo import fetch_ucirepo 
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error

# fetch dataset 
concrete_compressive_strength = fetch_ucirepo(id=165) 
  
# data (as pandas dataframes) 
X = concrete_compressive_strength.data.features.values
y = concrete_compressive_strength.data.targets.values
  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

  
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()

"""### Adding the input layer and the first hidden layer"""

ann.add(tf.keras.layers.Dense(units=16, activation='relu'))

"""### Adding the second hidden layer"""

ann.add(tf.keras.layers.Dense(units=32, activation='relu'))

ann.add(tf.keras.layers.Dense(units=64, activation='relu'))

"""### Adding the output layer"""

ann.add(tf.keras.layers.Dense(units=1))

"""## Part 3 - Training the ANN

### Compiling the ANN
"""

ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

"""### Training the ANN model on the Training set"""

ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)