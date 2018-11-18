import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential

model = Sequential([
Dense(40, input_shape=(1,)),
Activation('relu'),
Dense(12),
Activation('sigmoid'),
Dense(1)
    ])

model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['mean_squared_error'])

x = np.arange(0, 1, 0.001).reshape(-1, 1)
y = np.square(x)

print("x {}".format(x[:10]))
print("y {}".format(y[:10]))


validation_x = np.arange(0, 1, 0.1).reshape(-1, 1)
validation_y = np.square(validation_x)

for i in range(20):
    model.fit(x, y, nb_epoch=25, batch_size=16)
    predictions = model.predict(x)
    print(np.mean(np.square(predictions - y)))
    predictions_validation = model.predict(validation_x)
    print(np.mean(np.square(predictions_validation - validation_y)))
    print(np.square(predictions_validation - validation_y)[:3])


model.save('my_model.h5')

test_data = np.arange(0, 1, 0.2).reshape(-1, 1)
print("test data {}".format(test_data))
prediction = model.predict(test_data)
print("Prediction: {}".format(prediction))