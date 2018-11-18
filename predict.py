from keras.models import load_model
import numpy as np

model = load_model('my_model.h5')

test_data = np.arange(0, 1, 0.2).reshape(-1, 1)
print("test data {}".format(test_data))
prediction = model.predict(test_data)
print("Prediction: {}".format(prediction))
