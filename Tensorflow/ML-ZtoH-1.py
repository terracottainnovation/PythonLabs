# https://codelabs.developers.google.com/codelabs/tensorflow-lab1-helloworld/#2

import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt



xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 16.0, 37.0, 10.0, 13.0], dtype=float)


model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')
# plt.plot(xs,ys)
# plt.show()
model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))
