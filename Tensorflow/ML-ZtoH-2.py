# https://colab.research.google.com/github/lmoroney/mlday-tokyo/blob/master/Lab2-Computer-Vision.ipynb#scrollTo=PmxkHFpt31bM

import tensorflow as tf
import pandas as pd
import seaborn as sns
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# plt.imshow(training_images[0])
# print(training_labels[0])
# print(training_images[0])
# plt.show()


training_images  = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
dataframe = pd.DataFrame(training_labels)
dataframe.columns =['Count']
print(dataframe.head())
sns.countplot(x="Count", data=dataframe)
plt.show()

model.save('ML-ZtoH-2.h5')
print(model.evaluate(test_images, test_labels))
# classifications = model.predict(test_images[0])
# print(classifications[0])
# print(test_labels[0])

