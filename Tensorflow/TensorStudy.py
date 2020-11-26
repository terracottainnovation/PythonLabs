# https://towardsdatascience.com/mastering-tensorflow-tensors-in-5-easy-steps-35f21998bb86
import tensorflow as tf
import numpy as np

# You can create Tensor objects with the `tf.constant` function:
x = tf.constant( [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11] ])
# You can create Tensor objects only consisting of 1s with the `tf.ones` function:
y = tf.ones((1,5))
# You can create Tensor objects only consisting of 0s with the `tf.zeros` function:
z = tf.zeros((1,5) , dtype=tf.int32)
# You can use the `tf.range()` function to create Tensor objects:
q = tf.range(start=1, limit=6, delta=1, name="series", dtype=tf.int32)

print(x,"   No. of dimensions  for X", x.ndim)
print(y)
print(z)
print(q)
# A few rules about indexing:
#
#     Indices start at zero (0).
#     Negative index (“-n”) value means backward counting from the end.
#     Colons (“:”) are used for slicing: start:stop:step.
#     Commas (“,”) are used to reach deeper levels.