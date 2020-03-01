import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

prediction = model.predict(x_test)
class_names = [i for i in range(10)]

cols = 10
rows = 5
num_of_pics = cols * rows
fig, a = plt.subplots(rows, cols, sharey="all", sharex="all", figsize=(cols * 2, rows * 2))
index = 0
for row in range(rows):
    for col in range(cols):
        a[row][col].grid(False)
        a[row][col].imshow(x_test[index], cmap=plt.cm.binary)
        a[row][col].set_title("Predicted: " + str(class_names[int(np.argmax(prediction[index]))]))
        index += 1
plt.show()
