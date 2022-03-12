import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# variables
rotation_range = 30
width_range = 0.25
height_range = 0.25
shear_range = 15
zoom_range = [0.5, 1.5]
batch_size = 32
model_name = "mymodel"

# load data
(X_training, Y_training), (X_test, Y_test) = mnist.load_data()

# reshape images to 28x28 px
X_training = X_training.reshape(X_training.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

Y_training = to_categorical(Y_training)
Y_test = to_categorical(Y_test)

X_training = X_training.astype("float32") / 255
X_test = X_test.astype("float32") / 255


# image transformations to make the training set more rich


datagen = ImageDataGenerator(
    rotation_range=rotation_range,
    width_shift_range=width_range,
    height_shift_range=height_range,
    zoom_range=zoom_range,
    shear_range=shear_range,
)

datagen.fit(X_training)


# the model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
training_datagen = datagen.flow(X_training, Y_training, batch_size=batch_size)


# training
epochs = 60
history = model.fit(
    training_datagen,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, Y_test),
    steps_per_epoch=int(np.ceil(60000 / float(batch_size))),
    validation_steps=int(np.ceil(10000 / float(batch_size))),
)


# exporting the model
model.save(f"{model_name}.h5")
