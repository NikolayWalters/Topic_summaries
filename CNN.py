"""
Notes from Kaggle's Computer Vision course
"""

# Keras contains a bunch of pre-trained models useful for transfer learning
pretrained_base = tf.keras.models.load_model(
    '../input/cv-course-models/cv-course-models/vgg16-pretrained-base',
)
pretrained_base.trainable = False

# Attach classification head
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    pretrained_base,
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

# compile and run
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=30,
    verbose=0,
)

# plot loss and accuracy
import pandas as pd

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot();

# conv layer
model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=3), # activation is None
    # More layers follow
])

# with relu
model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=3, activation='relu')
    # More layers follow
])

# edge detection kernel
kernel = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
])

# sharpen
kernel = tf.constant([
    [0, -1,  0],
    [-1, 5, -1],
    [0, -1,  0],
])

# emboss
kernel = tf.constant([
    [-2, -1, 0],
    [-1,  1, 1],
    [ 0,  1, 2],
])

# e.g.
image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel,
    strides=1,
    padding='SAME',
)

image_detect = tf.nn.relu(image_filter) #pass to relu

# max pooling layer
model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=3), # activation is None
    layers.MaxPool2D(pool_size=2),
    # More layers follow
])

# or by using as separate layer
image_condense = tf.nn.pool(
    input=image_detect, # image in the Detect step above
    window_shape=(2, 2),
    pooling_type='MAX',
    strides=(2, 2),
    padding='SAME',
)

# strides parameter = how far the window should move at each step
# padding parameter = how we handle the pixels at the edges of the input

# padding='valid' = convolution window will stay entirely inside the input
# padding='same' = pad with zeros