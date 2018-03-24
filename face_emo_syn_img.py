import numpy
import tensorflow as tf
import random as rn
import time

# fix random seed for reproducibility
numpy.random.seed(2016)
rn.seed(2015)
tf.set_random_seed(2014)

import keras

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras import optimizers
from keras.utils import np_utils
from keras import regularizers
from keras.regularizers import l2
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

#Image parameters
w, h = 48, 48


#load datasets
data = numpy.loadtxt("fer2013_input_1.csv", delimiter = " ")
data_Y = numpy.loadtxt("fer2013_output_1.csv", delimiter = " ")

#Create input X
train_X = data[0:28709,:]
public_test_X = data[28710:32298,:]
private_test_X = data[32299:35887,:]
train_X = train_X.reshape((train_X.shape[0], w, h, 1))
public_test_X = public_test_X.reshape((public_test_X.shape[0], w, h, 1))
private_test_X = private_test_X.reshape((private_test_X.shape[0], w, h, 1))

#Synthesizing and augmenting the training images
datagen = ImageDataGenerator(rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
        
datagen.fit(train_X)

#Create label Y
data_Y_new = np_utils.to_categorical(data_Y, num_classes=7)
train_Y = data_Y_new[0:28709]
public_test_Y = data_Y_new[28710:32298]
private_test_Y = data_Y_new[32299:35887]

#CNN model build
model = Sequential()

model.add(Conv2D(32, 3, 3, border_mode='valid', input_shape=(w, h, 1), kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), dim_ordering='tf'))

model.add(Conv2D(64, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(MaxPooling2D(pool_size=(5, 5)))

model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(Conv2D(128, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(Conv2D(128, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))

model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3)))

model.add(Flatten())
model.add(Dense(1024, kernel_regularizer=regularizers.l2(0.001)))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.GaussianDropout(0.01))
model.add(Dense(1024, kernel_regularizer=regularizers.l2(0.001)))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.GaussianDropout(0.01))


model.add(Dense(7))

model.add(Activation('softmax'))

#Compile the model
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

start = time.time()
#Fit the model
model.fit_generator(datagen.flow(train_X, train_Y, batch_size=256), epochs=40, validation_data = (public_test_X, public_test_Y))
stop = time.time()

#Evaluate the model
scores_pub = model.evaluate(public_test_X, public_test_Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores_pub[1]*100))
scores_pri = model.evaluate(private_test_X, private_test_Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores_pri[1]*100))
print("\nTraining time: %.2f%%" % (stop-start))
