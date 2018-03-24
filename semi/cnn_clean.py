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
from keras.models import model_from_json

#Image parameters
w, h = 48, 48

#load datasets
data_lb = numpy.loadtxt("X_label.csv", delimiter = " ") #Labeled dataset
data_Y_lb = numpy.loadtxt("Y_label.csv", delimiter = " ") #Labeled dataset
data = numpy.loadtxt("fer2013_input_1.csv", delimiter = " ")
data_Y = numpy.loadtxt("fer2013_output_1.csv", delimiter = " ")

#Create input X
train_X = data_lb[0:15181,:] # Training only from labeled dataset
public_test_X = data[28710:32298,:]
private_test_X = data[32299:35887,:]
train_X = train_X.reshape((train_X.shape[0], w, h, 1))
public_test_X = public_test_X.reshape((public_test_X.shape[0], w, h, 1))
private_test_X = private_test_X.reshape((private_test_X.shape[0], w, h, 1))

datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, samplewise_center=False, samplewise_std_normalization=False, preprocessing_function=None)

datagen.fit(train_X)

#Create label Y
data_Y_new = np_utils.to_categorical(data_Y, num_classes=7)
data_Y_lb_new = np_utils.to_categorical(data_Y_lb, num_classes=7) 
train_Y = data_Y_lb_new[0:15181] # Training only from labeled dataset
public_test_Y = data_Y_new[28710:32298]
private_test_Y = data_Y_new[32299:35887]

#CNN model build
model = Sequential()

model.add(Conv2D(32, 3, 3, border_mode='valid', input_shape=(w, h, 1), kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), dim_ordering='tf'))

model.add(Conv2D(64, 3, 3))
model.add(BatchNormalization())
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(MaxPooling2D(pool_size=(5, 5)))

model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(Conv2D(128, 3, 3))
model.add(BatchNormalization())
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(Conv2D(128, 3, 3))
model.add(BatchNormalization())
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))

model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3)))

model.add(Flatten())
model.add(Dense(1024, kernel_regularizer=regularizers.l2(0.001)))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(Dropout(0.25))
model.add(Dense(1024, kernel_regularizer=regularizers.l2(0.001)))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(Dropout(0.25))


model.add(Dense(7))

model.add(Activation('softmax'))

#Compile the model
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

start = time.time()
#Fit the model
model.fit_generator(datagen.flow(train_X, train_Y, batch_size=256), epochs=10, validation_data = (public_test_X, public_test_Y))
stop = time.time()

# serialize model to JSON
model_json = model.to_json()
with open("model_ph1_cnn_cl_ep_10.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_ph1_cnn_cl_ep_10.h5")
print("Saved model to disk")

model.summary()
model.get_config()
