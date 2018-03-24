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
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras import optimizers
from keras.utils import np_utils
from keras import regularizers
from keras.regularizers import l2
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.models import Model

# For decoder
from keras.layers.convolutional import Conv2DTranspose

# For dense layer output into the decoder and the comb (reconstruction) function
from keras.layers import Lambda
from keras.layers import merge

# For mse and cat_entr
from keras import losses
from keras.losses import mean_squared_error
from keras.losses import categorical_crossentropy

# For callbacks - dynamic weight changing
from keras import callbacks
from keras.callbacks import Callback

#Image parameters
w, h = 48, 48
input_img = Input(shape=(w, h, 1))

#load datasets
data_lb = numpy.loadtxt("X_label_10.csv", delimiter = " ") #Labeled dataset
data_Y_lb = numpy.loadtxt("Y_label_10.csv", delimiter = " ") #Labeled dataset
data = numpy.loadtxt("fer2013_input_1.csv", delimiter = " ")
data_Y = numpy.loadtxt("fer2013_output_1.csv", delimiter = " ")

#Create input X
train_X = data[0:28709,:]
public_test_X = data[28710:32298,:]
private_test_X = data[32299:35887,:]
train_X = train_X.reshape((train_X.shape[0], w, h, 1))
public_test_X = public_test_X.reshape((public_test_X.shape[0], w, h, 1))
private_test_X = private_test_X.reshape((private_test_X.shape[0], w, h, 1))

#Create label Y
data_Y_new = np_utils.to_categorical(data_Y, num_classes=7)
train_Y = data_Y_new[0:28709]
public_test_Y = data_Y_new[28710:32298]
private_test_Y = data_Y_new[32299:35887]

# Load the pre-trained models as the encoder
# Clean encoder model
json_file = open('model_cl_lrelu_bt_1cnn_ep_15.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
cl_enc = model_from_json(loaded_model_json)
cl_enc.load_weights("model_cl_lrelu_bt_1cnn_ep_15.h5")
print("Loaded clean enocder model from disk")

# Noisy encoder model
json_file = open('model_ns_lrelu_bt_1cnn_ns_cnn_ep_15.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
ns_enc = model_from_json(loaded_model_json)
ns_enc.load_weights("model_ns_lrelu_bt_1cnn_ns_cnn_ep_15.h5")
print("Loaded noisy encoder model from disk")

# Denoising / Combinator function - This function as of now is directly from the paper of the Ladder Network

def comb(z,u):
	b_0, b_1 = 0, 0
	w_0_z, w_1_z = 1, 1
	w_0_u, w_1_u = 0, 0
	w_0_z_u, w_1_z_u = 0, 0
	w_sig = 1
	pr_0_z = Lambda(lambda x: x * w_0_z)(z)
	pr_1_z = Lambda(lambda x: x * w_1_z)(z)
	pr_0_u = Lambda(lambda x: x * w_0_u)(u)
	pr_1_u = Lambda(lambda x: x * w_1_u)(u)
	pr_z_u = merge([z,u], mode="mul")
	pr_0_z_u = Lambda(lambda x: x * w_0_z_u)(pr_z_u)
	pr_1_z_u = Lambda(lambda x: x * w_1_z_u)(pr_z_u)
	sig = merge([pr_1_z, pr_1_u, pr_1_z_u], mode = "sum")
	sig_moid = Lambda(tf.sigmoid)(sig)
	pr_sig_moid = Lambda(lambda x: x * w_sig)(sig_moid)
	z_cap = merge([pr_0_z, pr_0_u, pr_0_z_u, pr_sig_moid], mode = "sum")
	return z_cap;	

# Loss weights - Don't use if calculating loss by model.add_loss() method 

alpha = K.variable(1.0)
beta = K.variable(1.0)
bt_size = K.variable(256.0)
lb_data = K.variable(2816.0) # 10% labeled data

class upd_wt(Callback):
    def __init__(self, alpha, beta, lb_data, bt_size):
        self.alpha = alpha
        self.beta = beta
        self.lb_data = lb_data
        self.bt_size = bt_size
	# weight update
    def on_epoch_begin(self, epoch, logs = {}):
        count = 0
    def on_batch_begin(self, batch, logs={}):
        count = count + 1
        if(self.batch > count): # unsupervised learning only
            self.alpha = K.variable(0.) # Supervised learning loss function weight = 0.0
            self.beta = self.beta

# Calculating the loss if going by model.add_loss() method
cost = K.variable(0.)
def mse(y_true,y_pred):
	cost = cost + K.mean(K.square(y_pred - y_true), axis=-1) 

# Autoencoder (or) Denoising autoencoder with the comb denoising function

# Decoder begins here with the upsampling -- connecting to the last convolutional layer output of the noisy encoder
# NOTE: There is no scaling or shifting in the decoder implementation according to the paper. There is only batch normalization

# Getting the output from the final dense layer (dense_3) of the noisy encoder and passing it to the decoder module

lam = Lambda(lambda x: x, output_shape=(ns_enc.get_layer('dense_3').output_shape[1], w, h, 1))
lam.build((ns_enc.get_layer('dense_3').output_shape[1], w, h, 1))
x = K.reshape(ns_enc.get_layer('dense_3').output,(ns_enc.get_layer('dense_3').output_shape[1], w, h, 1))
out = lam(x)

cde1 = Conv2DTranspose(32, 3, 3, name = "conv_de_1")(out)
btde1 = BatchNormalization(center=False, scale=False, name = "btde1")(cde1)
a1 = LeakyReLU(name = "actde1")(btde1)
#denoising function comb call if denoising autoencoder
cde2 = Conv2DTranspose(64, 3, 3, name = "conv_de_2")(a1)
btde2 = BatchNormalization(center=False, scale=False, name = "btde2")(cde2)
a2 = LeakyReLU(name = "actde2")(btde2)
#denoising function comb call if denoising autoencoder
cde3 = Conv2DTranspose(64, 3, 3, name = "conv_de_3")(a2)
btde3 = BatchNormalization(center=False, scale=False, name = "btde3")(cde3)
a3 = LeakyReLU(name = "actde3")(btde3)
#denoising function comb call if denoising autoencoder
cde4 = Conv2DTranspose(32, 3, 3, name = "conv_de_4")(a3)
btde4 = BatchNormalization(center=False, scale=False, name = "btde4")(cde4)
a4 = LeakyReLU(name = "actde4")(btde4)

# Intermediate models for getting the output from the deconvolutional layer of the autoencoder
ae1 = Model(ns_enc.input,cde1)
ae2 = Model(ns_enc.input,cde2)
ae3 = Model(ns_enc.input,cde3)
ae4 = Model(ns_enc.input,cde4)

# Final model - Autoencoder
ae = Model(ns_enc.input, outputs = [ns_enc.get_layer('activation_1').output, ae1.get_layer('conv_de_1').output]) # As of now doing for only one layer comparison
ae.summary()

# Clean encoder "activation" array access for comparison with decoder layer
intermediate_layer_model = Model(inputs=cl_enc.input, outputs=cl_enc.get_layer('conv2d_1').output)
int_op_cl_1 = intermediate_layer_model.predict(train_X)
# or
#int_op_cl_1 = numpy.loadtxt('int_op_cl_1.csv') # Load from the pre-trained model

#Calculating loss function by model.add_loss() concept
#mse(cl_enc.get_layer('conv2d_1').output, ae1.get_layer('conv_de_1').output) # As of now doing for only one layer comparison
#ae.add_loss(cost)
#ae.compile(optimizer='adam', metrics=['accuracy'])

ae.compile(optimizer='adam', loss=['categorical_crossentropy','mean_squared_error'], metrics=['accuracy'], loss_weights=[alpha,beta])
ae.fit(train_X, [train_Y, int_op_cl_1], batch_size=256, epochs=10, callbacks=[upd_wt(alpha, beta, lb_data, bt_size)])

print("Done!")
