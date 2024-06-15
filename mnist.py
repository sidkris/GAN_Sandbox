import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow.keras.datasets import mnist 

(trainX, trainy), (testX, testy) = mnist.load_data()
(trainX, trainy), (testX, testy) = mnist.load_data()

print(f"training data shapes : X = {trainX.shape}, y = {trainy.shape}")
print(f"testing data shapes : X = {testX.shape}, y = {testy.shape}")

# for k in range(9):
	
# 	plt.figure(figsize = (9, 9))
	
# 	for j in range(9):

# 		i = np.random.randint(0, 10000)
	
# 		plt.subplot(990 + 1 + j)

# 		plt.imshow(trainX[i], cmap = 'gray_r')

# 		plt.axis('off')

# 	plt.show()

train_data = trainX.astype(float)/255
test_data = testX.astype(float)/255

train_data = np.reshape(train_data, (60000, 28, 28, 1))
test_data = np.reshape(test_data, (10000, 28, 28, 1))

print(train_data.shape, test_data.shape)

input_data = tf.keras.layers.Input(shape = (28, 28, 1))

# ENCODER

encoder = tf.keras.layers.Conv2D(64, (5, 5), activation = 'relu')(input_data)

encoder = tf.keras.layers.MaxPooling2D((2, 2))(encoder)

encoder = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu')(encoder)

encoder = tf.keras.layers.MaxPooling2D((2, 2))(encoder)

encoder = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu')(encoder)

encoder = tf.keras.layers.MaxPooling2D((2, 2))(encoder)

encoder = tf.keras.layers.Flatten()(encoder)

encoder = tf.keras.layers.Dense(16)(encoder)

distribution_mean = tf.keras.layers.Dense(2, name = 'mean')(encoder)

distribution_variance = tf.keras.layers.Dense(2, name = 'log_variance')(encoder)

def sample_latent_features(distribution):

    distribution_mean, distribution_variance = distribution 

    batch_size = tf.shape(distribution_variance)[0]

    random = tf.keras.backend.random_normal(shape = (batch_size, tf.shape(distribution_variance)[1]))

    return distribution_mean + tf.exp(0.5 * distribution_variance) * random


sampled_latent_encoding = tf.keras.layers.Lambda(sample_latent_features)([distribution_mean, distribution_variance])

encoder_model = tf.keras.Model(input_data, sampled_latent_encoding)

print(encoder_model.summary())

# DECODER

decoder_input = tf.keras.layers.Input(shape = (2, ))

decoder = tf.keras.layers.Dense(64)(decoder_input)

decoder = tf.keras.layers.Reshape((1, 1, 64))(decoder)

decoder = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation = 'relu')(decoder)

decoder = tf.keras.layers.UpSampling2D((2, 2))(decoder)

decoder = tf.keras.layers.Conv2DTranspose(1, (5, 5), activation = 'relu')(decoder)

decoder = tf.keras.layers.UpSampling2D((2, 2))(decoder)

decoder_output = tf.keras.layers.Conv2DTranspose(1, (5, 5), activation = 'relu')(decoder)

decoder_model = tf.keras.Model(decoder_input, decoder_output)

print(decoder_model.summary())