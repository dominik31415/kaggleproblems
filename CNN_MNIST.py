import pandas as pd
import numpy as np
np.random.seed(31415) 

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
#from keras.utils.visualize_util import plot

# input image dimensions
n_rows, n_cols = 28, 28

batch_size = 100 # Number of images used in each optimization step
n_classes = 10 # One class per digit
n_epoch = 20 # Number of times the whole data is used to learn

# Read the train and test datasets
train = pd.read_csv("train.csv").values
test  = pd.read_csv("test.csv").values

# Reshape the data to be used by a Theano CNN. Shape is
# (nb_of_samples, nb_of_color_channels, img_width, img_heigh)
X_train = train[:, 1:].reshape(train.shape[0], 1, n_rows, n_cols)
X_test = test.reshape(test.shape[0], 1, n_rows, n_cols)
y_train = train[:, 0] # extract labels

# normalize to 1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

# convert class vectors to one hot vectors
Y_train = np_utils.to_categorical(y_train, n_classes)

###########
model = Sequential()
#use relu, is faster than other activation functions
# generate  24 krnels for 5X5 filters, which are moved across the 1x28x28 image, 1 is the color axis
# use default stride = 1
# no zero padding = border vaild
model.add(Convolution2D(24, 5, 5, border_mode='valid',input_shape=(1, n_rows, n_cols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))	#reduced resolution with a 2x2 filter
model.add(Dropout(0.15))

# three layers of normal NN
model.add(Flatten()) # flatten: converts 3D array to 1D vector for oridnary NN
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dropout(0.15))

model.add(Dense(40))
model.add(Activation('relu'))

model.add(Dense(n_classes)) 
model.add(Activation('softmax')) 

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])


# learning
#model.load_weights('CNN_weights.txt')
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=n_epoch, verbose=1)
model.save_weights('CNN_weights.txt')

# Predict the label for X_test
yPred = model.predict_classes(X_test)

# Save prediction in file for Kaggle submission
np.savetxt('mnist-pred.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
#achieves >98% accuracy
