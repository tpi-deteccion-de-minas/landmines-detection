import numpy as np
import h5py
import pylab as pl

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

with h5py.File('train.h5', 'r') as hf:
	X_train = np.array(hf.get('train')).T.astype('float32')

with open('trainLabels.txt', 'r') as f:
	Y_train = np.loadtxt(f)

with h5py.File('test.h5', 'r') as hf:
	X_test = np.array(hf.get('test')).T.astype('float32')

with open('testLabels.txt', 'r') as f:
	Y_test = np.loadtxt(f)

print X_train.shape
print Y_train.shape
print X_test.shape
print Y_test.shape

# print X_train[0,:]
# pl.imshow(X_train[0,:].reshape((129, 48)), extent=[0, 1, 0, 1])

batch_size = 128
nb_epoch = 12

img_rows, img_cols = 129, 48
nb_filters = 32
pool_size = (4, 4)
kernel_size = (7, 7)

if K.image_dim_ordering() == 'th':
  X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
  X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
  input_shape = (1, img_rows, img_cols)
else:
  X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
  X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)


model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.3))

model.add(Convolution2D(nb_filters, 5, 5))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

pl.show()