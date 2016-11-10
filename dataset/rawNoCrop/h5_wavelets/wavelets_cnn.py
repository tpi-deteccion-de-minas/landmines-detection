import numpy as np
import h5py
import pylab as pl

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
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

batch_size = 64
nb_epoch = 60

img_rows, img_cols = 32, 32
nb_filters = 30

if K.image_dim_ordering() == 'th':
  X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
  X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
  input_shape = (1, img_rows, img_cols)
else:
  X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
  X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)

# Callbacks
models_folder = 'models/'
model_file_path = models_folder + 'wav-cnn-{val_acc:.4f}-{epoch:02d}-{val_loss:.4f}.h5'
checkpoint = ModelCheckpoint(model_file_path, monitor='val_loss', verbose=0, save_best_only=False, mode='min')

callbacks = [checkpoint]

model = Sequential()

model.add(Convolution2D(nb_filters, 5, 5,
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(nb_filters, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, init='glorot_uniform'))
model.add(Activation('sigmoid'))

adam = Adam(lr=0.0002)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test), callbacks=callbacks)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Pretty printing the confusion matrix with Pandas
predictions = model.predict_classes(X_test, verbose=0).squeeze()
from pandas import Series
from pandas import crosstab

test_labels = Series(Y_test, name='Truth')
predictions = Series(predictions, name='Predicted')
conf_matrix = crosstab(test_labels, predictions, rownames=['Truth'], colnames=['Predicted'], margins=True)
print "\nConfusion matrix\n\n 0: No mine, 1: Mine\n\n", conf_matrix

norm_conf_matrix = crosstab(test_labels, predictions, rownames=['Truth'], colnames=['Predicted'], margins=True, normalize=True)
print "\nConfusion matrix (normalised)\n\n", norm_conf_matrix

print("\n\nOverall accuracy: %.4f (%i out of %i samples)"
      % (norm_conf_matrix[0][0] + norm_conf_matrix[1][1], conf_matrix[0][0] + conf_matrix[1][1], len(test_labels)))
print("False positive rate: %.4f (%i samples)" % (norm_conf_matrix[1][0], conf_matrix[1][0]))
print("False negative rate: %.4f (%i samples)" % (norm_conf_matrix[0][1], conf_matrix[0][1]))

positive_truths = conf_matrix[1][1] + conf_matrix[1][0]
negative_truths = conf_matrix[1][1] + conf_matrix[0][1]
print("Precision: %.4f" % (1. * conf_matrix[1][1]/positive_truths))
print("Recall: %.4f" % (1. * conf_matrix[1][1]/negative_truths))

pl.show()