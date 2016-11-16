import numpy as np
import h5py

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

img_rows, img_cols = 48, 48
nb_filters = 28

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
model_file_path = models_folder + 'spec-cnn-{val_acc:.4f}-{epoch:02d}-{val_loss:.4f}.h5'
checkpoint = ModelCheckpoint(model_file_path, monitor='val_loss', verbose=0, save_best_only=False, mode='min')

callbacks = [checkpoint]

model = Sequential()

model.add(Convolution2D(nb_filters, 7, 7,
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.5))

model.add(Convolution2D(nb_filters, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(5, 5)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, init='glorot_uniform'))
model.add(Activation('sigmoid'))

adam = Adam(lr=0.00002)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.load_weights("models/spec-cnn-0.8759-54-0.3281.h5")
score_train = model.evaluate(X_train, Y_train, verbose=0)
score_test = model.evaluate(X_test, Y_test, verbose=0)

print ""
print('Train score:', score_train[0], 'Train accuracy:', score_train[1])
print('Test score:', score_test[0], 'Test accuracy:', score_test[1])


# Pretty printing the confusion matrix with Pandas
predictions = model.predict_classes(X_test, verbose=0).squeeze()
from pandas import Series
from pandas import crosstab

test_labels = Series(Y_test, name='Truth')
predictions = Series(predictions, name='Predicted')
conf_matrix = crosstab(test_labels, predictions, rownames=['Truth'], colnames=['Predicted'], margins=True)
print "\nConfusion matrix\n\n 0: No mine, 1: Mine\n\n", conf_matrix

positive_truths = conf_matrix[1][1] + conf_matrix[1][0]
negative_truths = conf_matrix[1][1] + conf_matrix[0][1]
precision = 1. * conf_matrix[1][1]/positive_truths
recall = 1. * conf_matrix[1][1]/negative_truths
f1_score = 2. *((precision*recall)/(precision+recall))

print("Precision: %.4f" % (1. * conf_matrix[1][1]/positive_truths))
print("Recall: %.4f" % (1. * conf_matrix[1][1]/negative_truths))
print("F1 Score: %.4f" % (f1_score))

print("\n")

from sklearn.metrics import roc_curve, auc
import pylab as plt

scores = model.predict(X_test, verbose=0).squeeze()
fpr, tpr, thresholds = roc_curve(Y_test, scores, pos_label=1.)
roc_auc = auc(fpr,tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()