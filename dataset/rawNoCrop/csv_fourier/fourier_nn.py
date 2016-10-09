import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

f_train = open('train.txt', 'r')
f_train_labels = open('trainLabels.txt', 'r')
f_test = open('test.txt', 'r')
f_test_labels = open('testLabels.txt', 'r')

X_train = np.loadtxt(f_train)
Y_train = np.loadtxt(f_train_labels)
X_test = np.loadtxt(f_test)
Y_test = np.loadtxt(f_test_labels)

f_train.close()
f_train_labels.close()
f_test.close()
f_test_labels.close()

print X_train.shape
print Y_train.shape
print X_test.shape
print Y_test.shape

nb_classes = 2
class_weight = {0: 1., 1: .5}

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
input_dim = X_train.shape[1]

model = Sequential()

model.add(Dense(22, input_dim=input_dim, init='uniform'))
model.add(Activation('relu'))
model.add(Dense(1, init='uniform'))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Callbacks
models_folder = 'models/'
model_file_path = models_folder + 'descriptors-{val_acc:.4f}-{epoch:02d}-{val_loss:.4f}.h5'
checkpoint = ModelCheckpoint(model_file_path, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=0, mode='auto')

callbacks = [checkpoint, stopping]

# With class weights
# model.fit(X_train, Y_train, nb_epoch=60, batch_size=64, verbose=1, validation_data=(X_test, Y_test), callbacks=callbacks, class_weight=class_weight)

# Without class weights
model.fit(X_train, Y_train, nb_epoch=60, batch_size=64, verbose=1, validation_data=(X_test, Y_test), callbacks=callbacks)

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

print("\n")