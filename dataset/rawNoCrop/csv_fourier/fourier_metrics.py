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

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
input_dim = X_train.shape[1]

model = Sequential()

model.add(Dense(22, input_dim=input_dim, init='uniform'))
model.add(Activation('relu'))
model.add(Dense(1, init='uniform'))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.load_weights("models/descriptors-0.7616-19-0.5464.h5")
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