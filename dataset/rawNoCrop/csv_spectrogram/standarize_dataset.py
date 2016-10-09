import numpy as np
import h5py

from sklearn.preprocessing import MinMaxScaler

isStandardization = True

if isStandardization:
	with h5py.File('train.h5', 'r+') as hf:
		data = hf.get('train')
		X_train = np.array(data).T.astype('float32')
		train_mean = np.mean(X_train)
		train_std = np.std(X_train)
		X_train = (X_train - train_mean) / train_std
		hf['train'][:] = X_train.T
		print X_train[0,:]
	del X_train
	del data
	with h5py.File('test.h5', 'r+') as hf:
		data = hf.get('test')
		X_test = np.array(data).T.astype('float32')
		X_test = (X_test - train_mean) / train_std
		hf['test'][:] = X_test.T
		print X_test[0,:]
	del X_test
	del data
else: # Normalization (0,1)
	min_max_scaler = MinMaxScaler()
	with h5py.File('train.h5', 'r+') as hf:
		data = hf.get('train')
		X_train = np.array(data).T.astype('float32')
		hf['train'][:] = min_max_scaler.fit_transform(X_train).T
		print X_train[0,:]
	del X_train
	del data
	with h5py.File('test.h5', 'r+') as hf:
		data = hf.get('test')
		hf['test'][:] = min_max_scaler.fit_transform(X_test).T
		print X_test[0,:]
	del X_test
	del data