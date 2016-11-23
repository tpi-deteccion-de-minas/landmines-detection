import numpy as np
import h5py

with h5py.File('descriptors.h5', 'r') as hf:
	X = np.array(hf.get('data')).T.astype('float32')
	Y = np.array(hf.get('labels')).T.astype('uint8')
	objectIds = np.array(hf.get('objectIds')).T.astype('uint8')

print X.shape, Y.shape, objectIds.shape
print np.max(objectIds), np.min(objectIds)