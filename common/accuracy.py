import gzip
import numpy as np
import pickle
import os
import gzip
import utils
from common.sorn import Sorn

data = utils.Bunch()
filepath = os.path.abspath(os.path.join(os.getcwd(),".."))

# ------------ Caculating the Performance of Reservior ------------
# Read the training sets.
filename = os.path.join(filepath,'training.pickle')
with gzip.open(filename,'rb') as f:
    temp = pickle.load(f)
    data.X_train = temp['R_x']
    # data.X_train = temp['X']
    data.y_train = temp['C']

# Read the testing sets.
filename = os.path.join(filepath,'testing.pickle')
with gzip.open(filename,'rb') as f:
    temp = pickle.load(f)
    data.X_test = temp['R_x']
    # data.X_test = temp['X']
    data.y_test = temp['C']

# Restore the outputs.
y_read_train = np.zeros((6,len(data.y_train)+1))
y_read_test = np.zeros((6,len(data.y_test)+1))
for i,y in enumerate(data.y_train):
    y_read_train[y,i] = 1
for i,y in enumerate(data.y_test):
    y_read_test[y,i] = 1
y_read_train = y_read_train[:,:5000]
y_read_test = y_read_test[:,:5000]
target = np.argmax(y_read_test,axis=0)

# States variables
X_train = (data.X_train>=0)+0.
X_test = (data.X_test>=0)+0.

# Typical Calculation of Reservior Computing Readout
X_train_pinv = np.linalg.pinv(X_train)
W_trained = np.dot(y_read_train,X_train_pinv.T)

y_predicted = np.dot(W_trained,X_test.T)
prediction = np.argmax(y_predicted, axis=0)
# perf_all = np.sum((prediction == target)) /float(len(data.y_test))
except_first = np.where((target!=0)& (target!=3))[0]
# Reduced performance(i.e. reduce the random initial char of word)
y_test_red = target[except_first]
y_pred_red = prediction[except_first]
perf_red = (y_test_red == y_pred_red).sum() / float(len(y_pred_red))

print("The testing accuracy of Counting Task is: %0.2f%%" % (perf_red*100))

