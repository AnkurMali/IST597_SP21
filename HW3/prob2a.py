from __future__ import print_function
import os 
import numpy as np
import tensorflow as tf
import pandas as pd  
import matplotlib.pyplot as plt  
import sys
#You have freedom of using eager execution in tensorflow
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

'''

IST 597: Foundations of Deep Learning

Problem 2a: 1-Layer MLP for IRIS

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
def computeCost(X,y,theta,reg):
	# WRITEME: write your code here to complete the routine
	return 0.0		
			
def computeGrad(X,y,theta,reg): # returns nabla
	W = theta[0]
	b = theta[1]
	W2 = theta[2]
	b2 = theta[3]
	# WRITEME: write your code here to complete the routine
	dW = W * 0.0
	db = b * 0.0
	dW2 = W2 * 0.0
	db2 = b2 * 0.0
	return (dW,db,dW2,db2)

def predict(X,theta):
	# WRITEME: write your code here to complete the routine
	scores = 0.0
	probs = 0.0
	
def create_mini_batch(X, y, start, end):
	# WRITEME: write your code here to complete the routine
	mb_x = None
	mb_y = None
	return (mb_x, mb_y)
		
"""def shuffle(X,y):
	ii = np.arange(X.shape[0])
	ii = np.random.shuffle(ii)
	X_rand = X[ii]
	y_rand = y[ii]
	X_rand = X_rand.reshape(X_rand.shape[1:])
	y_rand = y_rand.reshape(y_rand.shape[1:])
	return (X_rand,y_rand)"""
	
np.random.seed(0)
# Load in the data from disk
path = os.getcwd() + '/data/iris_train.dat'  
data = pd.read_csv(path, header=None) 

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)
y = y.flatten()
X_tf = tf.constant(X)
Y_tf = tf.constant(y)
# load in validation-set
path = os.getcwd() + '/data/iris_test.dat'
data = pd.read_csv(path, header=None) 
cols = data.shape[1]  
X_v = data.iloc[:,0:cols-1]  
y_v = data.iloc[:,cols-1:cols] 

X_v = np.array(X_v.values)  
y_v = np.array(y_v.values)
y_v = y_v.flatten()

X_V_tf = tf.constant(X_v)
Y_V_tf = tf.constant(y_v)


# initialize parameters randomly
D = X.shape[1]
K = np.amax(y) + 1

# initialize parameters randomly
h = 100 # size of hidden layer
initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
W = tf.Variable(initializer([D, h]))
b = tf.Variable(tf.random_normal([h]))
W2 = tf.Variable(initializer([h, K]))
b2 = tf.Variable(tf.random_normal([K]))
theta = (W,b,W2,b2)

# some hyperparameters
n_e = 100
n_b = 10
step_size = 0.01 #1e-0
reg = 0 #1e-3 # regularization strength

train_cost = []
valid_cost = []
# gradient descent loop
num_examples = X.shape[0]
for i in xrange(n_e):
	X, y = tf.random_shuffle([X_tf,Y_tf]) # re-shuffle the data at epoch start to avoid correlations across mini-batches
	# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
	#          you can use the "check" variable to decide when to calculate losses and record/print to screen (as in previous sub-problems)

	# WRITEME: write the inner training loop here (1 full pass, but via mini-batches instead of using the full batch to estimate the gradient)
	s = 0
	while (s < num_examples):
		# build mini-batch of samples
		X_mb, y_mb = create_mini_batch(X,y,s,s + n_b)
		
		# WRITEME: gradient calculations and update rules go here
		
		s += n_b

print(' > Training loop completed!')
# TODO: remove this line below once you have correctly implemented/gradient-checked your various sub-routines
sys.exit(0) 

scores, probs = predict(X,theta)
predicted_class = (tf.argmax(scores, axis=1))
print 'training accuracy: {0}' % ((tf.reduce_mean(predicted_class == y)))

scores, probs = predict(X_v,theta)
predicted_class = (tf.argmax(scores, axis=1))
print 'validation accuracy: {0}' % ((tf.reduce_mean(predicted_class == y_v)))

# NOTE: write your plot generation code here (for example, using the "train_cost" and "valid_cost" list variables)
