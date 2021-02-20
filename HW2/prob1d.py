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

Problem 1d: MLPs \& the Spiral Problem

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
	return (scores,probs)
	
np.random.seed(0)
# Load in the data from disk
path = os.getcwd() + '/data/spiral_train.dat'  
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
check = 10 # every so many pass/epochs, print loss/error to terminal
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
for i in xrange(n_e):
	# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
	loss = 0.0
	if i % check == 0:
		print "iteration %d: loss %f" % (i, loss)

# perform a parameter update
# WRITEME: write your update rule(s) here
 
# TODO: remove this line below once you have correctly implemented/gradient-checked your various sub-routines
sys.exit(0) 
  

scores, probs = predict(X,theta)
predicted_class = (tf.argmax(scores, axis=1))
print 'training accuracy: %.2f' % ((tf.reduce_mean(predicted_class == y)))

# plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
					 
Z, P = predict(tf.concatenate([xx.ravel(), yy.ravel()]), theta)

Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
#fig.savefig('spiral_net.png')

plt.show()
