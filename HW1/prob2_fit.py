import os  
import tensorflow as tf 
import pandas as pd  
import matplotlib.pyplot as plt  


'''
IST 597: Foundations of Deep Learning
Problem 1: Univariate Regression


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

# NOTE: you will need to tinker with the meta-parameters below yourself (do not think of them as defaults by any means)
# NOTE: You can work in pure eager mode or Keras or even hybrid
# NOTE : Use tf.Variable when using gradientTape, if you build your own gradientTape then use simple linear algebra using numpy or tensorflow math

# meta-parameters for program
trial_name = 'p1_fit' # will add a unique sub-string to output of this program
degree = 1 # p, order of model
beta = 0.0 # regularization coefficient
alpha = 0.0 # step size coefficient
eps = 0.0 # controls convergence criterion
n_epoch = 1 # number of epochs (full passes through the dataset)

# begin simulation

# Tip0: Use tf.function --> helps in speeding up the training
# Example

#@tf.function
#def regress(X, theta):
#	# WRITEME: write your code here to complete the routine
#	# Define your forward pass
#	return -1.0

def regress(X, theta):
	# WRITEME: write your code here to complete the routine
	# Define your forward pass
	return -1.0

def gaussian_log_likelihood(mu, y):
	# WRITEME: write your code here to complete the sub-routine
	# Define loss function
	return -1.0
	
def computeCost(X, y, theta, beta): # loss is now Bernoulli cross-entropy/log likelihood
	# WRITEME: write your code here to complete the routine
	# Cost function is also known as loss function 
	return -1.0
	
def computeGrad(X, y, theta, beta): 
	# WRITEME: write your code here to complete the routine
	# NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)
	dL_dfy = None # derivative w.r.t. to model output units (fy)
	dL_db = None # derivative w.r.t. model weights w
	dL_dw = None # derivative w.r.t model bias b
	nabla = (dL_db, dL_dw) # nabla represents the full gradient
	# You can also use gradient tape and replace this function
	return nabla


path = os.getcwd() + '/data/prob2.dat'  
data = pd.read_csv(path, header=None, names=['X', 'Y']) 
# Tip1: Convert .dat into numpy and use tensor flow api to process data
# display some information about the dataset itself here
# WRITEME: write your code here to print out information/statistics about the data-set "data" 
# WRITEME: write your code here to create a simple scatterplot of the dataset itself and print/save to disk the result


# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)

#TODO convert np array to tensor objects if working with Keras
# convert to numpy arrays and initalize the parameter array theta 
w = np.zeros((1,X.shape[1]))
b = np.array([0])
theta = (b, w)
### Important please read all comments
### or use tf.Variable to define w and b if using Keras and gradient tape
L = computeCost(X, y, theta)
print("-1 L = {0}".format(L))
L_best = L
i = 0
cost = [] # you can use this list variable to help you create the loss versus epoch plot at the end (if you want)



while(i < n_epoch):
	dL_db, dL_dw = computeGrad(X, y, theta, beta)
	b = theta[0]
	w = theta[1]
	# update rules go here...
	# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
	
	# (note: don't forget to override the theta variable...)
	L = computeCost(X, y, theta) # track our loss after performing a single step
	# Use function 
	
	print(" {0} L = {1}".format(i,L))
	i += 1
        TODO
	# print parameter values found after the search
	#print W
#print b
#Save everything into saver object in tensorflow
#Visualize using tensorboard
kludge = 0.25 # helps with printing the plots (you can tweak this value if you like)
# visualize the fit against the data

# WRITEME: write your code here to save plot to disk (look up documentation/inter-webs for matplotlib)

# visualize the loss as a function of passes through the dataset
# WRITEME: write your code here create and save a plot of loss versus epoch

# plt.show() # convenience command to force plots to pop up on desktop
