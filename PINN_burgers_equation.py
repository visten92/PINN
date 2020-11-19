"""
A Physics Informed Neural Network implementation in Tensorflow 2 - Keras
The model predicts the solution of the burgers equation.

Python interpreter version: 3.7
TensorFlow version: 2.1.0
TensorFlow Probability version: 0.9.0
NumPy version: 1.17.2
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from pyDOE import lhs
import scipy.io
from tensorflow.keras import layers
from PINN_keras_helper import optimizer_function_factory
from PINN_keras_helper import differential_equation_loss

N_u = 100                    # number of data points
N_f = 10000                  # number of points where the differential equation must be satisfied

data = scipy.io.loadmat('burgers_equation.mat')                     # load Burgers equation dataset from matlab file
t = data['t'].flatten()[:, None].astype('float32')                  # time discretization points
x = data['x'].flatten()[:, None].astype('float32')                  # spatial discretization points (1-D)
exact_sol = np.real(data['usol']).T.astype('float32')               # exact solution

X, T = np.meshgrid(x, t)                                            # create a mesh from data coordinates x,t

X_domain = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))  # includes spatial and time coordinates
u_domain = exact_sol.flatten()[:, None]

# Doman bounds
lb = X_domain.min(0)
ub = X_domain.max(0)

# In the code snipet below, we select only a few characteristic points for our final dataset
xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
uu1 = exact_sol[0:1, :].T
xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
uu2 = exact_sol[:, 0:1]
xx3 = np.hstack((X[:, -1:], T[:, -1:]))
uu3 = exact_sol[:, -1:]

X_u_train = np.vstack([xx1, xx2, xx3])      # the coordinates (x,t) of the training points
X_f_train = lb + (ub - lb) * lhs(2, N_f)    # the coordinates (x,t) of the points to satisfy the differential equation
X_f_train = tf.convert_to_tensor(np.vstack((X_f_train, X_u_train)).astype('float32'))
u_train = np.vstack([uu1, uu2, uu3])        # the solution of the training points

# We will further reduce our dataset and select only N_u number of points randomly
index = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = tf.convert_to_tensor(X_u_train[index, :].astype('float32'))
u_train = tf.convert_to_tensor(u_train[index, :].astype('float32'))

# split spatial and time coordinates
x_u = X_u_train[:, 0:1]
t_u = X_u_train[:, 1:2]
x_f = X_f_train[:, 0:1]
t_f = X_f_train[:, 1:2]

# create a custom layer class
class Linear(layers.Layer):
    def __init__(self, units=20):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer=tf.keras.initializers.GlorotNormal(),
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer=tf.keras.initializers.GlorotNormal(),
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# create a custom model class
class myModel(layers.Layer):
    def __init__(self):
        super(myModel, self).__init__()
        self.linear_1 = Linear(20)
        self.linear_2 = Linear(20)
        self.linear_3 = Linear(20)
        self.linear_4 = Linear(20)
        self.linear_5 = Linear(20)
        self.linear_6 = Linear(20)
        self.linear_7 = Linear(20)
        self.linear_8 = Linear(20)
        self.linear_9 = Linear(1)

    def call(self, t, x):
        y = self.linear_1(tf.concat([t, x], 1))
        y = tf.nn.tanh(y)
        y = self.linear_2(y)
        y = tf.nn.tanh(y)
        y = self.linear_3(y)
        y = tf.nn.tanh(y)
        y = self.linear_4(y)
        y = tf.nn.tanh(y)
        y = self.linear_5(y)
        y = tf.nn.tanh(y)
        y = self.linear_6(y)
        y = tf.nn.tanh(y)
        y = self.linear_7(y)
        y = tf.nn.tanh(y)
        y = self.linear_8(y)
        y = tf.nn.tanh(y)
        return self.linear_9(y)

# We choose to create custom classes for Model and Layer in order to use automatic differentiation
# with respect to the input variables (x,t) in later step

# create a custom loss function
def loss(u_true, u_pred, f_pred): # f_pred is the differential equation loss
    return tf.reduce_sum(tf.square(u_true - u_pred)) / u_pred.shape[0] + \
           tf.reduce_sum(tf.square(f_pred)) / f_pred.shape[0]

def predict(model, X_test):
    X_test = tf.convert_to_tensor(X_test)
    u_test = model(X_test[:, 0:1], X_test[:, 1:2])
    f_test = differential_equation_loss(model, X_test[:, 0:1], X_test[:, 1:2])
    return u_test, f_test

#run the code
model = myModel()

func = optimizer_function_factory(model, loss, X_u_train, u_train, X_f_train) # this function makes it possible to use L-BFGS optimizer in tensorflow

# convert initial model parameters to a 1D tf.Tensor
init_params = tf.dynamic_stitch(func.idx, model.trainable_weights)
start_time = time.time()

# train the model with L-BFGS solver
results = tfp.optimizer.lbfgs_minimize(
    value_and_gradients_function=func, initial_position=init_params, max_iterations=50000,tolerance=1e-12)
elapsed = time.time() - start_time

# after training, the final optimized parameters are still in results.position so we have to manually put them back to the model
func.assign_new_model_parameters(results.position)

u_pred, f_pred = predict(model, X_domain) # predictions for all points in the domain
error_u = np.linalg.norm(u_domain - u_pred) / np.linalg.norm(u_domain)
print('Training time:', elapsed)
print('Error u: %e' % (error_u))

