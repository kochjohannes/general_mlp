#Multi-Layer Perceptron

import numpy as np
import matplotlib.pyplot as plt

def loss_function(Y, Y_hat):
    #y true labels, y_hat predictions
    #y and y_hat is NxD, N is the number in the batch (1, more, all), D is the output dimension

    #Cross-entropy Loss (Negative Log Loss) - for binary classification
    N = np.size(Y, axis=0) #Average loss over samples
    loss = -1/N * (np.dot(np.transpose(Y), np.log(Y_hat)) + np.dot(np.transpose(1.0-Y), np.log(1.0 - Y_hat))) #Might need to transpose left side
    return loss

def forward_fully_connected(X, W, b=0):
    #Calculates input to next layer
    #X is NxD (network input or output from previous layer). N samples, D dimensions or hidden units in previous layer
    #W is DxK. K number of hidden units in layer
    #b is 1xK

    #Adding bias is optional so that it can be included before calling the function, by adding a row of ones onto the
    #left side of X, and adding another row of weights to W

    #Fully-connected forward pass (a linear combination of inputs + bias)
    return np.dot(X, W) + b

def activation_function(X):
    #returns output from the activation function

    #Sigmoid
    return 1.0/(1.0 + np.exp(-X))

def backward_fully_connected(Z_bar, X, W):
    #Backpropagation through a fully connected layer Z=XW
    #Z_bar = dL/dZ where L is loss function and Z is input to layer
    X_bar = 1.0/100 * np.dot(Z_bar, np.transpose(W)) #CHANGE TO DIVIDE BY BATCHSIZE OR REMOVE
    W_bar = 1.0/100 * np.dot(np.transpose(X), Z_bar) #CHANGE TO DIVIDE BY BATCHSIZE OR REMOVE
    b_bar = Z_bar

    #print(X.shape)
    #print(Z_bar.shape)
    return X_bar, W_bar, b_bar

def differentiate_activation(X):
    #Analytic differentiation of activation function for use in backpropagation of network

    return (1-activation_function(X))*activation_function(X)    #Note that this is valid for the sigmoid,
                                                                #Will be a different analytic derivative
                                                                #if another activation function is used

def backward_activation(Y_bar, Z):
    #Backpropagation through a layer with activation functions
    #Y_bar dL/dY, i.e the gradient of the loss function w.r.t the output from layer, Z is input to layer
    Z_bar = np.multiply(differentiate_activation(Z), Y_bar)
    return Z_bar





#Let's do one hidden layer with 3 units to start with:
#2D input to hidden layer with 3 units --> W1 will be 2x3, b1 will be 1x3
#3D output from hidden layer to 1D output from layer --> W2 will be 3x1, b2 will be 1x1

#X-->Z1-->H1-->Z2-->Y

N_input_dimensions = 2
N_hidden_layers = 3 #Number of layers not including input layer and output layer
N_units_hidden_layer = [100, 100, 100]
#N_hidden_units_layer_1 = 2
#N_hidden_units_layer_2 = 20
N_output_dimensions = 1

#Weight initialization
np.random.seed(seed=2)

#N_units_layer = []
#N_units_layer.append(N_input_dimensions)
#N_units_layer.append(N_units_hidden_layer)
#N_units_layer.append(N_output_dimensions)

W_list = []
b_list = []

for i in range(len(N_units_hidden_layer)+1):
    if(i==0):
        W = np.random.randn(N_input_dimensions,N_units_hidden_layer[i])*np.sqrt(2.0/N_units_hidden_layer[i])
        b = np.random.randn(1,N_units_hidden_layer[i])*np.sqrt(2.0/N_units_hidden_layer[i])
    elif(i==len(N_units_hidden_layer)):
        W = np.random.randn(N_units_hidden_layer[i-1],N_output_dimensions)*np.sqrt(2.0/N_output_dimensions)
        b = np.random.randn(1,N_output_dimensions)*np.sqrt(2.0/N_output_dimensions)
    else:
        W = np.random.randn(N_units_hidden_layer[i-1],N_units_hidden_layer[i])*np.sqrt(2.0/N_units_hidden_layer[i])
        b = np.random.randn(1,N_units_hidden_layer[i])*np.sqrt(2.0/N_units_hidden_layer[i])

    W_list.append(W)
    b_list.append(b)

Z_list = []
H_list = []

def forward_pass(X, W_list, b_list): #This function is network specific and is so created here instead of earlier
    #Calculates network output given input
    Z_list.clear()
    H_list.clear()
    layer_input = X
    for i in range(N_hidden_layers+1):
        Z = forward_fully_connected(layer_input, W_list[i], b_list[i])
        H = activation_function(Z)
        layer_input = H

        Z_list.append(Z)
        H_list.append(H)

    #Input to first layer
    #Z1 = forward_fully_connected(X, W1, b1)
    #Output of first (hidden) layer
    #H1 = activation_function(Z1)
    #Input to second hidden layer
    #Z2 = forward_fully_connected(H1, W2, b2)
    #Output from second hidden layer
    #H2 = activation_function(Z2)
    #Input to output layer
    #Z3 = forward_fully_connected(H2, W3, b3)
    #Output of network
    #Y = activation_function(Z3)

    return Z_list, H_list

def update_weights(stepsize, W, b, W_bar, b_bar):
    #print(W1_bar.shape)
    return W - stepsize*W_bar, b - stepsize*np.mean(b_bar, axis=0)
    #return W1 - stepsize*W1_bar, b1 - stepsize*b1_bar,W2 - stepsize*W2_bar, b2 - stepsize*b2_bar,W3 - stepsize*W3_bar, b3 - stepsize*b3_bar


#Generate data
#c0 = np.random.randn(50,2) #+ np.array([0, 4])#class 0
#c1 = np.dot(np.random.randn(50,2), np.array([[1, 0], [0, 3]])) #class 1
#c1 = np.random.randn(50,2) #+ np.array([0, 4])#class 0
#c1 += 1.5


phi_tmp = np.arange(0, 6*np.pi, 6*np.pi/50)
x_tmp = np.arange(0, 1, 1.0/50)
c0 = np.array([[np.multiply(x_tmp, np.cos(phi_tmp))], [np.multiply(x_tmp, np.sin(phi_tmp))]])
c1 = np.array([[0.8*np.multiply(x_tmp, np.cos(phi_tmp))], [0.8*np.multiply(x_tmp, np.sin(phi_tmp))]])
c = np.concatenate([c0, c1], axis=2)
c = c.reshape([2, 100])
X = c.transpose()

#X = np.concatenate((c0,c1), axis=0)
Y_true = np.concatenate((np.zeros([50,1]), np.ones([50,1])), axis=0)


N_epochs = 100
stepsize = 1
L = np.zeros(N_epochs)
k = 0
for i in range(N_epochs):

    Z_list, H_list = forward_pass(X, W_list, b_list)
    L[i] = loss_function(Y_true, H_list[-1])
    #Y_bar = (Y_hat - H_list[-1])
    H_bar = (H_list[-1] - Y_true) #CHECK THIS !!!
    for j in range(N_hidden_layers+1):
        if(j==0):
            Z_bar = H_bar
        else:
            Z_bar = backward_activation(H_bar, Z_list[-(j+1)])
        if(j == N_hidden_layers):
            H_bar, W_bar, b_bar = backward_fully_connected(Z_bar, X, W_list[-(j+1)])
        else:
            H_bar, W_bar, b_bar = backward_fully_connected(Z_bar, H_list[-(j+2)], W_list[-(j+1)])
        #print(-(j+2))
        #print(W_list[-(j+1)].shape)
        #print(W_bar.shape)
        W_list[-(j+1)], b_list[-(j+1)] = update_weights(stepsize, W_list[-(j+1)], b_list[-(j+1)], W_bar, b_bar)


    #Z1, H1, Z2, H2, Z3, Y_hat = forward_pass(X, W1, b1, W2, b2, W3, b3)
    #L[i] = loss_function(Y_true, Y_hat)
    ##N = np.size(X,0)
    #Y_bar = (Y_hat - Y_true) #CHECK THIS
    ##Y_bar = -1/N*(np.divide(Y_true,Y_hat) + np.divide((1-Y_true),(Y_hat-1))) #dL/dY, specific for NLL with sigmoids
    ##print(Y_bar.shape)

    #Z3_bar = backward_activation(Y_bar, Z3)
    #H2_bar, W3_bar, b3_bar = backward_fully_connected(Z3_bar, H2, W3)

    #Z2_bar = backward_activation(H2_bar, Z2)
    #H1_bar, W2_bar, b2_bar = backward_fully_connected(Z2_bar, H1, W2)

    #Z1_bar = backward_activation(H1_bar, Z1)
    #X_bar, W1_bar, b1_bar = backward_fully_connected(Z1_bar, X, W1)

    #W1, b1, W2, b2, W3, b3 = update_weights(stepsize, W1, b1, W2, b2, W3, b3, W1_bar, b1_bar, W2_bar, b2_bar, W3_bar, b3_bar)

    ##print(W3.shape)

    ##if(i==np.round(N_epochs/(10-k))):
    # #  print(100*i/N_epochs)
    #  # k += 1
    ##if(i == np.round(N_epochs/2)):
    ##    stepsize = 0.0001

plt.plot(L[0:])
plt.show()
plt.plot(X[0:50,0], X[0:50,1], 'ro')
plt.plot(X[50:100,0], X[50:100,1], 'bx')
plt.show()

boundary_x = []
boundary_y = []
for x in np.arange(-1, 1, 6/200):
    for y in np.arange(-1, 1, 6/200):
        Z_list, H_list = forward_pass([x, y], W_list, b_list)
        if(H_list[-1]>= 0.49 and H_list[-1]<=0.51):
            boundary_x.append(x)
            boundary_y.append(y)

plt.plot(X[0:50,0], X[0:50,1], 'ro')
plt.plot(X[50:100,0], X[50:100,1], 'bx')
plt.plot(boundary_x, boundary_y, 'c.')
plt.show()
