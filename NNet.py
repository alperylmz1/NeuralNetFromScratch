import numpy as np
from sklearn.metrics import accuracy_score

class NNet():
    def __init__(self , input_size , hidden_size , output_size):
        self.model = self.initialize_parameters(input_size , hidden_size , output_size)
        self.losses = {"loss":[] , "accuracy":[]}

        self.activations = {"relu" : [self.relu , self.relu_derivative] , "tanh" : [np.tanh , self.tanh_derivative] , "sigmoid" : [self.sigmoid , self.sigmoid_derivative] , "leaky_relu":[self.leakyrelu , self.leakyrelu_derivative]}

        return

    def softmax(self , z):
        exp_ = np.exp(z)
        return exp_ / np.sum(exp_ , axis = 1 , keepdims=True)
    
    def softmax_loss(self , y,y_hat):
        minval = 0.000000000001

        m = y.shape[0]
        loss = -1/m * np.sum(y * np.log(y_hat.clip(min=minval)))
        return loss
    
    def loss_derivative(self , y,y_hat):
        return (y_hat-y)

    
    def _tanh():
        pass
        #use np version.
    
    def tanh_derivative(self , x):
        return (1 - np.power(x, 2))
    

    def sigmoid(self, x):
        return (1/(1 + np.exp(-x)))

    def sigmoid_derivative(self , x):
        return x*(1-x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        x = np.where(x < 0, 0, x)
        x = np.where(x >= 0, 1, x)
        return x

    def leakyrelu(self, x):
        return np.where(x > 0, x, x * 0.01) 
    
    def leakyrelu_derivative(self, x):
        return np.where(x > 0, 1, 0.01) 


    def forward_prop(self, X_train , activation):

        W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']
        
        z1 = X_train.dot(W1) + b1

        a1 = self.activations[activation][0](z1)
    
        z2 = a1.dot(W2) + b2
    
        a2 = self.activations[activation][0](z2)
    
        z3 = a2.dot(W3) + b3
    
        a3 = self.softmax(z3)
    
  
        cache = {'a0':X_train,'z1':z1,'a1':a1,'z2':z2,'a2':a2,'a3':a3,'z3':z3}
        return cache

    def backward_prop(self, cache, y_train , activation):

        W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']
        
        a0, a1, a2, a3 = cache['a0'], cache['a1'], cache['a2'], cache['a3']

        m = y_train.shape[0]
        
        dz3 = self.loss_derivative(y=y_train , y_hat=a3)

        dW3 = 1/m*(a2.T).dot(dz3) 
        
        db3 = 1/m*np.sum(dz3, axis=0)

        dz2 = np.multiply(dz3.dot(W3.T) ,self.activations[activation][1](a2))
        
        dW2 = 1/m*np.dot(a1.T, dz2)
        
        db2 = 1/m*np.sum(dz2, axis=0)
        
        dz1 = np.multiply(dz2.dot(W2.T),self.activations[activation][1](a1))
        
        dW1 = 1/m*np.dot(a0.T,dz1)
        
        db1 = 1/m*np.sum(dz1,axis=0)
        
        grads = {'dW3':dW3, 'db3':db3, 'dW2':dW2,'db2':db2,'dW1':dW1,'db1':db1}
        return grads
    

    def initialize_parameters(self, nn_input_dim, nn_hdim, nn_output_dim):

        W1 = np.random.randn(nn_input_dim, nn_hdim) *0.1
        
        b1 = np.zeros((1, nn_hdim))
        
        W2 = np.random.randn(nn_hdim, nn_hdim) *0.1
        
        b2 = np.zeros((1, nn_hdim))
        W3 = np.random.rand(nn_hdim, nn_output_dim) *0.1
        b3 = np.zeros((1, nn_output_dim))
        
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,'W3':W3,'b3':b3}
        return model
    

    def update_parameters(self, grads, learning_rate):

        W1, b1, W2, b2,b3,W3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['b3'], self.model["W3"]
        
        W1 -= learning_rate * grads['dW1']
        b1 -= learning_rate * grads['db1']
        W2 -= learning_rate * grads['dW2']
        b2 -= learning_rate * grads['db2']
        W3 -= learning_rate * grads['dW3']
        b3 -= learning_rate * grads['db3']
        
        self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3':W3,'b3':b3}
        return self.model
            
    def predict(self, x , activation):

        c = self.forward_prop(x , activation=activation)

        y_hat = np.argmax(c['a3'], axis=1)
        return y_hat
    
    def calc_accuracy(self, x ,y):

        m = y.shape[0]

        pred = self.predict(self.model,x)

        pred = pred.reshape(y.shape)

        error = np.sum(np.abs(pred-y))

        return (m - error)/m * 100
    


    def train(self, X_, y_, learning_rate, activation, epochs=20000, print_loss=False):

        for i in range(0, epochs):

            cache = self.forward_prop(X_ , activation=activation)
            
            grads = self.backward_prop(cache,y_ , activation=activation)

            self.model = self.update_parameters(grads=grads,learning_rate=learning_rate)
            
            a3 = cache['a3']
            y_hat = self.predict(X_ ,activation=activation)
            y_true = y_.argmax(axis=1)
            print("Epoch {} / {} : loss: {}  -  accuracy: {}%".format(i , epochs , self.softmax_loss(y_,a3) ,  accuracy_score(y_pred=y_hat,y_true=y_true)*100))


            self.losses["accuracy"].append(accuracy_score(y_pred=y_hat,y_true=y_true)*100)
            self.losses["loss"].append(self.softmax_loss(y_,a3))




        return self.losses
    
    

