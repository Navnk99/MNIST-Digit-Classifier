import numpy as np
import pickle
import pandas as pd
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Initialization of the Parent Class Layer
class Layer:
  def __init__(self,ipsize,opsize,wt_arr=[0],bias_arr=[0]):
    self.input=ipsize
    self.output=opsize
    if(len(wt_arr)!= len(bias_arr)):
          self.wt=wt_arr
          self.bias=bias_arr
    else:
          self.wt = np.random.rand(ipsize, opsize) - 0.5
          self.bias = np.random.rand(1, opsize) - 0.5
    self.z=0
  def forward(self, input):
    raise NotImplementedError
  def backward(self, op_err, learning_rate):
     raise NotImplementedError
    
#The linear layer is initialized, and forward and backward propagation are used to update the weights and biases.
class linearlayer(Layer):
  def forward(self,input):
    self.ipdata=input
    f=np.dot(self.ipdata, self.wt) + self.bias
    return f
  def backward(self,op_err,learning_rate):
    dfw=self.ipdata
    dfx=self.wt
    ip_err = np.dot(op_err, dfx.T)
    wt_error = np.dot(dfw.T, op_err)
    
    self.wt=self.wt-learning_rate*wt_error
    self.bias=self.bias-learning_rate*op_err

#This gives the sensitivity of the cost function relative to the preceding activation function, which is helpful for calculating the change in weight required at inp.
    return ip_err
  
#Sigmoid activation function class is initialized 
class sigmoid(Layer):
  def __init__(self):
    pass
  #forward propagation sigmoid(z) formula is found using formula 
  def forward(self,input):
    self.ipdata=input
    #return 1. / (1. + np.exp(-self.z))
    return 1. / (1. + np.exp(-self.ipdata))
  #backward propagation sigmoid_prime(z) is found using formula
  def backward(self, op_err, learning_rate):
    sig_1= (1. / (1. + np.exp(-self.ipdata)))*(1- (1. / (1. + np.exp(-self.ipdata))))*op_err
    return sig_1
  
#Tanh activation function class is initialized 
class tanh(Layer):
  def __init__(self):
    pass
  #forward propagation tanh(z) formula is found using formula
  def forward(self,input):
    self.ipdata=input
    tanh_value=(np.exp(self.ipdata) - np.exp(-self.ipdata)) / (np.exp(self.ipdata) + np.exp(-self.ipdata))
    return tanh_value

  #backward propagation tanh_1(z) is found using formula 
  def backward(self, op_err,learning_rate):
    tanh_value=(np.exp(self.ipdata) - np.exp(-self.ipdata)) / (np.exp(self.ipdata) + np.exp(-self.ipdata))
    tanh_1= (1-(tanh_value)**2)*op_err 
    return tanh_1
  
#softmax activation function class is initialized 
class softmax(Layer):
  def forward(self,input):
    self.ipdata=input
    self.z=(np.dot(self.ipdata, self.wt) + self.bias)
    num = np.exp(self.z- np.max(self.z))
    return num / np.sum(num, axis=0, keepdims=True)

  def backward(self, probs, bp_err):
    dim = probs.shape[1]
    output = np.empty(probs.shape)
    for j in range(dim):
        d_prob = - (probs * probs[:,[j]]) 
        d_prob[:,j] += probs[:,j]  
        output[:,j] = np.sum(bp_err * d_prob, axis=1)
    return output

#Crossentropy class initiallized
class crossentropy(Layer):
  def forward(pred, target):
    return -target * np.log(pred)
  def backward(pred, target):
    return target - pred
  
#To figure out the loss function's derivative and the class mean square error:
def mse(y_true, y_pred):
  return np.mean(np.power(y_true-y_pred, 2));

def mse_prime(y_true, y_pred):
  return 2*(y_pred-y_true)/y_true.size;

class sequential_class(Layer):
#We initialize loss and loss_grad as zero to change and are variable between mse and crossentropy loss.
  def __init__(self,loss=None,loss_grad=None):
    self.layers = []
    self.loss=loss
    self.loss_grad=loss_grad
  #new layers are added here
  def add_layer(self,newlayer):
    self.layers.append(newlayer)
    return(self.layers)
  #this is to display weight
  def display_wt(self):
    for layer in self.layers:
      print(layer.wt)
  def predict(self,input):
    smp_size=len(input)
    predic=[]
    for i in range(smp_size):
      output=input[i]
      #print("output1= ",smp_size)
      for layer in self.layers:
        output = layer.forward(output)
        #print("output= "+str(layer),output)
      predic.append(output)
    return predic
  def fit(self, x_train, y_train, epochs, learning_rate,modelname=""):
    count=0
    self.modelname=modelname
    smp_size=len(x_train)
    list_of_epochs = []
    list_of_errors = []
    for i in range(epochs):
      err_val=0
      for j in range(smp_size):
    # forward propagation
        output = x_train[j]
    #print(output)
        for layer in self.layers:
    #print(layer)
          output = layer.forward(output)
        err_val += self.loss(y_train[j], output)
    # backward propagation
        error = self.loss_grad(y_train[j], output)
        for layer in reversed(self.layers):
          error = layer.backward(error, learning_rate)

    # average the errors for all samples.
      err_val /= smp_size
      list_of_errors.append(err_val)
      if((err_val==list_of_errors[-1]) and (count==5)):
        break
        count+=1
      #print(len(list_of_errors))
      list_of_epochs.append(i)
      print('epoch %d/%d   error=%f' % (i+1, epochs, err_val))
      if(count==5):
        break
    fig = plt.figure(figsize=[6,10])
    ax = fig.add_subplot(1,1, 1)
    ax.plot(list_of_epochs,list_of_errors, color='b', linestyle="-")
    ax.set_title(modelname)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')

  def testaccuracy(self, y_pred, y):
    samples = len(y_pred)
    accuracy = 0
    for i in range(samples):
      abs_out = np.array([abs(np.round(x)) for x in y_pred[i]])[0]
      if np.array_equal(abs_out,np.array(y[i])):
        accuracy += 1
    return accuracy/samples     

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


def XOR_Test():
  x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
  y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
  XOR_Mod1=sequential_class(mse,mse_prime)
  XOR_Mod1.add_layer(linearlayer(2,2))
  XOR_Mod1.add_layer(sigmoid())
  XOR_Mod1.add_layer(linearlayer(2,1))
  XOR_Mod1.add_layer(sigmoid())
  XOR_Mod1.fit(x_train, y_train, epochs=5000, learning_rate=0.01,modelname='XOR prediction using Sigmoid Training: ')
  op_sigmoid = XOR_Mod1.predict(x_train)
  with open('XOR_solved_sigmoid.w', 'wb') as files:
     pickle.dump(XOR_Mod1, files)
  print("XOR prediction using Sigmoid Training: ",op_sigmoid)
  print("\n------------------------------------------\n")

  XOR_Mod2=sequential_class(mse,mse_prime)
  XOR_Mod2.add_layer(linearlayer(2,2))
  XOR_Mod2.add_layer(tanh())
  XOR_Mod2.add_layer(linearlayer(2,1))
  XOR_Mod2.add_layer(tanh())
  XOR_Mod2.fit(x_train, y_train, epochs=5000, learning_rate=0.01,modelname='XOR prediction using Tanh Training: ')
  op_tanh = XOR_Mod2.predict(x_train)
  with open('XOR_solved_tanh.w', 'wb') as files:
     pickle.dump(XOR_Mod2, files)
  print("XOR prediction using Tanh Training: ",op_tanh)
  print("\n------------------------------------------\n")


  def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output
 
XOR_Test()
