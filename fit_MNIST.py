import numpy as np
from NNLibrary import mnist
from NNLibrary import sequential_class
from NNLibrary import tanh
from NNLibrary import mse
from NNLibrary import mse_prime
from NNLibrary import linearlayer
import pickle
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from NNLibrary import sigmoid
import matplotlib.pyplot as plt

def mnist_train_test():

  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  print("xtrain shape",len(x_train))
  print("xtest shape",len(x_test))
# Resize and normalize input data
  x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = np_utils.to_categorical(y_train)
# Separate the training dataset from some validation datasets.
  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train)
  x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
  x_test = x_test.astype('float32')
  x_test /= 255
  y_test = np_utils.to_categorical(y_test)
  
  print("\n\n Model 1")
  mnist_l1 = sequential_class(mse,mse_prime)
  mnist_l1.add_layer(linearlayer(28*28, 100))                
  mnist_l1.add_layer(tanh())
  mnist_l1.add_layer(linearlayer(100, 50))                   
  mnist_l1.add_layer(tanh())
  mnist_l1.add_layer(linearlayer(50, 10))                    
  mnist_l1.add_layer(tanh())
  mnist_l1.fit(x_train[:1000], y_train[:1000], epochs=100, learning_rate=0.1,modelname='Mnist Model 1: ')
  x_val_pred = (mnist_l1.predict(x_val))
  print("\n\n Model 1")
  print('\n784 -> 100 -> 50')
  print("Activation Function: Tangent")
  print("Learning Rate: 0.1")
  print("Batch Size: 32")
  print("Max Epochs: 100")
  print("Early Stopping: 5")
  print('Model 1 validation set Accuracy : ',mnist_l1.testaccuracy(x_val_pred,y_val))
  x_test_pred = (mnist_l1.predict(x_test))
  print('Model 1 test set Accuracy : ',mnist_l1.testaccuracy(x_test_pred,y_test))
  samples = 5
  for test, true in zip(x_test[:samples], y_test[:samples]):
      pred = mnist_l1.predict(test)
      indx = np.argmax(pred)
      indx_true = np.argmax(true)
      print('mnist_l1---pred: %s, true: %d' % (indx, indx_true))
  with open('mnist_l1.w', 'wb') as files:
     pickle.dump(mnist_l1, files)

  print("\n\n Model 2")
  mnist_l2 = sequential_class(mse,mse_prime)
  mnist_l2.add_layer(linearlayer(28*28, 50))                
  mnist_l2.add_layer(sigmoid())
  mnist_l2.add_layer(linearlayer(50, 50))                   
  mnist_l2.add_layer(sigmoid())
  mnist_l2.add_layer(linearlayer(50, 10))                    
  mnist_l2.add_layer(tanh())
  mnist_l2.fit(x_train[:1000], y_train[:1000], epochs=100, learning_rate=0.1,modelname='Mnist Model 2: ')
  print("\n\n Model 2")
  print('\n784 -> 50 -> 50')
  print("Activation Function: Sigmoid")
  print("Learning Rate: 0.1")
  print("Batch Size: 32")
  print("Max Epochs: 100")
  print("Early Stopping: 5")
  x_val_pred = (mnist_l2.predict(x_val))
  print('Model 2 validation set Accuracy : ',mnist_l2.testaccuracy(x_val_pred,y_val))
  x_test_pred = (mnist_l2.predict(x_test))
  print('Model 2 test set Accuracy : ',mnist_l2.testaccuracy(x_test_pred,y_test))
  for test, true in zip(x_test[:samples], y_test[:samples]):
      pred = mnist_l2.predict(test)
      indx = np.argmax(pred)
      indx_true = np.argmax(true)
      print('mnist_l2---pred: %s, true: %d' % (indx, indx_true))
  with open('mnist_l2.w', 'wb') as files:
     pickle.dump(mnist_l2, files)

  print("\n\n Model 3")
  mnist_l3 = sequential_class(mse,mse_prime)
  mnist_l3.add_layer(linearlayer(28*28, 200))                
  mnist_l3.add_layer(tanh())
  mnist_l3.add_layer(linearlayer(200, 100))                   
  mnist_l3.add_layer(tanh())
  mnist_l3.add_layer(linearlayer(100, 50))                    
  mnist_l3.add_layer(tanh())
  mnist_l3.add_layer(linearlayer(50, 25))                    
  mnist_l3.add_layer(tanh())
  mnist_l3.add_layer(linearlayer(25, 10))                    
  mnist_l3.add_layer(tanh()) 
  mnist_l3.fit(x_train[:1000], y_train[:1000], epochs=100, learning_rate=2,modelname='Mnist Model 3: ')
  x_val_pred = (mnist_l3.predict(x_val))
  print("\n\n Model 3")
  print('\n784 -> 200 -> 100 -> 50 -> 25')
  print("Activation Function: Tangent")
  print("Learning Rate: 2")
  print("Batch Size: 32")
  print("Max Epochs: 100")
  print("Early Stopping: 5")
  print('Model 3 validation set Accuracy : ',mnist_l3.testaccuracy(x_val_pred,y_val))
  x_test_pred = (mnist_l3.predict(x_test))
  print('Model 3 test set Accuracy : ',mnist_l3.testaccuracy(x_test_pred,y_test))
  for test, true in zip(x_test[:samples], y_test[:samples]):
      pred = mnist_l3.predict(test)
      indx = np.argmax(pred)
      indx_true = np.argmax(true)
      print(' mnist_l3--- pred: %s, true: %d' % (indx, indx_true))
  with open('mnist_l3.w', 'wb') as files:
     pickle.dump(mnist_l3, files)


def hyperparameters():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
# Resize and normalize input data
  x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = np_utils.to_categorical(y_train)
# Separate the training dataset from some validation datasets.
  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train)
  x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
  x_test = x_test.astype('float32')
  x_test /= 255
  y_test = np_utils.to_categorical(y_test)
  
  print("\n\n Hyperparameter Model 1")
  hyperparameter_l1 = sequential_class(mse,mse_prime)
  hyperparameter_l1.add_layer(linearlayer(28*28, 50, np.zeros([28*28,50]), np.zeros([1,50])))                
  hyperparameter_l1.add_layer(tanh())
  hyperparameter_l1.add_layer(linearlayer(50, 10))                    
  hyperparameter_l1.add_layer(tanh())
  hyperparameter_l1.fit(x_train[:1000], y_train[:1000], epochs=100, learning_rate=0.1,modelname='Hyperparameter Model 1: ')
  x_val_pred = (hyperparameter_l1.predict(x_val))
  print("\n\n Hyperparameter Model 1")
  print('\n784 -> 50')
  print("Activation Function: Tangent")
  print("Learning Rate: 0.1")
  print("Batch Size: 32")
  print("Max Epochs: 100")
  print("Early Stopping: 5")
  print('Hyperparameter Model 1 validation set Accuracy : ',hyperparameter_l1.testaccuracy(x_val_pred,y_val))
  x_test_pred = (hyperparameter_l1.predict(x_test))
  print('Hyperparameter Model 1 test set Accuracy : ',hyperparameter_l1.testaccuracy(x_test_pred,y_test))
  samples = 5
  for test, true in zip(x_test[:samples], y_test[:samples]):
      pred = hyperparameter_l1.predict(test)
      indx = np.argmax(pred)
      indx_true = np.argmax(true)
      print('Hyperparameter Model 1---pred: %s, true: %d' % (indx, indx_true))
  with open('hyperparameter_l1.w', 'wb') as files:
     pickle.dump(hyperparameter_l1, files)

  print("\n\n Hyperparameter Model 2")
  high = 10
  low = -10   
  hyperparameter_l2 = sequential_class(mse,mse_prime)
  hyperparameter_l2.add_layer(linearlayer(28*28, 50, np.random.rand(28*28,50) * (high - low) + low, np.random.rand(1,50) * (high - low) + low ))                
  hyperparameter_l2.add_layer(tanh())
  hyperparameter_l2.add_layer(linearlayer(50, 10))                   
  hyperparameter_l2.fit(x_train[:1000], y_train[:1000], epochs=100, learning_rate=0.1,modelname='Hyperparameter Model 2: ')
  x_val_pred = (hyperparameter_l1.predict(x_val))
  print("\n\n Hyperparameter Model 2")
  print('\n784 -> 50')
  print("Activation Function: Tangent")
  print("Learning Rate: 0.1")
  print("Batch Size: 32")
  print("Max Epochs: 100")
  print("Early Stopping: 5")
  print('Hyperparameter Model 2 validation set Accuracy : ',hyperparameter_l1.testaccuracy(x_val_pred,y_val))
  x_test_pred = (hyperparameter_l1.predict(x_test))
  print('Hyperparameter Model 2 test set Accuracy : ',hyperparameter_l1.testaccuracy(x_test_pred,y_test))
  samples = 5
  for test, true in zip(x_test[:samples], y_test[:samples]):
      pred = hyperparameter_l1.predict(test)
      indx = np.argmax(pred)
      indx_true = np.argmax(true)
      print('Hyperparameter Model 2---pred: %s, true: %d' % (indx, indx_true))

  print("\n\n Hyperparameter Model 3")
  hyperparameter_l3 = sequential_class(mse,mse_prime)
  hyperparameter_l3.add_layer(linearlayer(28*28, 100))                
  hyperparameter_l3.add_layer(tanh())                  
  hyperparameter_l3.add_layer(linearlayer(100, 50))                  
  hyperparameter_l3.add_layer(tanh())
  hyperparameter_l3.add_layer(linearlayer(50, 10))                    
  hyperparameter_l3.add_layer(tanh())

  hyperparameter_l3.fit(x_train[:1000], y_train[:1000], epochs=100, learning_rate=1,modelname='Hyperparameter Model 3: ')
  x_val_pred = (hyperparameter_l1.predict(x_val))
  print("\n\n Hyperparameter Model 3")
  print('\n784 -> 100 -> 50')
  print("Activation Function: Tangent")
  print("Learning Rate: 1")
  print("Batch Size: 32")
  print("Max Epochs: 100")
  print("Early Stopping: 5")
  print('Hyperparameter Model 3 validation set Accuracy : ',hyperparameter_l1.testaccuracy(x_val_pred,y_val))
  x_test_pred = (hyperparameter_l1.predict(x_test))
  print('Hyperparameter Model 3 test set Accuracy : ',hyperparameter_l1.testaccuracy(x_test_pred,y_test))
  samples = 5
  for test, true in zip(x_test[:samples], y_test[:samples]):
      pred = hyperparameter_l3.predict(test)
      indx = np.argmax(pred)
      indx_true = np.argmax(true)
      print('Hyperparameter Model 3---pred: %s, true: %d' % (indx, indx_true))


  print("\n\n Hyperparameter Model 4")
  hyperparameter_l4 = sequential_class(mse,mse_prime)
  hyperparameter_l4.add_layer(linearlayer(28*28, 100))                
  hyperparameter_l4.add_layer(tanh())
  hyperparameter_l4.add_layer(linearlayer(100, 50))                   
  hyperparameter_l4.add_layer(tanh())
  hyperparameter_l4.add_layer(linearlayer(50, 10))                    
  hyperparameter_l4.add_layer(tanh())
  hyperparameter_l4.fit(x_train[:1000], y_train[:1000], epochs=100, learning_rate=0.001,modelname='Hyperparameter Model 4: ')
  x_val_pred = (hyperparameter_l1.predict(x_val))
  print("\n\n Hyperparameter Model 4")
  print('\n784 -> 100 -> 50')
  print("Activation Function: Tangent")
  print("Learning Rate: 0.001")
  print("Batch Size: 32")
  print("Max Epochs: 100")
  print("Early Stopping: 5")
  print('Hyperparameter Model 4 validation set Accuracy : ',hyperparameter_l1.testaccuracy(x_val_pred,y_val))
  x_test_pred = (hyperparameter_l1.predict(x_test))
  print('Hyperparameter Model 4 test set Accuracy : ',hyperparameter_l1.testaccuracy(x_test_pred,y_test))
  samples = 5
  for test, true in zip(x_test[:samples], y_test[:samples]):
      pred = hyperparameter_l4.predict(test)
      indx = np.argmax(pred)
      indx_true = np.argmax(true)
      print('Hyperparameter Model 4---pred: %s, true: %d' % (indx, indx_true))

mnist_train_test()
hyperparameters()
plt.show()