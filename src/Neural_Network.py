import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense,  BatchNormalization, LeakyReLU,Dropout,ReLU
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tensorflow import keras


X = np.load(r"FEATURES-COMP.npy")
##declaring variables
n_inner_layer = 1
n_hidden_layer = 64
n_input = 3
n_output = 3


tf.keras.backend.clear_session()
##Sequential Model
model = Sequential()
model.add(Dense(n_hidden_layer,input_shape=(n_input,)))
## Dropout - nullifies the contribution of some neurons
model.add(Dropout(0.2))
""" ReLU- does not activate all the neurons at the same time"""
model.add(ReLU())
model.add(Dense(n_inner_layer))
model.add(Dropout(0.2))
##LReLu - solves gradient vanishing problem
model.add(LeakyReLU(0.6))
model.add(Dense(n_hidden_layer))
model.add(Dropout(0.2))
model.add(ReLU())
##tanh (produces a zero-centered output)
model.add(Dense(n_output,activation='tanh'))

opt = keras.optimizers.SGD(learning_rate=0.01,momentum=0.9)
model.compile(loss='mse',
              metrics=['accuracy'],
              optimizer=opt)

kf = KFold(n_splits=5,shuffle=True)
kf.split(X)
z=0
for train_index, test_index in kf.split(X):
    # Split train-test
    X_train, X_test = X[train_index,:], X[test_index,:]
    y_train, y_test = X[train_index,:],X[test_index,:]
    # Train the model
    print(X_train.shape)
    train_report = model.fit(X_train, y_train, epochs=50,batch_size=10, validation_data=(X_test, y_test),verbose=0)
    # Append to accuracy_model the accuracy of the model
    print("******Evaluate on test data with the model:**********",z)
    z=z+1
    results = model.evaluate(X_test, y_test)
    print("test loss, test acc:", results)
    print(train_report.history.keys())

model.save('ANN_64')




