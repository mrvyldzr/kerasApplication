#CNN

import librosa
import librosa.feature
import librosa.display
import glob
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers import  Activation, Flatten,Dense,Dropout
from keras.optimizers import SGD
from keras.layers.normalization import  BatchNormalization
from keras.models import Sequential
from keras.layers import  Dense,Activation,Conv2D,Dropout,MaxPooling2D
from numpy import loadtxt



girisverisi=np.load("girisverimiz.npy")
giriverisi=np.reshape(girisverisi,(-1,224,224,3))
cikisverisi=np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]])

splitverisi=girisverisi[1:6]
splitverisi=np.append(splitverisi,girisverisi[24:29])
splitverisi=splitverisi.reshape(-1,224,224,3)
splitcikis=np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[0,1]])

#print(cikisverisi.shape)
model=Sequential()
model.add(Conv2D(50,11,strides=(4,4),input_shape=(224,224,3) ))
model =Sequential()
model.add(Conv2D(50 ,11 ,strides=(4 ,4) ,input_shape=(224 ,224 ,3)))
model.add(Conv2D(50 ,2))
model.add(Conv2D(50 ,2))
model.add(Conv2D(50 ,2))
model.add(Conv2D(50 ,2))
model.add(Conv2D(50 ,3))
model.add(Conv2D(50 ,2))
model.add(Conv2D(50 ,2))
model.add(Conv2D(50 ,2))
model.add(MaxPooling2D((5 ,5)))
model.add(Conv2D(50 ,2))
model.add(Conv2D(50 ,2))
model.add(Conv2D(50 ,2))
model.add(MaxPooling2D((3 ,3)))
model.add(Conv2D(50 ,2))



model.add(Flatten())

model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(optimizer= 'adam' , loss= keras.losses.binary_crossentropy, metrics=['accuracy'])


model.summary()
print(splitverisi.shape)
model.fit(girisverisi/255,cikisverisi,batch_size=10,epochs=1,validation_data=(splitverisi,splitcikis))
model.save("kerasileuygulama")
