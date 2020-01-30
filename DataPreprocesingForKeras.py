import cv2
import numpy as np
import os

datadir = "/Users/chekumis/Desktop/TestData/"
labels_directory = "/Users/chekumis/Desktop/Palmar/HandInfo.txt"

training_data = []


def create_training_data():

    f = open(labels_directory, 'r')

    fileList = os.listdir(datadir)

    fileLines = f.readlines()

    for filename in fileList:

        try:
            image_path = datadir + filename

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            for lineNumber in range(1, len(fileLines)):

                if filename in fileLines[lineNumber]:
                    temp_array = fileLines[lineNumber].split(',')

                    label = temp_array[0]

                    break

            training_data.append([image, int(label)])

        except Exception as e:

            pass

    f.close()


create_training_data()

print(len(training_data))


X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)


X = np.array(X).reshape(-1,360, 480,1)
y = np.array(y)

import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten, Conv2D, MaxPooling2D


X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))


X = X/255.0

model = Sequential()
model.add(Conv2D(64,(3,3),input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss = "categorical_hinge",
              optimizer = "adam",
              metrics=['accuracy'])

model.fit(X,y, batch_size = 32, epochs =3, validation_split = 0.1)


