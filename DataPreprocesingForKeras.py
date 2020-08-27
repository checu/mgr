import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt





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

print(len(set(y)))
# print(X[0].reshape(-1, 360, 480, 1))

X = np.array(X).reshape(-1,360,480,1)

# X = np.reshape(np.array(X),(-1, 360, 480, 1))

# X = np.array(X)

# X = np.array(X).reshape(-1,1,360,480)

y = np.array(y)


#transform y values to binary


encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
#
# # convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

print(len(dummy_y))

#

import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization


X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))


X = X/255.0



model = Sequential()
model.add(Conv2D(32,(3,3),input_shape = X.shape[1:], activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(128,(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(256,(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

# Fully connected layer
model.add(Flatten())
model.add(Dense(20))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(50))
# model.add(Activation("sigmoid"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#softmax clas8sifier
model.add(Dense(len(np.unique(y))))
model.add(Activation("softmax"))


model.compile(loss = 'categorical_crossentropy',
              optimizer = "adam",
              metrics=['accuracy'])

history = model.fit(X,dummy_y, batch_size = 50, epochs = 25, validation_split = 0.2)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

