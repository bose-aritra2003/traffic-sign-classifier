# IMPORT NECESSARY LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator

# PARAMETERS
path = "data/myData"
labelFile = "data/labels.csv"
batch_size_val = 50  # how many to process together
epochs_val = 30  # how many epochs to do
imageDimensions = (32, 32, 3)
testRatio = 0.2  # If we have 1000 images, 200 will be used for testing
validationRatio = 0.2  # 20% of the remaining 800 images will be used for validation

# Step 1: IMPORT THE IMAGES
count = 0
images = []
classNo = []
myList = os.listdir(path)  # List of all the names of the folders in the path
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes.....")
for x in range(0, len(myList)):
    myPicList = os.listdir(path + "/" + str(count))
    for y in myPicList:
        currImg = cv2.imread(path + "/" + str(count) + "/" + y)
        images.append(currImg)
        classNo.append(count)
    print(count, end=" ")
    count += 1
print(" ")
images = np.array(images)
classNo = np.array(classNo)

# Step 2: SPLIT THE DATA INTO TRAINING, TESTING AND VALIDATION
# X: image array, y: corresponding classNo
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

steps_per_epoch_val = len(X_train) // batch_size_val


# Step 3: CHECK THE NUMBER OF IMAGES MATCHES THE NUMBER OF LABELS (FOR EACH DATASET)
print("Data Shapes")
print("Train", end="")
print(X_train.shape, y_train.shape)
print("Validation", end="")
print(X_validation.shape, y_validation.shape)
print("Test", end="")
print(X_test.shape, y_test.shape)

assert (X_train.shape[0] == y_train.shape[
    0]), "The number of images in not equal to the number of lables in training set"
assert (X_validation.shape[0] == y_validation.shape[
    0]), "The number of images in not equal to the number of lables in validation set"
assert (X_test.shape[0] == y_test.shape[0]), "The number of images in not equal to the number of lables in test set"

assert (X_train.shape[1:] == imageDimensions), "The dimesions of the Training images are wrong "
assert (X_validation.shape[1:] == imageDimensions), "The dimesions of the Validation images are wrong "
assert (X_test.shape[1:] == imageDimensions), "The dimesions of the Test images are wrong "

# Step 4: READ THE CSV FILE AND GET THE NAMES OF THE CLASSES
data = pd.read_csv(labelFile)
print("data shape", data.shape, type(data))

# Step 5: DISPLAY SOME SAMPLE IMAGES OF EACH CLASS
num_of_samples = []
cols = 5
num_classes = noOfClasses
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["Name"])
            num_of_samples.append(len(x_selected))

# Step 6: DISPLAY THE DISTRIBUTION OF THE NUMBER OF IMAGES FOR EACH CLASS
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()


# Step 7: PRE-PROCESS THE IMAGES
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)  # Convert to grayscale
    img = equalize(img)  # Equalize the histogram
    img = img / 255  # Normalize the image
    return img


X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

cv2.imshow("GrayScale Images", X_train[random.randint(0, len(X_train) - 1)])  # Check if the images are pre-processed

# Step 8: ADD A DEPTH OF 1 TO THE IMAGES
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Step 9: AUGMENT THE IMAGES
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, batch_size=20)  # Generate batches of augmented images
X_batch, y_batch = next(batches)  # Get a batch of augmented images

# Step 10: SHOW AUGMENTED IMAGES
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimensions[0], imageDimensions[1]))
    axs[i].axis("off")
plt.show()

# Step 11: CONVERT THE LABELS TO ONE-HOT ENCODING
y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)


# Step 12: CREATE THE CNN MODEL
def myModel():
    # A slightly modified LeNet-5 architecture
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNodes = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0], imageDimensions[1], 1), activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))

    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    # Compile the model
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Step 13: TRAIN THE MODEL
model = myModel()
print(model.summary())
history = model.fit(
    dataGen.flow(X_train, y_train, batch_size=batch_size_val),
    steps_per_epoch=steps_per_epoch_val,
    epochs=epochs_val,
    validation_data=(X_validation, y_validation),
    shuffle=1
)


# Step 14: PLOT THE MODEL ACCURACY AND LOSS
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')

plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy = ', score[1])


# Step 15: STORE THE MODEL AS A PICKLE FILE
pickle_out = open("models/model_trained.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()

cv2.waitKey(0)
